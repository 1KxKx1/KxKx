"""Model registry with a memory budget and LRU eviction.

An inference platform is limited by accelerator *memory*, not by model count, so
loading is a bin-packing problem: a new model may require evicting idle ones. The
registry makes that explicit — every model declares its footprint, and the pool
either fits it or evicts least-recently-used models until it does.

When a :class:`~kubeai.scheduler.scheduler.Scheduler` is supplied, each load is
also *placed* on a node, which is what ties the serving path to the GPU inventory
instead of assuming one big pool.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..scheduler.model import ResourceRequest, TaskRequest
from ..scheduler.scheduler import Scheduler
from .backends import Backend, CharTokenizer, build_backend
from .batcher import ContinuousBatcher
from .metrics import ServingMetrics

log = logging.getLogger(__name__)


class ModelNotFound(KeyError):
    pass


class InsufficientMemory(RuntimeError):
    """Raised when a model cannot be loaded even after evicting everything idle."""


@dataclass
class ModelSpec:
    """Declared resources for one servable model."""

    id: str
    backend: str = "numpy"
    footprint_mib: int = 512
    max_batch_size: int = 32
    max_batch_tokens: int = 4096
    replicas: int = 1
    labels: dict[str, str] = field(default_factory=dict)
    backend_options: dict[str, Any] = field(default_factory=dict)
    #: Keep resident across evictions (e.g. the model behind the main endpoint).
    pinned: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "backend": self.backend,
            "footprint_mib": self.footprint_mib,
            "max_batch_size": self.max_batch_size,
            "replicas": self.replicas,
            "pinned": self.pinned,
            "labels": self.labels,
        }


@dataclass
class LoadedModel:
    spec: ModelSpec
    backend: Backend
    batcher: ContinuousBatcher
    tokenizer: CharTokenizer
    loaded_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    node: str | None = None
    devices: list[int] = field(default_factory=list)

    def touch(self) -> None:
        self.last_used_at = time.time()

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used_at

    def to_dict(self) -> dict:
        return {
            **self.spec.to_dict(),
            "loaded_at": self.loaded_at,
            "idle_seconds": round(self.idle_seconds, 2),
            "node": self.node,
            "devices": self.devices,
            "batcher": self.batcher.stats(),
        }


class ModelRegistry:
    def __init__(
        self,
        memory_budget_mib: int = 8192,
        scheduler: Scheduler | None = None,
        metrics: ServingMetrics | None = None,
        idle_eviction_seconds: float | None = None,
    ) -> None:
        self.memory_budget_mib = memory_budget_mib
        self.scheduler = scheduler
        self.metrics = metrics
        self.idle_eviction_seconds = idle_eviction_seconds
        self._specs: dict[str, ModelSpec] = {}
        self._loaded: dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()
        self.evictions = 0

    # ------------------------------------------------------------------ #
    def register(self, spec: ModelSpec) -> ModelSpec:
        self._specs[spec.id] = spec
        return spec

    def register_many(self, specs: list[ModelSpec]) -> None:
        for spec in specs:
            self.register(spec)

    @property
    def specs(self) -> dict[str, ModelSpec]:
        return dict(self._specs)

    @property
    def loaded(self) -> dict[str, LoadedModel]:
        return dict(self._loaded)

    @property
    def used_mib(self) -> int:
        return sum(m.spec.footprint_mib for m in self._loaded.values())

    @property
    def available_mib(self) -> int:
        return max(self.memory_budget_mib - self.used_mib, 0)

    def get(self, model_id: str) -> LoadedModel:
        model = self._loaded.get(model_id)
        if model is None:
            raise ModelNotFound(f"model {model_id!r} is not loaded")
        model.touch()
        return model

    def list_models(self) -> list[dict]:
        out: list[dict] = []
        for spec in self._specs.values():
            entry = spec.to_dict()
            entry["loaded"] = spec.id in self._loaded
            out.append(entry)
        return sorted(out, key=lambda item: item["id"])

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _eviction_candidates(self, needed_mib: int) -> list[LoadedModel]:
        """Idle, unpinned models, least-recently-used first."""
        return sorted(
            (m for m in self._loaded.values() if not m.spec.pinned and m.batcher.is_idle),
            key=lambda m: m.last_used_at,
        )

    def _place_on_node(self, spec: ModelSpec) -> tuple[str | None, list[int]]:
        """Ask the scheduler for accelerator memory; best-effort."""
        if self.scheduler is None:
            return None, []
        task = TaskRequest(
            name=f"serve-{spec.id}",
            resources=ResourceRequest(
                count=spec.replicas,
                memory_mib_per_device=max(spec.footprint_mib // max(spec.replicas, 1), 1),
                cpu_millicores=1000,
                host_memory_mib=2048,
            ),
            # Serving outranks batch work; `owner` is what marks the workload class
            # (TaskRequest has no free-form labels — its selectors are node_selector
            # and affinity, which match against *node* labels).
            priority=100,
            preemptible=True,
            owner="serving",
        )
        self.scheduler.add_task(task)
        placements, _unscheduled = self.scheduler.schedule_once()
        if not placements:
            return None, []
        placement = placements[0]
        return placement.node, placement.device_indices

    def _release_from_node(self, spec: ModelSpec) -> None:
        if self.scheduler is None:
            return
        self.scheduler.release(f"serve-{spec.id}")

    async def load(self, model_id: str) -> LoadedModel:
        async with self._lock:
            if model_id in self._loaded:
                return self._loaded[model_id]

            spec = self._specs.get(model_id)
            if spec is None:
                raise ModelNotFound(f"model {model_id!r} is not registered")

            if spec.footprint_mib > self.memory_budget_mib:
                raise InsufficientMemory(
                    f"model {model_id!r} needs {spec.footprint_mib}MiB but the pool is "
                    f"{self.memory_budget_mib}MiB"
                )

            while self.available_mib < spec.footprint_mib:
                candidates = self._eviction_candidates(spec.footprint_mib)
                if not candidates:
                    raise InsufficientMemory(
                        f"cannot load {model_id!r}: need {spec.footprint_mib}MiB, "
                        f"{self.available_mib}MiB free, nothing evictable"
                    )
                victim = candidates[0]
                log.info(
                    "evicting %s (idle %.1fs) to make room for %s",
                    victim.spec.id,
                    victim.idle_seconds,
                    model_id,
                )
                await self._unload_locked(victim.spec.id)

            backend = build_backend(spec.backend, **spec.backend_options)
            tokenizer = getattr(backend, "tokenizer", None) or CharTokenizer()
            batcher = ContinuousBatcher(
                backend,
                tokenizer=tokenizer,
                max_batch_size=spec.max_batch_size,
                max_batch_tokens=spec.max_batch_tokens,
                metrics=self.metrics,
            )
            await batcher.start()

            node, devices = self._place_on_node(spec)
            model = LoadedModel(
                spec=spec,
                backend=backend,
                batcher=batcher,
                tokenizer=tokenizer,
                node=node,
                devices=devices,
            )
            self._loaded[model_id] = model
            log.info(
                "loaded %s (backend=%s footprint=%dMiB node=%s devices=%s)",
                model_id,
                spec.backend,
                spec.footprint_mib,
                node,
                devices,
            )
            return model

    async def _unload_locked(self, model_id: str) -> bool:
        model = self._loaded.pop(model_id, None)
        if model is None:
            return False
        await model.batcher.stop()
        self._release_from_node(model.spec)
        self.evictions += 1
        log.info("unloaded %s", model_id)
        return True

    async def unload(self, model_id: str) -> bool:
        async with self._lock:
            return await self._unload_locked(model_id)

    async def shutdown(self) -> None:
        async with self._lock:
            for model_id in list(self._loaded):
                await self._unload_locked(model_id)

    # ------------------------------------------------------------------ #
    async def reap_idle(self) -> list[str]:
        """Evict models idle beyond ``idle_eviction_seconds`` (called periodically)."""
        if self.idle_eviction_seconds is None:
            return []
        evicted: list[str] = []
        async with self._lock:
            for model in list(self._loaded.values()):
                if model.spec.pinned:
                    continue
                if model.idle_seconds < self.idle_eviction_seconds:
                    continue
                if model.batcher.is_idle and await self._unload_locked(model.spec.id):
                    evicted.append(model.spec.id)
        return evicted

    def snapshot(self) -> dict:
        return {
            "memory_budget_mib": self.memory_budget_mib,
            "used_mib": self.used_mib,
            "available_mib": self.available_mib,
            "utilization": round(self.used_mib / self.memory_budget_mib, 4)
            if self.memory_budget_mib
            else 0.0,
            "evictions": self.evictions,
            "registered": len(self._specs),
            "loaded": {k: v.to_dict() for k, v in self._loaded.items()},
        }
