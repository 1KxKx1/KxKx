"""OpenAI-compatible inference gateway.

Serves ``/v1/chat/completions``, ``/v1/completions`` and ``/v1/embeddings`` on top
of the continuous batcher, exposes Prometheus metrics and the live accelerator
inventory, and lets models be loaded and evicted at runtime under a memory budget.

Being OpenAI-compatible is a deliberate choice: existing SDKs, evaluation
harnesses and agent frameworks can point at the platform without a custom client.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..scheduler.device import DeviceType, discover
from ..scheduler.model import Node
from ..scheduler.scheduler import Scheduler
from .batcher import GenerationRequest, QueueFull
from .metrics import ServingMetrics
from .registry import InsufficientMemory, ModelNotFound, ModelRegistry, ModelSpec

log = logging.getLogger(__name__)

DEFAULT_MODEL = "kubeai-tiny"


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | None = None
    user: str | None = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str = ""
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | None = None


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str]


class LoadModelRequest(BaseModel):
    id: str
    backend: str = "numpy"
    footprint_mib: int = 512
    max_batch_size: int = 32
    replicas: int = 1
    pinned: bool = False
    backend_options: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Prompt formatting
# --------------------------------------------------------------------------- #
def render_chat(messages: list[ChatMessage]) -> str:
    """Flatten a chat into the prompt form a decoder-only model expects."""
    parts: list[str] = []
    for message in messages:
        role = message.role or "user"
        parts.append(f"<|{role}|>{message.content}")
    parts.append("<|assistant|>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class GatewayConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    default_model: str = DEFAULT_MODEL
    default_backend: str = "numpy"
    memory_budget_mib: int = 24000
    max_batch_size: int = 32
    sim_gpus: str = "nvidia-rtx4090:4"
    nodes: int = 1
    model_dim: int = 256
    model_layers: int = 4
    model_vocab: int = 4100


def build_scheduler(config: GatewayConfig) -> tuple[Scheduler, str]:
    """Discover accelerators and wrap them in the kubeai scheduler.

    On a real host this reads ``nvidia-smi``; elsewhere it falls back to a
    simulated fleet so the gateway (and its placement decisions) still work.
    """
    from ..scheduler.device import Device, Vendor
    from ..scheduler.model import DeviceSlot

    devices, source = discover(allow_simulation=True, simulated_spec=config.sim_gpus)
    node = Node(
        name="node-00",
        labels={
            "kubernetes.io/hostname": "node-00",
            "node.kubernetes.io/instance-type": "gpu-node",
            "kubeai.io/accelerator-source": source,
        },
    )
    for device in devices:
        node.slots.append(DeviceSlot(device=device))

    if not node.slots:
        # CPU-only host: register a zero-memory slot so scheduling still answers.
        node.slots.append(
            DeviceSlot(
                device=Device(
                    index=0,
                    name="cpu-only",
                    vendor=Vendor.SIMULATED,
                    device_type=DeviceType.CPU,
                    memory_total_mib=0,
                )
            )
        )
    return Scheduler([node], policy="binpack", preemption_enabled=True), source


def create_app(config: GatewayConfig | None = None) -> FastAPI:
    config = config or GatewayConfig()
    metrics = ServingMetrics().install()
    scheduler, device_source = build_scheduler(config)
    registry = ModelRegistry(
        memory_budget_mib=config.memory_budget_mib,
        scheduler=scheduler,
        metrics=metrics,
        idle_eviction_seconds=None,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from .backends import TransformerConfig

        registry.register(
            ModelSpec(
                id=config.default_model,
                backend=config.default_backend,
                footprint_mib=64,
                max_batch_size=config.max_batch_size,
                pinned=True,
                backend_options={
                    "config": TransformerConfig(
                        vocab_size=config.model_vocab,
                        dim=config.model_dim,
                        n_layers=config.model_layers,
                    )
                }
                if config.default_backend in ("numpy", "numpy-transformer")
                else {},
            )
        )
        await registry.load(config.default_model)
        app.state.started_at = time.time()
        try:
            yield
        finally:
            await registry.shutdown()

    app = FastAPI(
        title="kubeai inference gateway",
        version="0.4.0",
        description="OpenAI-compatible inference with continuous batching, "
        "memory-budgeted model registry and GPU-aware placement.",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.metrics = metrics
    app.state.registry = registry
    app.state.scheduler = scheduler
    app.state.device_source = device_source

    # ------------------------------------------------------------------ #
    # Health / introspection
    # ------------------------------------------------------------------ #
    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "uptime_seconds": round(time.time() - getattr(app.state, "started_at", time.time()), 2),
            "accelerator_source": app.state.device_source,
            "loaded_models": sorted(app.state.registry.loaded),
        }

    @app.get("/readyz")
    async def readyz() -> dict:
        loaded = app.state.registry.loaded
        if not loaded:
            raise HTTPException(status_code=503, detail="no models loaded")
        return {"status": "ready", "models": sorted(loaded)}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus() -> str:
        return app.state.metrics.registry.render()

    @app.get("/admin/stats")
    async def stats() -> dict:
        return {
            "registry": app.state.registry.snapshot(),
            "scheduler": app.state.scheduler.snapshot(),
            "metrics": app.state.metrics.summary(),
        }

    @app.get("/admin/gpus")
    async def gpus() -> dict:
        snapshot = app.state.scheduler.snapshot()
        return {
            "source": app.state.device_source,
            "nodes": snapshot["nodes"],
            "stats": snapshot["stats"],
        }

    @app.get("/admin/queue")
    async def queue() -> dict:
        return {
            model_id: model.batcher.stats()
            for model_id, model in app.state.registry.loaded.items()
        }

    # ------------------------------------------------------------------ #
    # Model management
    # ------------------------------------------------------------------ #
    @app.get("/v1/models")
    async def list_models() -> dict:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": entry["id"],
                    "object": "model",
                    "created": now,
                    "owned_by": "kubeai",
                    "metadata": {
                        "loaded": entry["loaded"],
                        "footprint_mib": entry["footprint_mib"],
                        "backend": entry["backend"],
                    },
                }
                for entry in app.state.registry.list_models()
            ],
        }

    @app.post("/v1/models/load")
    async def load_model(payload: LoadModelRequest) -> dict:
        """Register the model if needed, then load it.

        The contract is "ensure this model is loaded", so an unknown id is not a
        404 — creating it is what this endpoint is *for*. What can fail is running
        out of memory (507) or naming a backend/option that does not exist (400).
        """
        spec = ModelSpec(
            id=payload.id,
            backend=payload.backend,
            footprint_mib=payload.footprint_mib,
            max_batch_size=payload.max_batch_size,
            replicas=payload.replicas,
            pinned=payload.pinned,
            backend_options=payload.backend_options,
        )
        app.state.registry.register(spec)
        try:
            model = await app.state.registry.load(payload.id)
        except InsufficientMemory as exc:
            raise HTTPException(status_code=507, detail=str(exc)) from exc
        except ValueError as exc:
            # An unknown backend, or an option that backend does not accept.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return model.to_dict()

    @app.delete("/v1/models/{model_id}")
    async def unload_model(model_id: str) -> dict:
        removed = await app.state.registry.unload(model_id)
        if not removed:
            raise HTTPException(status_code=404, detail=f"model {model_id!r} is not loaded")
        return {"id": model_id, "unloaded": True}

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _resolve(model_id: str | None):
        registry_: ModelRegistry = app.state.registry
        target = model_id or app.state.config.default_model
        try:
            return registry_.get(target)
        except ModelNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def _submit(model, prompt: str, payload) -> GenerationRequest:
        request = GenerationRequest(
            prompt=prompt,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            stop=payload.stop or [],
            stream=payload.stream,
            metadata={"model": model.spec.id},
        )
        try:
            model.batcher.submit_nowait(request)
        except QueueFull as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        while not request.finished:
            await asyncio.sleep(0.0005)
        return request

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request):
        model = _resolve(payload.model)
        prompt = render_chat(payload.messages)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not payload.stream:
            result = await _submit(model, prompt, payload)
            if result.finish_reason and result.finish_reason.value == "error":
                raise HTTPException(status_code=500, detail=result.error or "generation failed")
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model.spec.id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.output_text},
                        "finish_reason": result.finish_reason.value if result.finish_reason else None,
                    }
                ],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.output_tokens,
                    "total_tokens": result.prompt_tokens + result.output_tokens,
                },
                "kubeai": {
                    "latency_seconds": round(result.latency, 4),
                    "ttft_seconds": round(result.ttft, 4) if result.ttft else None,
                    "batch_size_seen": result.batch_size_seen,
                },
            }

        queue = None
        submitted = GenerationRequest(
            prompt=prompt,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            stop=payload.stop or [],
            stream=True,
            metadata={"model": model.spec.id},
        )
        queue = submitted.subscribe()
        try:
            model.batcher.submit_nowait(submitted)
        except QueueFull as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        async def event_stream() -> AsyncIterator[str]:
            first = True
            while True:
                if await request.is_disconnected():
                    submitted.cancel()
                    break
                event = await queue.get()
                if event is None:
                    break
                if event.get("done"):
                    yield "data: [DONE]\n\n"
                    break
                delta: dict[str, Any] = {"content": event["token"]}
                if first:
                    delta = {"role": "assistant", "content": event["token"]}
                    first = False
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model.spec.id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                yield f"data: {_json(chunk)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(payload: CompletionRequest):
        model = _resolve(payload.model)
        result = await _submit(model, payload.prompt, payload)
        if result.finish_reason and result.finish_reason.value == "error":
            raise HTTPException(status_code=500, detail=result.error or "generation failed")
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model.spec.id,
            "choices": [
                {
                    "index": 0,
                    "text": result.output_text,
                    "finish_reason": result.finish_reason.value if result.finish_reason else None,
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.output_tokens,
                "total_tokens": result.prompt_tokens + result.output_tokens,
            },
        }

    @app.post("/v1/embeddings")
    async def embeddings(payload: EmbeddingRequest) -> dict:
        model = _resolve(payload.model)
        inputs = [payload.input] if isinstance(payload.input, str) else payload.input
        data = []
        for index, text in enumerate(inputs):
            vector = _embed(model, text)
            data.append(
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": [round(float(x), 6) for x in vector],
                }
            )
        return {
            "object": "list",
            "data": data,
            "model": model.spec.id,
            "usage": {
                "prompt_tokens": sum(len(model.tokenizer.encode(t)) for t in inputs),
                "total_tokens": sum(len(model.tokenizer.encode(t)) for t in inputs),
            },
        }

    # ------------------------------------------------------------------ #
    # Autoscaling signal
    # ------------------------------------------------------------------ #
    @app.get("/admin/autoscaling-hint")
    async def autoscaling_hint() -> dict:
        """A queue-depth signal an HPA/KEDA ScaledObject can consume.

        Scaling on queue depth rather than CPU is the difference between a
        responsive and a permanently-lagging inference deployment.
        """
        loaded = app.state.registry.loaded
        pending = sum(len(m.batcher._waiting) for m in loaded.values())
        active = sum(len(m.batcher._active) for m in loaded.values())
        capacity = sum(m.spec.max_batch_size for m in loaded.values())
        saturation = (active / capacity) if capacity else 0.0
        if pending > 0 or saturation > 0.9:
            recommendation = "scale-up"
        elif saturation < 0.25 and pending == 0:
            recommendation = "scale-down"
        else:
            recommendation = "hold"
        return {
            "pending_requests": pending,
            "active_sequences": active,
            "batch_capacity": capacity,
            "saturation": round(saturation, 4),
            "recommendation": recommendation,
            "desired_replicas": max(1, min(10, int((active + pending) / 8) + 1)),
        }

    return app


def _json(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


def _embed(model, text: str) -> np.ndarray:
    """Mean-pooled token embeddings when the backend has them, else a stable hash.

    Real deployments point this at a dedicated embedding model; the fallback keeps
    the endpoint honest (deterministic, normalised) without pretending to be one.
    """
    tokenizer = model.tokenizer
    backend = model.backend
    table = getattr(backend, "token_embedding", None)
    if table is not None:
        ids = tokenizer.encode(text)
        vector = np.asarray(table)[ids].mean(axis=0)
    else:
        digest = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**32)
        vector = np.random.default_rng(seed).standard_normal(384)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


app = None  # populated by ``kubeai serve`` / uvicorn factory


def default_app() -> FastAPI:
    return create_app()
