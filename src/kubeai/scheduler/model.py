"""Scheduler data model: nodes, device slots, workload requests and placements."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum

from .device import Device, DeviceType

_counter = itertools.count(1)


class TopologyPreference(str, Enum):
    """How a task wants its devices spread across interconnect domains."""

    NONE = "none"
    SAME_DOMAIN = "same-domain"
    SPREAD_DOMAINS = "spread-domains"


class PolicyName(str, Enum):
    BINPACK = "binpack"
    SPREAD = "spread"
    TOPOLOGY_AWARE = "topology-aware"


@dataclass(frozen=True)
class ResourceRequest:
    """What a task needs. ``count`` accelerators each with ``memory_mib_per_device``."""

    device_type: DeviceType = DeviceType.GPU
    count: int = 1
    memory_mib_per_device: int = 0
    cpu_millicores: int = 0
    host_memory_mib: int = 0

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be >= 0")
        if self.memory_mib_per_device < 0:
            raise ValueError("memory_mib_per_device must be >= 0")


@dataclass
class TaskRequest:
    """A unit of work asking for accelerators (a training job, a replica, ...)."""

    name: str
    resources: ResourceRequest
    priority: int = 0
    #: Hard requirement — every key/value must be present on the node.
    node_selector: dict[str, str] = field(default_factory=dict)
    #: Soft preference — matching labels add to the node score.
    affinity: dict[str, str] = field(default_factory=dict)
    tolerations: set[str] = field(default_factory=set)
    topology: TopologyPreference = TopologyPreference.NONE
    #: May this task be evicted to make room for something more important?
    preemptible: bool = True
    created_at: float = field(default_factory=time.time)
    owner: str = "default"
    #: Monotonic sequence number — breaks priority ties in FIFO order.
    sequence: int = field(default_factory=lambda: next(_counter))

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task name must not be empty")


@dataclass
class DeviceSlot:
    """A device plus the scheduler's own view of what it has handed out."""

    device: Device
    assignments: dict[str, int] = field(default_factory=dict)

    @property
    def index(self) -> int:
        return self.device.index

    @property
    def allocated_mib(self) -> int:
        return sum(self.assignments.values())

    @property
    def external_used_mib(self) -> int:
        """Memory used by processes the scheduler does not own (other tenants)."""
        return self.device.memory_used_mib

    @property
    def free_mib(self) -> int:
        return max(self.device.memory_total_mib - self.external_used_mib - self.allocated_mib, 0)

    @property
    def is_accelerator(self) -> bool:
        return self.device.is_accelerator

    def fits(self, memory_mib: int) -> bool:
        return self.device.healthy and self.free_mib >= memory_mib

    def allocate(self, task: str, memory_mib: int) -> None:
        if not self.fits(memory_mib):
            raise ValueError(
                f"device {self.index} cannot fit {memory_mib}MiB (free={self.free_mib}MiB)"
            )
        self.assignments[task] = self.assignments.get(task, 0) + memory_mib

    def release(self, task: str) -> int:
        return self.assignments.pop(task, 0)

    def utilization(self) -> float:
        if self.device.memory_total_mib == 0:
            return 0.0
        used = self.external_used_mib + self.allocated_mib
        return min(used / self.device.memory_total_mib, 1.0)


@dataclass
class Node:
    """A worker node in the fleet."""

    name: str
    slots: list[DeviceSlot] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    taints: set[str] = field(default_factory=set)
    cpu_millicores_total: int = 32_000
    cpu_millicores_allocated: int = 0
    host_memory_total_mib: int = 131_072
    host_memory_allocated_mib: int = 0
    unschedulable: bool = False

    # --- derived views ---------------------------------------------------- #
    def accelerators(self, device_type: DeviceType | None = None) -> list[DeviceSlot]:
        return [
            s
            for s in self.slots
            if s.is_accelerator and (device_type is None or s.device.device_type == device_type)
        ]

    @property
    def capacity_mib(self) -> int:
        return sum(s.device.memory_total_mib for s in self.slots if s.is_accelerator)

    @property
    def free_mib(self) -> int:
        return sum(s.free_mib for s in self.slots if s.is_accelerator)

    @property
    def allocated_mib(self) -> int:
        return sum(s.allocated_mib for s in self.slots if s.is_accelerator)

    @property
    def utilization(self) -> float:
        if self.capacity_mib == 0:
            return 0.0
        external = sum(s.external_used_mib for s in self.slots if s.is_accelerator)
        return min((external + self.allocated_mib) / self.capacity_mib, 1.0)

    @property
    def max_temperature(self) -> int:
        temps = [s.device.temperature_c for s in self.slots if s.device.temperature_c is not None]
        return max(temps) if temps else 0

    def has_capacity_for(self, request: ResourceRequest) -> bool:
        if self.cpu_millicores_total - self.cpu_millicores_allocated < request.cpu_millicores:
            return False
        if self.host_memory_total_mib - self.host_memory_allocated_mib < request.host_memory_mib:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "labels": self.labels,
            "taints": sorted(self.taints),
            "unschedulable": self.unschedulable,
            "utilization": round(self.utilization, 4),
            "accelerators": [
                {
                    "index": s.index,
                    "name": s.device.name,
                    "type": s.device.device_type.value,
                    "total_mib": s.device.memory_total_mib,
                    "allocated_mib": s.allocated_mib,
                    "free_mib": s.free_mib,
                    "utilization": round(s.utilization(), 4),
                    "tasks": sorted(s.assignments),
                }
                for s in self.slots
                if s.is_accelerator
            ],
        }


@dataclass
class Placement:
    """Result of a successful scheduling decision."""

    task: str
    node: str
    device_indices: list[int]
    memory_mib_per_device: int
    score: float
    policy: str
    scheduled_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "node": self.node,
            "devices": self.device_indices,
            "memory_mib_per_device": self.memory_mib_per_device,
            "score": round(self.score, 4),
            "policy": self.policy,
        }


@dataclass
class Unscheduleable:
    """Why a task could not be placed this round."""

    task: str
    reason: str

    def to_dict(self) -> dict:
        return {"task": self.task, "reason": self.reason}
