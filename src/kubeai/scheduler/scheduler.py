"""The scheduling loop: filter, score, bind, and preempt when nothing fits.

The loop mirrors the shape of the Kubernetes scheduler — *filter* nodes that
cannot satisfy a request, *score* the survivors, *bind* the best one — with two
differences that matter for AI workloads:

* the unit of allocation is accelerator memory, not whole devices, so several
  small models share one card;
* preemption is a first-class path: a high-priority job that does not fit evicts
  lower-priority work, which is how training jobs make progress on a busy fleet.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from .device import DeviceType
from .model import (
    DeviceSlot,
    Node,
    Placement,
    ResourceRequest,
    TaskRequest,
    Unscheduleable,
)
from .policies import Policy, build_policy, device_utilization_after, select_devices

log = logging.getLogger(__name__)


@dataclass
class SchedulerStats:
    scheduled: int = 0
    preempted: int = 0
    failed: int = 0
    evicted_tasks: list[str] = field(default_factory=list)
    schedule_rounds: int = 0
    total_wait_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "scheduled": self.scheduled,
            "preempted": self.preempted,
            "failed": self.failed,
            "schedule_rounds": self.schedule_rounds,
            "avg_wait_seconds": round(
                self.total_wait_seconds / self.scheduled, 3
            )
            if self.scheduled
            else 0.0,
            "evicted_tasks": list(self.evicted_tasks),
        }


class Scheduler:
    """A single-cycle (non-blocking) accelerator scheduler."""

    def __init__(
        self,
        nodes: list[Node],
        policy: str | Policy = "topology-aware",
        preemption_enabled: bool = True,
    ) -> None:
        if not nodes:
            raise ValueError("at least one node is required")
        self.nodes = nodes
        self.policy: Policy = policy if not isinstance(policy, str) else build_policy(policy)
        self.preemption_enabled = preemption_enabled
        self.stats = SchedulerStats()

        self._pending: list[TaskRequest] = []
        self._running: dict[str, Placement] = {}
        self._tasks: dict[str, TaskRequest] = {}

    # ------------------------------------------------------------------ #
    # Queue management
    # ------------------------------------------------------------------ #
    def add_task(self, task: TaskRequest) -> None:
        if task.name in self._running:
            raise ValueError(f"task {task.name!r} is already running")
        self._pending.append(task)
        self._tasks[task.name] = task

    def add_tasks(self, tasks: list[TaskRequest]) -> None:
        for task in tasks:
            self.add_task(task)

    @property
    def pending(self) -> list[TaskRequest]:
        return list(self._pending)

    @property
    def running(self) -> dict[str, Placement]:
        return dict(self._running)

    def release(self, task_name: str) -> None:
        """Free everything ``task_name`` holds.

        Every resource the task was charged for must come back: accelerator memory
        *and* the node's CPU and host memory. Getting this wrong is subtle — the
        accelerator frees up, so tests and small runs look fine, while the node's
        CPU budget leaks until nothing can be scheduled onto it at all.
        """
        task = self._tasks.pop(task_name, None)
        placement = self._running.pop(task_name, None)
        if placement is None:
            return
        node = self._node(placement.node)
        for index in placement.device_indices:
            slot = self._slot(node, index)
            if slot is not None:
                slot.release(task_name)
        if task is not None:
            node.cpu_millicores_allocated = max(
                node.cpu_millicores_allocated - task.resources.cpu_millicores, 0
            )
            node.host_memory_allocated_mib = max(
                node.host_memory_allocated_mib - task.resources.host_memory_mib, 0
            )

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #
    def _node(self, name: str) -> Node:
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(name)

    @staticmethod
    def _slot(node: Node, index: int) -> DeviceSlot | None:
        for slot in node.slots:
            if slot.index == index:
                return slot
        return None

    def _reject_reason(self, node: Node, task: TaskRequest) -> str | None:
        """Return why ``node`` is ineligible, or ``None`` when it survives filtering."""
        if node.unschedulable:
            return f"node {node.name} is cordoned"
        missing = [k for k, v in task.node_selector.items() if node.labels.get(k) != v]
        if missing:
            return f"node {node.name} missing labels {missing}"
        blocking = sorted(node.taints - task.tolerations)
        if blocking:
            return f"node {node.name} has untolerated taints {blocking}"
        if not node.has_capacity_for(task.resources):
            return f"node {node.name} lacks cpu/host memory"
        if task.resources.device_type not in (DeviceType.GPU, DeviceType.NPU, DeviceType.CPU):
            return f"unsupported device type {task.resources.device_type}"
        if select_devices(node, task.resources, self.policy.name) is None:
            return (
                f"node {node.name} has no {task.resources.count}x "
                f"{task.resources.device_type.value} with "
                f"{task.resources.memory_mib_per_device}MiB free"
            )
        return None

    def feasible_nodes(self, task: TaskRequest) -> list[Node]:
        return [n for n in self.nodes if self._reject_reason(n, task) is None]

    # ------------------------------------------------------------------ #
    # Preemption
    # ------------------------------------------------------------------ #
    def _preemption_victims(self, node: Node, task: TaskRequest) -> list[str] | None:
        """Tasks to evict from ``node`` so ``task`` fits, or ``None``.

        A victim must be strictly lower priority *and* preemptible. We search for
        the cheapest eviction set (fewest tasks, then lowest total priority) so a
        small job does not blow away a whole fleet to make room.
        """
        if not self.preemption_enabled or not task.preemptible:
            # ``preemptible`` on the *incoming* task describes whether it may be
            # evicted later; preemption of others depends on their own flags.
            if not self.preemption_enabled:
                return None

        request = task.resources
        needed = request.memory_mib_per_device
        victims: set[str] = set()
        freed_devices = 0

        for slot in node.accelerators(request.device_type):
            if not slot.device.healthy:
                continue
            if slot.fits(needed):
                freed_devices += 1
                continue
            # Try to free this device by evicting lower-priority work.
            evictable: list[tuple[int, int, str]] = []
            for name, held in slot.assignments.items():
                other = self._tasks.get(name)
                if other is None or not other.preemptible:
                    continue
                if other.priority >= task.priority:
                    continue
                evictable.append((other.priority, held, name))
            # Lowest priority first, then largest footprint (frees memory fastest).
            evictable.sort(key=lambda item: (item[0], -item[1]))
            freed = 0
            picked: list[str] = []
            for _priority, held, name in evictable:
                if slot.free_mib + freed >= needed:
                    break
                freed += held
                picked.append(name)
            if slot.free_mib + freed >= needed:
                victims.update(picked)
                freed_devices += 1
            if freed_devices >= request.count:
                break

        if freed_devices < request.count:
            return None
        return sorted(victims)

    def _preempt(self, node: Node, task: TaskRequest, victims: list[str]) -> None:
        for name in victims:
            victim_placement = self._running.get(name)
            victim_task = self._tasks.get(name)
            if victim_placement is None:
                continue
            log.info(
                "preempting %s (priority=%s) on %s for %s (priority=%s)",
                name,
                victim_task.priority if victim_task else "?",
                node.name,
                task.name,
                task.priority,
            )
            self.release(name)
            self.stats.preempted += 1
            self.stats.evicted_tasks.append(name)
            if victim_task is not None:
                # Requeue so it can be rescheduled once capacity returns.
                victim_task.created_at = time.time()
                self._pending.append(victim_task)
                self._tasks[victim_task.name] = victim_task

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def schedule_once(self) -> tuple[list[Placement], list[Unscheduleable]]:
        """Run one scheduling pass over the pending queue.

        Tasks are considered in priority order (highest first), FIFO within a
        priority band, so a big training job is not starved by a stream of small
        serving replicas.
        """
        self.stats.schedule_rounds += 1
        placements: list[Placement] = []
        unscheduled: list[Unscheduleable] = []

        queue = sorted(self._pending, key=lambda t: (-t.priority, t.sequence))
        self._pending = []

        for task in queue:
            if task.name in self._running:
                continue
            placement = self._try_place(task)
            if placement is not None:
                placements.append(placement)
            else:
                reason = self._explain_failure(task)
                unscheduled.append(Unscheduleable(task=task.name, reason=reason))
                self._pending.append(task)
                self.stats.failed += 1

        return placements, unscheduled

    def _try_place(self, task: TaskRequest) -> Placement | None:
        feasible = self.feasible_nodes(task)
        scored = self._score(feasible, task)

        if not scored:
            if not self.preemption_enabled:
                return None
            for node in self._preemption_candidates(task):
                victims = self._preemption_victims(node, task)
                if victims is None:
                    continue
                self._preempt(node, task, victims)
                rescored = self._score([node], task)
                if rescored:
                    return self._bind(rescored[0][0], rescored[0][1], task)
            return None

        return self._bind(scored[0][0], scored[0][1], task)

    def _preemption_candidates(self, task: TaskRequest) -> list[Node]:
        """Nodes worth trying to preempt on, most-occupied first.

        Preempting is disruptive, so we only consider nodes where the request is
        at least *plausible*: right accelerator type, enough total memory, and
        something to evict.
        """
        candidates: list[Node] = []
        for node in self.nodes:
            if node.unschedulable or not node.has_capacity_for(task.resources):
                continue
            missing = [k for k, v in task.node_selector.items() if node.labels.get(k) != v]
            if missing or (node.taints - task.tolerations):
                continue
            accels = node.accelerators(task.resources.device_type)
            if len(accels) < task.resources.count:
                continue
            if not any(s.assignments for s in accels):
                continue
            candidates.append(node)
        candidates.sort(key=lambda n: n.utilization, reverse=True)
        return candidates

    def _score(self, nodes: list[Node], task: TaskRequest) -> list[tuple[Node, list[DeviceSlot]]]:
        scored: list[tuple[float, Node, list[DeviceSlot]]] = []
        for node in nodes:
            devices = select_devices(node, task.resources, self.policy.name)
            if devices is None:
                continue
            if not devices and task.resources.count > 0:
                continue
            score = self.policy.score(node, task, devices)
            scored.append((score, node, devices))
        # Highest score wins; node name breaks ties so runs are reproducible.
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [(node, devices) for _s, node, devices in scored]

    def _bind(self, node: Node, devices: list[DeviceSlot], task: TaskRequest) -> Placement:
        request = task.resources
        for slot in devices:
            slot.allocate(task.name, request.memory_mib_per_device)
        node.cpu_millicores_allocated += request.cpu_millicores
        node.host_memory_allocated_mib += request.host_memory_mib

        devices_ = list(devices)
        score = self.policy.score(node, task, devices_)
        placement = Placement(
            task=task.name,
            node=node.name,
            device_indices=[s.index for s in devices_],
            memory_mib_per_device=request.memory_mib_per_device,
            score=score,
            policy=self.policy.name,
        )
        self._running[task.name] = placement
        self.stats.scheduled += 1
        self.stats.total_wait_seconds += max(time.time() - task.created_at, 0.0)
        return placement

    def _explain_failure(self, task: TaskRequest) -> str:
        reasons = [self._reject_reason(n, task) for n in self.nodes]
        reasons = [r for r in reasons if r]
        if not reasons:
            return "no nodes registered"
        # Surface the most common cause; that is what an operator needs to fix.
        counts: dict[str, int] = defaultdict(int)
        for reason in reasons:
            counts[reason.split(" with ")[0]] += 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        accel_total = sum(len(n.accelerators()) for n in self.nodes)
        used = sum(
            len([s for s in n.accelerators() if s.assignments]) for n in self.nodes
        )
        return {
            "policy": self.policy.name,
            "preemption_enabled": self.preemption_enabled,
            "nodes": [n.to_dict() for n in self.nodes],
            "pending": [t.name for t in self._pending],
            "running": {k: v.to_dict() for k, v in self._running.items()},
            "stats": {
                **self.stats.to_dict(),
                "accelerators_total": accel_total,
                "accelerators_used": used,
                "accelerator_utilization": round(used / accel_total, 4) if accel_total else 0.0,
                "memory_utilization": round(
                    sum(n.utilization for n in self.nodes) / len(self.nodes), 4
                ),
            },
        }
