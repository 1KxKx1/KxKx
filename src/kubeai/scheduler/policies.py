"""Node scoring policies.

Every policy turns ``(node, task, chosen devices)`` into a score where **higher
is better**. Filtering happens before scoring and lives in ``scheduler.py``; a
policy is only ever asked about nodes that can already satisfy the request.

Two things make this more than a toy:

* the bin-packing score is *fragmentation aware* — it looks at the memory left
  on a device after admission and penalises remainders that are too small to be
  useful again;
* the topology-aware score understands interconnect domains, so a job that wants
  four GPUs can be steered onto a single NVLink island instead of being split
  across PCIe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .model import DeviceSlot, Node, PolicyName, ResourceRequest, TaskRequest, TopologyPreference

#: Default smallest allocation worth keeping a device free for. An operator sets
#: this to the footprint of the smallest model the fleet actually serves.
FRAGMENT_FLOOR_MIB = 1024


def device_utilization_after(slot: DeviceSlot, extra_mib: int) -> float:
    total = slot.device.memory_total_mib
    if total == 0:
        return 0.0
    return min((slot.external_used_mib + slot.allocated_mib + extra_mib) / total, 1.0)


def fragmentation_penalty(
    slot: DeviceSlot, extra_mib: int, floor_mib: int = FRAGMENT_FLOOR_MIB
) -> float:
    """0.0 when the device ends up full or keeps a usable block; up to 1.0 otherwise.

    ``floor_mib`` is the smallest block that is still worth anything. Leaving less
    than that behind strands memory that no future workload can use, which is how
    a fleet ends up "80% utilised" and still unable to schedule anything.
    """
    total = slot.device.memory_total_mib
    if total == 0 or floor_mib <= 0:
        return 0.0
    remainder = total - (slot.external_used_mib + slot.allocated_mib + extra_mib)
    if remainder <= 0:
        return 0.0
    if remainder < floor_mib:
        return 1.0 - (remainder / floor_mib)
    return 0.0


class Policy(Protocol):
    name: str

    def score(self, node: Node, task: TaskRequest, devices: Sequence[DeviceSlot]) -> float: ...


@dataclass
class _BasePolicy:
    name: str = "base"
    #: Soft-affinity bonus per matching label.
    affinity_weight: float = 0.15

    def _affinity_bonus(self, node: Node, task: TaskRequest) -> float:
        if not task.affinity:
            return 0.0
        matches = sum(1 for k, v in task.affinity.items() if node.labels.get(k) == v)
        return self.affinity_weight * (matches / len(task.affinity))

    def score(self, node: Node, task: TaskRequest, devices: Sequence[DeviceSlot]) -> float:  # pragma: no cover
        raise NotImplementedError


@dataclass
class BinPackPolicy(_BasePolicy):
    """Concentrate workloads so whole nodes stay free for large jobs.

    Favours the node whose devices are already the most occupied, and rewards
    devices that end up *full* rather than half empty.
    """

    name: str = PolicyName.BINPACK.value
    #: Smallest leftover still worth keeping; below it the memory is stranded.
    fragment_floor_mib: int = FRAGMENT_FLOOR_MIB

    def score(self, node: Node, task: TaskRequest, devices: Sequence[DeviceSlot]) -> float:
        extra = task.resources.memory_mib_per_device
        fill = sum(device_utilization_after(s, extra) for s in devices) / max(len(devices), 1)
        frag = sum(
            fragmentation_penalty(s, extra, self.fragment_floor_mib) for s in devices
        ) / max(len(devices), 1)
        thermal = min(node.max_temperature / 100.0, 1.0) * 0.05
        return fill - 0.25 * frag - thermal + self._affinity_bonus(node, task)


@dataclass
class SpreadPolicy(_BasePolicy):
    """Spread workloads for resilience and to avoid noisy-neighbour effects.

    Scores the *node's* overall load, not just the device we would pick: a node
    that is already half full must not look as attractive as an empty one merely
    because it still has one untouched device.
    """

    name: str = PolicyName.SPREAD.value
    #: Weight between node-level balance and per-device fit.
    node_weight: float = 0.6

    def score(self, node: Node, task: TaskRequest, devices: Sequence[DeviceSlot]) -> float:
        extra = task.resources.memory_mib_per_device
        device_fill = sum(device_utilization_after(s, extra) for s in devices) / max(
            len(devices), 1
        )
        balance = 1.0 - node.utilization
        fit = 1.0 - device_fill
        return self.node_weight * balance + (1.0 - self.node_weight) * fit + self._affinity_bonus(
            node, task
        )


@dataclass
class TopologyAwarePolicy(_BasePolicy):
    """Bin-packing plus an interconnect-domain bonus.

    ``same-domain`` requests get a bonus when every chosen device shares one
    NVLink/HCCS island (cheap all-reduce). ``spread-domains`` requests get the
    opposite, so a fleet-wide job does not contend on a single fabric.
    """

    name: str = PolicyName.TOPOLOGY_AWARE.value
    topology_weight: float = 0.35
    fragment_floor_mib: int = FRAGMENT_FLOOR_MIB
    inner: BinPackPolicy | None = None

    def __post_init__(self) -> None:
        if self.inner is None:
            self.inner = BinPackPolicy(fragment_floor_mib=self.fragment_floor_mib)

    def score(self, node: Node, task: TaskRequest, devices: Sequence[DeviceSlot]) -> float:
        base = self.inner.score(node, task, devices)
        if task.topology is TopologyPreference.NONE or len(devices) < 2:
            return base

        domains = {s.device.interconnect_domain for s in devices}
        if task.topology is TopologyPreference.SAME_DOMAIN:
            # An unknown domain (None) means "no fast link" — treat each device
            # as its own island so the bonus is only paid on real NVLink groups.
            if None in domains:
                return base
            bonus = self.topology_weight if len(domains) == 1 else -self.topology_weight
            return base + bonus

        # spread-domains: reward the number of distinct fabrics used.
        known = {d for d in domains if d is not None}
        if not known:
            return base
        ratio = len(known) / len(devices)
        return base + self.topology_weight * ratio


def build_policy(name: str | PolicyName) -> Policy:
    key = PolicyName(name) if not isinstance(name, PolicyName) else name
    return {
        PolicyName.BINPACK: BinPackPolicy,
        PolicyName.SPREAD: SpreadPolicy,
        PolicyName.TOPOLOGY_AWARE: TopologyAwarePolicy,
    }[key]()


def device_sort_key(policy_name: str, extra_mib: int):
    """Order candidate devices inside a node.

    Bin-packing takes the *tightest* fit that still works (best-fit-decreasing),
    which is what keeps large contiguous blocks free; spreading does the reverse.
    """

    def key(slot: DeviceSlot) -> tuple[float, int]:
        free = slot.free_mib
        if policy_name == PolicyName.SPREAD.value:
            return (-free, slot.index)
        # best fit: smallest sufficient free memory first
        return (free, slot.index)

    return key


def select_devices(node: Node, request: ResourceRequest, policy_name: str) -> list[DeviceSlot] | None:
    """Pick devices on ``node`` for ``request``, or ``None`` if impossible."""
    candidates = [
        s
        for s in node.accelerators(request.device_type)
        if s.fits(request.memory_mib_per_device)
    ]
    if len(candidates) < request.count:
        return None
    if request.count == 0:
        return []
    candidates.sort(key=device_sort_key(policy_name, request.memory_mib_per_device))
    return candidates[: request.count]
