"""Scheduler tests: discovery, filtering, scoring, preemption and simulation."""

from __future__ import annotations

import pytest

from kubeai.scheduler.device import Device, DeviceType, Vendor, discover, simulate_devices
from kubeai.scheduler.model import (
    DeviceSlot,
    Node,
    ResourceRequest,
    TaskRequest,
    TopologyPreference,
)
from kubeai.scheduler.policies import (
    BinPackPolicy,
    SpreadPolicy,
    TopologyAwarePolicy,
    fragmentation_penalty,
    select_devices,
)
from kubeai.scheduler.scheduler import Scheduler
from kubeai.scheduler.simulate import build_fleet, compare_policies, generate_workload, run_simulation


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_node(
    name: str = "node-0",
    memory: tuple[int, ...] = (1000,),
    labels: dict[str, str] | None = None,
    taints: set[str] | None = None,
    domains: tuple[str | None, ...] | None = None,
) -> Node:
    node = Node(name=name, labels={"kubernetes.io/hostname": name, **(labels or {})},
                taints=taints or set())
    for index, total in enumerate(memory):
        domain = None if domains is None else domains[index]
        node.slots.append(
            DeviceSlot(
                device=Device(
                    index=index,
                    name="test-gpu",
                    vendor=Vendor.SIMULATED,
                    device_type=DeviceType.GPU,
                    memory_total_mib=total,
                    interconnect_domain=domain,
                    uuid=f"{name}-{index}",
                )
            )
        )
    return node


def task(name: str, memory: int = 100, count: int = 1, priority: int = 0, **kwargs) -> TaskRequest:
    return TaskRequest(
        name=name,
        resources=ResourceRequest(count=count, memory_mib_per_device=memory),
        priority=priority,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def test_simulate_devices_parses_spec():
    devices = simulate_devices("nvidia-rtx4090:2,nvidia-a100:1")
    assert len(devices) == 3
    assert [d.name for d in devices] == ["nvidia-rtx4090", "nvidia-rtx4090", "nvidia-a100"]
    assert devices[2].memory_total_mib == 40960
    assert all(d.device_type is DeviceType.GPU for d in devices)


def test_simulate_devices_rejects_unknown_profile():
    with pytest.raises(ValueError, match="unknown simulated profile"):
        simulate_devices("not-a-gpu:1")


def test_simulate_devices_npu_profile():
    devices = simulate_devices("ascend-910b:1")
    assert devices[0].device_type is DeviceType.NPU
    assert devices[0].vendor is Vendor.ASCEND


def test_discover_falls_back_to_simulation_when_no_nvidia(monkeypatch):
    monkeypatch.setattr("kubeai.scheduler.device.discover_nvidia", lambda **_: [])
    devices, source = discover(allow_simulation=True, simulated_spec="nvidia-rtx4060:2")
    assert source == "simulated"
    assert len(devices) == 2


def test_discover_returns_empty_when_simulation_disabled(monkeypatch):
    monkeypatch.setattr("kubeai.scheduler.device.discover_nvidia", lambda **_: [])
    devices, source = discover(allow_simulation=False)
    assert devices == []
    assert source == "none"


# --------------------------------------------------------------------------- #
# filtering
# --------------------------------------------------------------------------- #
def test_scheduler_places_task_and_accounts_memory():
    node = make_node(memory=(1000,))
    scheduler = Scheduler([node], policy="binpack")
    scheduler.add_task(task("a", memory=600))
    placements, unscheduled = scheduler.schedule_once()
    assert not unscheduled
    assert len(placements) == 1
    assert placements[0].device_indices == [0]
    assert node.allocated_mib == 600
    assert node.slots[0].free_mib == 400


def test_scheduler_rejects_task_that_does_not_fit():
    node = make_node(memory=(1000,))
    scheduler = Scheduler([node], policy="binpack")
    scheduler.add_task(task("big", memory=2000))
    placements, unscheduled = scheduler.schedule_once()
    assert placements == []
    assert "no 1x gpu" in unscheduled[0].reason
    assert [t.name for t in scheduler.pending] == ["big"]


def test_node_selector_is_a_hard_requirement():
    node = make_node(memory=(1000,), labels={"zone": "a"})
    scheduler = Scheduler([node], policy="binpack")
    scheduler.add_task(task("a", node_selector={"zone": "b"}))
    placements, unscheduled = scheduler.schedule_once()
    assert placements == []
    assert "missing labels" in unscheduled[0].reason


def test_taints_block_unless_tolerated():
    node = make_node(memory=(1000,), taints={"dedicated:training"})
    scheduler = Scheduler([node], policy="binpack")
    scheduler.add_task(task("a"))
    placements, _ = scheduler.schedule_once()
    assert placements == []

    scheduler.add_task(task("b", tolerations={"dedicated:training"}))
    placements, _ = scheduler.schedule_once()
    assert [p.task for p in placements] == ["b"]


def test_cordoned_node_is_skipped():
    node = make_node(memory=(1000,))
    node.unschedulable = True
    scheduler = Scheduler([node], policy="binpack")
    scheduler.add_task(task("a"))
    placements, unscheduled = scheduler.schedule_once()
    assert placements == []
    assert "cordoned" in unscheduled[0].reason


def test_multi_device_request_needs_enough_devices():
    node = make_node(memory=(1000, 1000, 1000))
    scheduler = Scheduler([node], policy="binpack")
    scheduler.add_task(task("four", count=4, memory=10))
    placements, _ = scheduler.schedule_once()
    assert placements == []  # only 3 devices exist


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #
def test_binpack_concentrates_where_spread_distributes():
    nodes = [make_node("node-a", memory=(1000, 1000)), make_node("node-b", memory=(1000, 1000))]

    binpack = Scheduler(nodes, policy="binpack")
    for i in range(2):
        binpack.add_task(task(f"bp-{i}", memory=500))
    binpack.schedule_once()
    placed = [p.node for p in binpack.running.values()]
    assert len(set(placed)) == 1, "bin-packing should fill one node before the other"

    fresh = [make_node("node-a", memory=(1000, 1000)), make_node("node-b", memory=(1000, 1000))]
    spread = Scheduler(fresh, policy="spread")
    for i in range(2):
        spread.add_task(task(f"sp-{i}", memory=500))
    spread.schedule_once()
    placed = [p.node for p in spread.running.values()]
    assert len(set(placed)) == 2, "spreading should use both nodes"


def test_fragmentation_penalty_flags_stranded_slivers():
    """The penalty exists so a device is not left with memory nothing can use."""
    slot = make_node("n", memory=(1000,)).slots[0]
    assert fragmentation_penalty(slot, 990, floor_mib=100) == pytest.approx(0.9)  # leaves 10MiB
    assert fragmentation_penalty(slot, 500, floor_mib=100) == 0.0  # leaves 500MiB
    assert fragmentation_penalty(slot, 1000, floor_mib=100) == 0.0  # leaves nothing


def test_fragmentation_penalty_is_configurable():
    """A fleet whose smallest model is 8GiB should not treat 4GiB as a sliver."""
    slot = make_node("n", memory=(24564,)).slots[0]
    assert fragmentation_penalty(slot, 24564 - 4096, floor_mib=8192) == pytest.approx(0.5)
    assert fragmentation_penalty(slot, 24564 - 4096, floor_mib=1024) == 0.0


def test_binpack_policy_penalises_stranding_memory():
    policy = BinPackPolicy(fragment_floor_mib=100)
    node = make_node("n", memory=(1000,))
    sliver = policy.score(node, task("s", memory=990), node.slots)
    # Without the fragmentation term this would simply be the fill, 0.99.
    assert sliver < 0.99


def test_binpack_prefers_the_tightest_fitting_device():
    """Best-fit keeps large contiguous blocks free for the next big job."""
    node = make_node("n", memory=(1000, 2000))
    node.slots[0].allocate("existing", 400)  # device 0 has 600MiB free
    slot = select_devices(node, ResourceRequest(count=1, memory_mib_per_device=500), "binpack")
    assert [s.index for s in slot or []] == [0]


def test_spread_prefers_the_least_loaded_node():
    policy = SpreadPolicy()
    busy = make_node("busy", memory=(1000,))
    idle = make_node("idle", memory=(1000,))
    busy.slots[0].allocate("existing", 800)
    request = task("t", memory=100)
    assert policy.score(idle, request, [idle.slots[0]]) > policy.score(busy, request, [busy.slots[0]])


def test_spread_looks_at_the_whole_node_not_one_device():
    """A half-full node must not win just because it still has an idle device."""
    policy = SpreadPolicy()
    half_full = make_node("half", memory=(1000, 1000))
    half_full.slots[0].allocate("existing", 1000)
    empty = make_node("empty", memory=(1000, 1000))
    request = task("t", memory=100)
    assert policy.score(empty, request, [empty.slots[1]]) > policy.score(
        half_full, request, [half_full.slots[1]]
    )


def test_topology_policy_prefers_a_single_nvlink_island():
    policy = TopologyAwarePolicy()
    same = make_node("same", memory=(1000, 1000), domains=("nvlink-0", "nvlink-0"))
    split = make_node("split", memory=(1000, 1000), domains=("nvlink-0", "nvlink-1"))
    request = task("t", memory=100, count=2, topology=TopologyPreference.SAME_DOMAIN)

    assert policy.score(same, request, same.slots) > policy.score(split, request, split.slots)


def test_topology_policy_ignores_unknown_domains():
    """No NVLink information must not be rewarded as if it were a fast link."""
    policy = TopologyAwarePolicy()
    unknown = make_node("unknown", memory=(1000, 1000), domains=(None, None))
    request = task("t", memory=100, count=2, topology=TopologyPreference.SAME_DOMAIN)
    base = policy.inner.score(unknown, request, unknown.slots)
    assert policy.score(unknown, request, unknown.slots) == pytest.approx(base)


def test_spread_domains_rewards_distinct_fabrics():
    policy = TopologyAwarePolicy()
    same = make_node("same", memory=(1000, 1000), domains=("nvlink-0", "nvlink-0"))
    split = make_node("split", memory=(1000, 1000), domains=("nvlink-0", "nvlink-1"))
    request = task("t", memory=100, count=2, topology=TopologyPreference.SPREAD_DOMAINS)
    assert policy.score(split, request, split.slots) > policy.score(same, request, same.slots)


# --------------------------------------------------------------------------- #
# ordering and preemption
# --------------------------------------------------------------------------- #
def test_higher_priority_is_scheduled_first():
    node = make_node(memory=(1000,))
    scheduler = Scheduler([node], policy="binpack")
    scheduler.add_task(task("low", memory=900, priority=1))
    scheduler.add_task(task("high", memory=900, priority=100))
    placements, unscheduled = scheduler.schedule_once()
    assert [p.task for p in placements] == ["high"]
    assert [t.name for t in scheduler.pending] == ["low"]


def test_same_priority_is_fifo():
    node = make_node(memory=(1000,))
    scheduler = Scheduler([node], policy="binpack")
    scheduler.add_task(task("first", memory=900))
    scheduler.add_task(task("second", memory=900))
    placements, _ = scheduler.schedule_once()
    assert [p.task for p in placements] == ["first"]


def test_preemption_evicts_lower_priority_work():
    node = make_node(memory=(1000,))
    scheduler = Scheduler([node], policy="binpack", preemption_enabled=True)
    scheduler.add_task(task("batch", memory=800, priority=10))
    scheduler.schedule_once()
    assert "batch" in scheduler.running

    scheduler.add_task(task("serving", memory=800, priority=100))
    placements, _ = scheduler.schedule_once()

    assert [p.task for p in placements] == ["serving"]
    assert scheduler.stats.preempted == 1
    assert "batch" not in scheduler.running
    # The victim is requeued rather than dropped.
    assert "batch" in [t.name for t in scheduler.pending]


def test_preemption_respects_non_preemptible_work():
    node = make_node(memory=(1000,))
    scheduler = Scheduler([node], policy="binpack", preemption_enabled=True)
    scheduler.add_task(task("training", memory=800, priority=10, preemptible=False))
    scheduler.schedule_once()

    scheduler.add_task(task("serving", memory=800, priority=100))
    placements, unscheduled = scheduler.schedule_once()

    assert placements == []
    assert "training" in scheduler.running
    assert scheduler.stats.preempted == 0
    assert [u.task for u in unscheduled] == ["serving"]


def test_preemption_can_be_disabled():
    node = make_node(memory=(1000,))
    scheduler = Scheduler([node], policy="binpack", preemption_enabled=False)
    scheduler.add_task(task("batch", memory=800, priority=10))
    scheduler.schedule_once()
    scheduler.add_task(task("serving", memory=800, priority=100))
    placements, _ = scheduler.schedule_once()
    assert placements == []


def test_release_frees_capacity_for_the_next_task():
    node = make_node(memory=(1000,))
    scheduler = Scheduler([node], policy="binpack")
    # Equal priority: preemption requires a *strictly* lower-priority victim, so
    # this isolates release() from the preemption path.
    scheduler.add_task(task("a", memory=900, priority=5))
    scheduler.schedule_once()
    scheduler.add_task(task("b", memory=900, priority=5))
    placements, _ = scheduler.schedule_once()
    assert placements == []
    assert scheduler.stats.preempted == 0

    scheduler.release("a")
    assert node.allocated_mib == 0
    placements, _ = scheduler.schedule_once()
    assert [p.task for p in placements] == ["b"]


def test_release_returns_cpu_and_host_memory_too():
    """A leak here is silent: accelerators look free while the node stops accepting work."""
    node = make_node(memory=(4000,))
    scheduler = Scheduler([node], policy="binpack")
    greedy = task("greedy", memory=100, priority=5)
    greedy.resources = ResourceRequest(
        count=1, memory_mib_per_device=100, cpu_millicores=20_000, host_memory_mib=60_000
    )
    scheduler.add_task(greedy)
    scheduler.schedule_once()
    assert node.cpu_millicores_allocated == 20_000
    assert node.host_memory_allocated_mib == 60_000

    scheduler.release("greedy")
    assert node.cpu_millicores_allocated == 0
    assert node.host_memory_allocated_mib == 0

    # The node must be usable again, not just its GPUs.
    scheduler.add_task(task("next", memory=100))
    placements, _ = scheduler.schedule_once()
    assert [p.task for p in placements] == ["next"]


def test_cpu_starved_node_is_rejected():
    node = make_node(memory=(4000,))
    node.cpu_millicores_allocated = node.cpu_millicores_total
    scheduler = Scheduler([node], policy="binpack")
    hungry = task("cpu-hungry", memory=100)
    hungry.resources = ResourceRequest(count=1, memory_mib_per_device=100, cpu_millicores=1000)
    scheduler.add_task(hungry)
    placements, unscheduled = scheduler.schedule_once()
    assert placements == []
    assert "lacks cpu" in unscheduled[0].reason


def test_running_task_cannot_be_submitted_twice():
    node = make_node(memory=(1000,))
    scheduler = Scheduler([node], policy="binpack")
    scheduler.add_task(task("a", memory=100))
    scheduler.schedule_once()
    with pytest.raises(ValueError, match="already running"):
        scheduler.add_task(task("a", memory=100))


def test_snapshot_reports_utilisation():
    node = make_node(memory=(1000, 1000))
    scheduler = Scheduler([node], policy="binpack")
    scheduler.add_task(task("a", memory=1000, count=1))
    scheduler.schedule_once()
    snapshot = scheduler.snapshot()
    assert snapshot["stats"]["accelerators_total"] == 2
    assert snapshot["stats"]["accelerators_used"] == 1
    assert snapshot["stats"]["memory_utilization"] == pytest.approx(0.5)
    assert snapshot["policy"] == "binpack"


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #
def test_fleet_shape_and_interconnect_domains():
    fleet = build_fleet(nodes=2, gpus_per_node=4, interconnect=True)
    assert len(fleet) == 2
    assert len(fleet[0].accelerators()) == 4
    domains = {s.device.interconnect_domain for s in fleet[0].slots}
    # 4 GPUs grouped in NVLink pairs -> two islands per node.
    assert len(domains) == 2


def test_workload_generation_is_deterministic():
    first = generate_workload(count=20, seed=3)
    second = generate_workload(count=20, seed=3)
    assert [t.request.name for t in first] == [t.request.name for t in second]
    assert [t.request.resources.memory_mib_per_device for t in first] == [
        t.request.resources.memory_mib_per_device for t in second
    ]


def test_simulation_completes_and_reports_utilisation():
    tasks = generate_workload(count=40, seed=11)
    nodes = build_fleet(nodes=2, gpus_per_node=4)
    report = run_simulation(tasks, nodes, policy="topology-aware", max_time=600)
    assert report.submitted == 40
    assert report.completed > 0
    assert 0.0 <= report.avg_utilization <= 1.0
    assert report.peak_utilization >= report.avg_utilization
    assert report.accelerators == 8


def test_policy_comparison_returns_one_report_each():
    reports = compare_policies(
        generate_workload(count=20, seed=5), build_fleet(nodes=1, gpus_per_node=4),
        max_time=300,
    )
    assert {r.policy for r in reports} == {"binpack", "spread", "topology-aware"}
    for report in reports:
        assert report.submitted == 20
