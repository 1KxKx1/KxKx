"""Discrete-event simulation of the scheduler over a synthetic workload.

This is what produces the utilisation numbers in the README. It is deterministic
given a seed, so a policy change can be measured rather than argued about.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from .device import DeviceType
from .model import Node, ResourceRequest, TaskRequest, TopologyPreference
from .scheduler import Scheduler

WORKLOAD_MIX = (
    # (weight, label, gpu_count choices, memory MiB choices, duration seconds)
    (0.45, "serving", (1,), (2048, 4096, 6144), (8, 20)),
    (0.30, "finetune", (1, 2), (8192, 16384), (30, 90)),
    (0.15, "pretrain", (2, 4), (20480, 32768), (60, 180)),
    (0.10, "batch-infer", (1,), (1024,), (3, 10)),
)

PRIORITY_BY_KIND = {
    "serving": 100,      # latency sensitive, must not be starved
    "batch-infer": 20,
    "finetune": 50,
    "pretrain": 10,
}


@dataclass
class SimTask:
    """A TaskRequest plus the lifetime the simulation needs."""

    request: TaskRequest
    duration: float
    kind: str
    start: float | None = None
    finish: float | None = None


@dataclass
class SimulationReport:
    policy: str
    nodes: int
    accelerators: int
    submitted: int
    completed: int
    preempted: int
    makespan: float
    avg_utilization: float
    peak_utilization: float
    final_utilization: float
    stranded_mib: int
    pending: int
    timeline: list[dict] = field(default_factory=list)
    per_kind: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "nodes": self.nodes,
            "accelerators": self.accelerators,
            "submitted": self.submitted,
            "completed": self.completed,
            "preempted": self.preempted,
            "makespan_seconds": round(self.makespan, 2),
            "avg_utilization": round(self.avg_utilization, 4),
            "peak_utilization": round(self.peak_utilization, 4),
            "final_utilization": round(self.final_utilization, 4),
            "stranded_mib": self.stranded_mib,
            "pending": self.pending,
            "per_kind": self.per_kind,
            "timeline": self.timeline,
        }


def generate_workload(
    count: int = 200,
    seed: int = 7,
    device_type: DeviceType = DeviceType.GPU,
    max_memory_mib: int = 24564,
    max_devices: int = 4,
) -> list[SimTask]:
    """A synthetic but *feasible* workload.

    Requests are clamped to what the fleet can actually serve. Generating a task
    that no device can ever satisfy is worse than useless in a simulation: it stays
    pending forever, which silently turns "makespan" into "the time limit" and
    makes every utilisation number meaningless.
    """
    rng = random.Random(seed)
    kinds = [row[1] for row in WORKLOAD_MIX]
    weights = [row[0] for row in WORKLOAD_MIX]
    tasks: list[SimTask] = []

    for i in range(count):
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        row = next(r for r in WORKLOAD_MIX if r[1] == kind)
        _w, _label, gpu_choices, mem_choices, dur_range = row

        feasible_mem = tuple(m for m in mem_choices if m <= max_memory_mib) or (
            max(1024, max_memory_mib // 2),
        )
        feasible_count = tuple(g for g in gpu_choices if g <= max_devices) or (1,)

        topology = TopologyPreference.NONE
        if kind == "pretrain" and rng.random() < 0.7:
            topology = TopologyPreference.SAME_DOMAIN
        request = TaskRequest(
            name=f"{kind}-{i:04d}",
            resources=ResourceRequest(
                device_type=device_type,
                count=rng.choice(feasible_count),
                memory_mib_per_device=rng.choice(feasible_mem),
                cpu_millicores=rng.choice((1000, 2000, 4000)),
                host_memory_mib=rng.choice((2048, 4096, 8192)),
            ),
            priority=PRIORITY_BY_KIND[kind],
            topology=topology,
            # Training jobs are long and expensive to restart, so the simulation
            # marks them as non-preemptible; serving replicas stay evictable.
            preemptible=kind in {"serving", "batch-infer"},
            owner=f"team-{rng.randint(0, 3)}",
        )
        tasks.append(SimTask(request=request, duration=rng.uniform(*dur_range), kind=kind))
    return tasks


def build_fleet(
    nodes: int = 4,
    gpus_per_node: int = 4,
    memory_mib: int = 24564,
    device_name: str = "nvidia-rtx4090",
    interconnect: bool = True,
) -> list[Node]:
    """A homogeneous fleet. GPUs 0-1 and 2-3 share an interconnect island."""
    fleet: list[Node] = []
    for n in range(nodes):
        node = Node(
            name=f"node-{n:02d}",
            labels={
                "kubernetes.io/hostname": f"node-{n:02d}",
                "node.kubernetes.io/instance-type": "gpu-node",
                "topology.kubernetes.io/zone": f"zone-{n % 2}",
                "nvidia.com/gpu.product": device_name,
            },
        )
        from .device import Device, Vendor  # local import keeps the module import-light

        for g in range(gpus_per_node):
            domain = f"nvlink-{n}-{g // 2}" if interconnect else None
            node.slots.append(
                _slot(
                    Device(
                        index=g,
                        name=device_name,
                        vendor=Vendor.NVIDIA,
                        device_type=DeviceType.GPU,
                        memory_total_mib=memory_mib,
                        uuid=f"GPU-{n:02d}-{g:02d}",
                        node=node.name,
                        interconnect_domain=domain,
                    )
                )
            )
        fleet.append(node)
    return fleet


def _slot(device):
    from .model import DeviceSlot

    return DeviceSlot(device=device)


def run_simulation(
    tasks: list[SimTask],
    nodes: list[Node],
    policy: str = "topology-aware",
    preemption: bool = True,
    tick: float = 1.0,
    max_time: float = 2000.0,
    sample_every: int = 5,
) -> SimulationReport:
    """Run the workload through the scheduler and collect utilisation over time."""
    scheduler = Scheduler(nodes, policy=policy, preemption_enabled=preemption)

    # Stagger arrivals so the queue is not trivially full at t=0. Arrival times are
    # kept locally rather than written onto the shared TaskRequest objects: the same
    # workload is replayed under each policy, and mutating it would make the runs
    # order-dependent.
    arrival_rng = random.Random(1234)
    arrivals = sorted(tasks, key=lambda t: t.request.name)
    schedule_at = {t.request.name: arrival_rng.uniform(0, 40) for t in arrivals}
    arrivals.sort(key=lambda t: schedule_at[t.request.name])

    inflight: dict[str, SimTask] = {}
    completed = 0
    timeline: list[dict] = []
    util_samples: list[float] = []
    next_index = 0
    t = 0.0

    while t <= max_time:
        # 1. admit arrivals
        while next_index < len(arrivals) and schedule_at[arrivals[next_index].request.name] <= t:
            task = arrivals[next_index]
            task.request.created_at = t
            scheduler.add_task(task.request)
            inflight[task.request.name] = task
            next_index += 1

        # 2. finish whatever is due
        for name in [n for n, st in inflight.items() if st.finish is not None and st.finish <= t]:
            scheduler.release(name)
            del inflight[name]
            completed += 1

        # 3. place what we can
        placements, _unscheduled = scheduler.schedule_once()
        for placement in placements:
            sim = inflight.get(placement.task)
            if sim is None:
                continue
            sim.start = t
            sim.finish = t + sim.duration

        # 4. sample
        snapshot = scheduler.snapshot()
        util = snapshot["stats"]["memory_utilization"]
        util_samples.append(util)
        if len(timeline) == 0 or t % (tick * sample_every) < tick:
            timeline.append(
                {
                    "t": round(t, 1),
                    "utilization": round(util, 4),
                    "running": len(snapshot["running"]),
                    "pending": len(snapshot["pending"]),
                }
            )

        if next_index >= len(arrivals) and not inflight:
            break

        # Nothing is running and nothing new can be placed: with no allocation held
        # anywhere, no memory can ever be released, so the remaining tasks are
        # unschedulable. Spinning to max_time would report a meaningless makespan.
        if (
            not placements
            and next_index >= len(arrivals)
            and not any(s.assignments for n in nodes for s in n.slots)
        ):
            break

        t += tick

    stranded = sum(
        s.free_mib
        for n in nodes
        for s in n.accelerators()
        if 0 < s.free_mib < 1024
    )
    peak = max(util_samples) if util_samples else 0.0
    avg = sum(util_samples) / len(util_samples) if util_samples else 0.0
    final = util_samples[-1] if util_samples else 0.0

    per_kind: dict[str, int] = {}
    for task in tasks:
        per_kind[task.kind] = per_kind.get(task.kind, 0) + 1

    return SimulationReport(
        policy=scheduler.policy.name,
        nodes=len(nodes),
        accelerators=sum(len(n.accelerators()) for n in nodes),
        submitted=len(tasks),
        completed=completed,
        preempted=scheduler.stats.preempted,
        makespan=t,
        avg_utilization=avg,
        peak_utilization=peak,
        final_utilization=final,
        stranded_mib=stranded,
        pending=len(scheduler.pending),
        timeline=timeline,
        per_kind=per_kind,
    )


def compare_policies(
    tasks: list[SimTask] | None = None,
    nodes: list[Node] | None = None,
    policies: tuple[str, ...] = ("binpack", "spread", "topology-aware"),
    **kwargs,
) -> list[SimulationReport]:
    """Run the same workload under several policies — the README table comes from here.

    Each policy gets its **own copy** of the fleet. A `Scheduler` allocates directly
    on the `Node`/`DeviceSlot` objects it is given, so replaying a workload against a
    shared fleet would measure the pollution of the previous run, not the policy.
    """
    tasks = tasks or generate_workload()
    reports: list[SimulationReport] = []
    for policy in policies:
        fleet = copy.deepcopy(nodes) if nodes is not None else build_fleet()
        reports.append(run_simulation(tasks, fleet, policy=policy, **kwargs))
    return reports
