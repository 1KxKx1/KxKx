"""GPU/NPU-aware scheduling for training and serving workloads."""

from .device import Device, DeviceType, Vendor, discover, discover_nvidia, simulate_devices
from .model import (
    DeviceSlot,
    Node,
    Placement,
    PolicyName,
    ResourceRequest,
    TaskRequest,
    TopologyPreference,
    Unscheduleable,
)
from .policies import BinPackPolicy, Policy, SpreadPolicy, TopologyAwarePolicy, build_policy
from .scheduler import Scheduler, SchedulerStats
from .simulate import (
    SimTask,
    SimulationReport,
    build_fleet,
    compare_policies,
    generate_workload,
    run_simulation,
)

__all__ = [
    "BinPackPolicy",
    "Device",
    "DeviceSlot",
    "DeviceType",
    "Node",
    "Placement",
    "Policy",
    "PolicyName",
    "ResourceRequest",
    "Scheduler",
    "SchedulerStats",
    "SimTask",
    "SimulationReport",
    "SpreadPolicy",
    "TaskRequest",
    "TopologyAwarePolicy",
    "TopologyPreference",
    "Unscheduleable",
    "Vendor",
    "build_fleet",
    "build_policy",
    "compare_policies",
    "discover",
    "discover_nvidia",
    "generate_workload",
    "run_simulation",
    "simulate_devices",
]
