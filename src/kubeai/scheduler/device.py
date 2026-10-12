"""Accelerator discovery.

Real hardware is read from the vendor tool when it is present (``nvidia-smi``
today); everywhere else the platform falls back to a deterministic simulated
fleet so that the scheduler can be developed, unit-tested and demoed on a laptop
or in CI without accelerators.

The simulated fleet is *not* a stub that always succeeds: memory, utilisation and
health are modelled per device, which is what the scheduling tests assert on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum


class DeviceType(str, Enum):
    GPU = "gpu"
    NPU = "npu"
    CPU = "cpu"


class Vendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    ASCEND = "ascend"
    INTEL = "intel"
    SIMULATED = "simulated"


_NVIDIA_QUERY = (
    "index,name,uuid,memory.total,memory.used,memory.free,"
    "utilization.gpu,temperature.gpu,compute_cap"
)


@dataclass(frozen=True)
class Device:
    """Immutable facts about one accelerator, as observed on the host."""

    index: int
    name: str
    vendor: Vendor
    device_type: DeviceType
    memory_total_mib: int
    memory_used_mib: int = 0
    utilization: int = 0
    temperature_c: int | None = None
    compute_capability: str | None = None
    uuid: str | None = None
    node: str = "local"
    healthy: bool = True
    #: Devices that share a high-bandwidth interconnect (NVLink / HCCS) get the
    #: same domain id. Used by the topology-aware scoring pass.
    interconnect_domain: str | None = None

    @property
    def memory_free_mib(self) -> int:
        return max(self.memory_total_mib - self.memory_used_mib, 0)

    @property
    def is_accelerator(self) -> bool:
        return self.device_type in (DeviceType.GPU, DeviceType.NPU)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "vendor": self.vendor.value,
            "type": self.device_type.value,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_mib": self.memory_used_mib,
            "memory_free_mib": self.memory_free_mib,
            "utilization": self.utilization,
            "temperature_c": self.temperature_c,
            "compute_capability": self.compute_capability,
            "uuid": self.uuid,
            "node": self.node,
            "healthy": self.healthy,
            "interconnect_domain": self.interconnect_domain,
        }


# --------------------------------------------------------------------------- #
# nvidia-smi
# --------------------------------------------------------------------------- #
def _parse_int(value: str) -> int | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NA"}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def discover_nvidia(timeout: float = 5.0) -> list[Device]:
    """Read the local NVIDIA fleet via ``nvidia-smi``.

    Returns an empty list when the tool is missing, fails, or reports no devices
    (a machine with the driver but no GPU, a container without the device plugin,
    and so on) — the caller then decides whether to simulate.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []

    cmd = [exe, f"--query-gpu={_NVIDIA_QUERY}", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []

    devices: list[Device] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        index = _parse_int(parts[0])
        total = _parse_int(parts[3])
        if index is None or total is None:
            continue
        devices.append(
            Device(
                index=index,
                name=parts[1],
                uuid=parts[2] or None,
                vendor=Vendor.NVIDIA,
                device_type=DeviceType.GPU,
                memory_total_mib=total,
                memory_used_mib=_parse_int(parts[4]) or 0,
                utilization=_parse_int(parts[6]) or 0,
                temperature_c=_parse_int(parts[7]),
                compute_capability=parts[8] or None,
                healthy=True,
            )
        )
    return _annotate_interconnect(devices, timeout=timeout)


def _annotate_interconnect(devices: list[Device], timeout: float = 5.0) -> list[Device]:
    """Group devices that talk over NVLink, using ``nvidia-smi topo -m``.

    Anything we cannot parse keeps its own domain, which degenerates the
    topology score to "prefer the same node" — a safe fallback.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None or len(devices) < 2:
        return devices
    try:
        proc = subprocess.run(
            [exe, "topo", "-m"], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return devices
    if proc.returncode != 0:
        return devices

    # Map each device index to the set of peers reachable over NVLink.
    peers: dict[int, set[int]] = {d.index: {d.index} for d in devices}
    header: list[str] | None = None
    for raw in proc.stdout.splitlines():
        if raw.startswith("GPU") and header is None:
            header = raw.split()
            continue
        if header is None or not raw.strip():
            continue
        cols = raw.split()
        if not cols or not cols[0].startswith("GPU"):
            continue
        try:
            src = int(cols[0].replace("GPU", ""))
        except ValueError:
            continue
        if src not in peers:
            continue
        for col_index, link in enumerate(cols[1:]):
            if not link.startswith("NV"):
                continue
            try:
                dst = int(header[col_index + 1].replace("GPU", ""))
            except (ValueError, IndexError):
                continue
            if dst in peers:
                peers[src].add(dst)
                peers[dst].add(src)

    # Union-find so a chain of NVLink pairs collapses into one domain.
    parent = {d.index: d.index for d in devices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for src, dsts in peers.items():
        for dst in dsts:
            a, b = find(src), find(dst)
            if a != b:
                parent[max(a, b)] = min(a, b)

    domains = {d.index: f"nvlink-{find(d.index)}" for d in devices}
    return [
        Device(**{**d.__dict__, "interconnect_domain": domains[d.index]})  # type: ignore[arg-type]
        for d in devices
    ]


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
_SIM_PROFILES = {
    # name: (memory_mib, interconnect_domain)
    "nvidia-a100": (40960, "nvlink-a"),
    "nvidia-h100": (81920, "nvlink-h"),
    "nvidia-rtx4090": (24564, None),
    "nvidia-rtx4060": (8188, None),
    "ascend-910b": (65536, "hccs-a"),
}


def simulate_devices(spec: str | None = None, node: str = "sim-node-0") -> list[Device]:
    """Build a deterministic simulated fleet.

    ``spec`` is ``"<profile>:<count>,<profile>:<count>,..."`` (for example
    ``"nvidia-rtx4090:2,nvidia-a100:4"``); it defaults to the ``KUBEAI_SIMULATED_GPUS``
    environment variable, and finally to two RTX 4090s.
    """
    spec = spec or os.environ.get("KUBEAI_SIMULATED_GPUS") or "nvidia-rtx4090:2"
    devices: list[Device] = []
    index = 0
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        profile, _, count_text = chunk.partition(":")
        profile = profile.strip().lower()
        try:
            count = int(count_text) if count_text else 1
        except ValueError:
            count = 1
        if profile not in _SIM_PROFILES:
            raise ValueError(f"unknown simulated profile {profile!r}; known: {sorted(_SIM_PROFILES)}")
        memory, domain = _SIM_PROFILES[profile]
        vendor = Vendor.ASCEND if profile.startswith("ascend") else Vendor.NVIDIA
        dtype = DeviceType.NPU if profile.startswith("ascend") else DeviceType.GPU
        for _ in range(max(count, 0)):
            domains = None if domain is None else f"{domain}-{index}"
            devices.append(
                Device(
                    index=index,
                    name=profile,
                    vendor=vendor,
                    device_type=dtype,
                    memory_total_mib=memory,
                    uuid=f"GPU-sim-{index:04d}",
                    node=node,
                    interconnect_domain=domains,
                )
            )
            index += 1
    return devices


def discover(
    allow_simulation: bool = True,
    simulated_spec: str | None = None,
    node: str = "local",
) -> tuple[list[Device], str]:
    """Discover accelerators, falling back to simulation.

    Returns ``(devices, source)`` where ``source`` is ``"nvidia-smi"`` or
    ``"simulated"`` so callers (and the ``/healthz`` endpoint) can be explicit
    about what they are looking at.
    """
    devices = discover_nvidia()
    if devices:
        return [Device(**{**d.__dict__, "node": node}) for d in devices], "nvidia-smi"  # type: ignore[arg-type]
    if not allow_simulation:
        return [], "none"
    return simulate_devices(simulated_spec, node=node), "simulated"
