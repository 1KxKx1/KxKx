"""A tiny Prometheus-compatible metrics registry.

Hand-rolled rather than pulling in ``prometheus_client`` so the exposition format
is testable as plain strings and the serving path stays dependency-light. It
implements the parts that matter here: counters, gauges, histograms with labels,
and text exposition.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

DEFAULT_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_labels(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape(str(v))}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, help_text: str, label_names: tuple[str, ...] = ()) -> None:
        self.name = name
        self.help = help_text
        self.label_names = label_names
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, str] | None) -> tuple:
        if not self.label_names:
            return ()
        labels = labels or {}
        missing = [n for n in self.label_names if n not in labels]
        if missing:
            raise ValueError(f"metric {self.name} missing labels {missing}")
        return tuple(str(labels[n]) for n in self.label_names)

    def _label_dict(self, key: tuple) -> dict[str, str]:
        return dict(zip(self.label_names, key, strict=True))


class Counter(_Metric):
    kind = "counter"

    def __init__(self, name: str, help_text: str, label_names: tuple[str, ...] = ()) -> None:
        super().__init__(name, help_text, label_names)
        self._values: dict[tuple, float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        return self._values.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            items = sorted(self._values.items())
        for key, value in items:
            lines.append(f"{self.name}{_format_labels(self._label_dict(key))} {value}")
        if not items:
            lines.append(f"{self.name} 0")
        return lines


class Gauge(_Metric):
    kind = "gauge"

    def __init__(self, name: str, help_text: str, label_names: tuple[str, ...] = ()) -> None:
        super().__init__(name, help_text, label_names)
        self._values: dict[tuple, float] = {}

    def set(self, value: float, **labels: str) -> None:
        with self._lock:
            self._values[self._key(labels)] = float(value)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        with self._lock:
            key = self._key(labels)
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        self.inc(-amount, **labels)

    def value(self, **labels: str) -> float:
        return self._values.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        with self._lock:
            items = sorted(self._values.items())
        for key, value in items:
            lines.append(f"{self.name}{_format_labels(self._label_dict(key))} {value}")
        return lines


@dataclass
class Histogram(_Metric):
    kind = "histogram"
    buckets: tuple[float, ...] = DEFAULT_BUCKETS

    def __init__(
        self,
        name: str,
        help_text: str,
        label_names: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    ) -> None:
        super().__init__(name, help_text, label_names)
        self.buckets = tuple(sorted(buckets))
        self._counts: dict[tuple, list[int]] = {}
        self._sums: dict[tuple, float] = {}
        self._totals: dict[tuple, int] = {}

    def observe(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    counts[i] += 1
            counts[-1] += 1  # +Inf
            self._sums[key] = self._sums.get(key, 0.0) + value
            self._totals[key] = self._totals.get(key, 0) + 1

    def snapshot(self, **labels: str) -> dict[str, float]:
        key = self._key(labels)
        total = self._totals.get(key, 0)
        return {
            "count": total,
            "sum": self._sums.get(key, 0.0),
            "avg": (self._sums.get(key, 0.0) / total) if total else 0.0,
        }

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            keys = sorted(self._counts)
        for key in keys:
            label_dict = self._label_dict(key)
            counts = self._counts[key]
            cumulative = 0
            for i, bound in enumerate(self.buckets):
                cumulative = counts[i]
                labels = {**label_dict, "le": _num(bound)}
                lines.append(f"{self.name}_bucket{_format_labels(labels)} {cumulative}")
            labels = {**label_dict, "le": "+Inf"}
            lines.append(f"{self.name}_bucket{_format_labels(labels)} {counts[-1]}")
            lines.append(
                f"{self.name}_sum{_format_labels(label_dict)} {self._sums.get(key, 0.0)}"
            )
            lines.append(
                f"{self.name}_count{_format_labels(label_dict)} {self._totals.get(key, 0)}"
            )
        return lines


def _num(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return repr(value)


class MetricsRegistry:
    """Holds metrics and renders the Prometheus text exposition format."""

    def __init__(self, namespace: str = "kubeai") -> None:
        self.namespace = namespace
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def _register(self, metric: _Metric) -> _Metric:
        with self._lock:
            existing = self._metrics.get(metric.name)
            if existing is not None:
                return existing
            self._metrics[metric.name] = metric
        return metric

    def counter(self, name: str, help_text: str, label_names: tuple[str, ...] = ()) -> Counter:
        return self._register(Counter(f"{self.namespace}_{name}", help_text, label_names))  # type: ignore[return-value]

    def gauge(self, name: str, help_text: str, label_names: tuple[str, ...] = ()) -> Gauge:
        return self._register(Gauge(f"{self.namespace}_{name}", help_text, label_names))  # type: ignore[return-value]

    def histogram(
        self,
        name: str,
        help_text: str,
        label_names: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    ) -> Histogram:
        return self._register(  # type: ignore[return-value]
            Histogram(f"{self.namespace}_{name}", help_text, label_names, buckets)
        )

    def render(self) -> str:
        with self._lock:
            metrics = list(self._metrics.values())
        lines: list[str] = []
        for metric in metrics:
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._metrics.clear()


@dataclass
class ServingMetrics:
    """The metric set the inference gateway publishes."""

    registry: MetricsRegistry = field(default_factory=MetricsRegistry)
    _ready: bool = False

    def install(self) -> ServingMetrics:
        if self._ready:
            return self
        reg = self.registry
        self.requests_total = reg.counter(
            "requests_total", "Generation requests accepted, by terminal state.", ("status",)
        )
        self.prompt_tokens_total = reg.counter("prompt_tokens_total", "Prompt tokens processed.")
        self.generation_tokens_total = reg.counter(
            "generation_tokens_total", "Tokens generated."
        )
        self.queue_depth = reg.gauge("queue_depth", "Requests waiting for a batch slot.")
        self.active_sequences = reg.gauge("active_sequences", "Sequences currently decoding.")
        self.batch_size = reg.histogram(
            "batch_size", "Sequences per decode step.",
            buckets=(1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128),
        )
        self.ttft = reg.histogram(
            "time_to_first_token_seconds", "Time from submit to first token."
        )
        self.tpot = reg.histogram(
            "time_per_output_token_seconds", "Inter-token latency while decoding."
        )
        self.e2e = reg.histogram(
            "request_duration_seconds", "Total request latency."
        )
        self.tokens_per_second = reg.gauge(
            "generation_tokens_per_second", "Rolling generation throughput."
        )
        self.kv_padding_waste = reg.gauge(
            "kv_padding_waste_ratio",
            "Fraction of padded KV slots wasted by naive batched attention.",
        )
        self.queue_wait = reg.histogram(
            "queue_wait_seconds", "Time a request spent waiting before its first prefill."
        )
        self._ready = True
        return self

    def summary(self) -> dict:
        return {
            "requests": {
                "succeeded": self.requests_total.value(status="succeeded"),
                "failed": self.requests_total.value(status="failed"),
                "cancelled": self.requests_total.value(status="cancelled"),
            },
            "prompt_tokens": self.prompt_tokens_total.value(),
            "generation_tokens": self.generation_tokens_total.value(),
            "queue_depth": self.queue_depth.value(),
            "active_sequences": self.active_sequences.value(),
            "ttft_seconds": self.ttft.snapshot(),
            "tpot_seconds": self.tpot.snapshot(),
            "request_seconds": self.e2e.snapshot(),
            "tokens_per_second": self.tokens_per_second.value(),
            "kv_padding_waste_ratio": self.kv_padding_waste.value(),
        }

    def __getattr__(self, item: str):  # pragma: no cover - only hit before install()
        if item.startswith("_"):
            raise AttributeError(item)
        raise AttributeError(f"{item!r} is unavailable until install() has been called")
