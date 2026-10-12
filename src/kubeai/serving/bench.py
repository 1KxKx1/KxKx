"""Benchmarks that make the platform's claims measurable.

Two experiments:

``run_benchmark``
    Serial vs **static** batching vs **continuous** batching on the same model,
    prompt mix and token budget. Static batching waits for the slowest request in
    a group before admitting the next; continuous batching rebuilds the batch every
    decode step. The gap between them is the whole point of the scheduler.

``run_workflow_benchmark``
    The same DAG executed with ``max_parallel=1`` and with several workers, which
    isolates the speedup the workflow engine actually delivers.
"""

from __future__ import annotations

import asyncio
import json
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .backends import EOS, Backend, CharTokenizer, Sequence, build_backend
from .batcher import ContinuousBatcher, FinishReason, GenerationRequest, sample_token
from .metrics import ServingMetrics


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def synthetic_prompts(count: int, prompt_tokens: int, seed: int = 7) -> list[str]:
    """A reproducible prompt mix with a realistic length spread."""
    rng = np.random.default_rng(seed)
    words = ["tensor", "kernel", "gradient", "shard", "throughput", "latency",
             "pipeline", "checkpoint", "attention", "quantise", "replica", "warmup"]
    prompts: list[str] = []
    for _ in range(count):
        # Vary length by +/- 25% so the batch is not uniform — uniform batches hide
        # exactly the scheduling problem we are measuring.
        length = max(4, int(prompt_tokens * float(rng.uniform(0.75, 1.25))))
        prompts.append(" ".join(str(rng.choice(words)) for _ in range(length)))
    return prompts


def synthetic_output_lengths(count: int, max_tokens: int, seed: int = 7,
                             spread: float = 0.85) -> list[int]:
    """Per-request output budgets.

    Real traffic does **not** stop at the same token. This is the single most
    important property for the static-vs-continuous comparison: when every request
    in a group runs to the same length, static batching is already optimal and the
    two look identical. Give them different finish times and static batching starts
    paying for slots it can no longer use.
    """
    rng = np.random.default_rng(seed + 977)
    low = max(1, int(max_tokens * (1.0 - spread)))
    return [int(rng.integers(low, max_tokens + 1)) for _ in range(count)]


def summarise(
    label: str,
    batch_size: int,
    requests: list[GenerationRequest],
    wall_seconds: float,
    backend: Backend,
    notes: str = "",
    metrics: ServingMetrics | None = None,
    decode_slots: int | None = None,
) -> dict[str, Any]:
    latencies = [r.latency for r in requests]
    ttfts = [r.ttft for r in requests if r.ttft is not None]
    tpots = [r.tpot for r in requests if r.tpot is not None]
    generated = sum(r.output_tokens for r in requests)
    prompt_total = sum(r.prompt_tokens for r in requests)
    padding = float(getattr(backend, "padding_waste", 0.0))

    summary: dict[str, Any] = {
        "label": label,
        "batch_size": batch_size,
        "notes": notes,
        "requests": len(requests),
        "prompt_tokens": prompt_total,
        "generated_tokens": generated,
        "wall_seconds": round(wall_seconds, 4),
        "throughput_tokens_per_second": round(generated / wall_seconds, 2)
        if wall_seconds > 0
        else 0.0,
        "requests_per_second": round(len(requests) / wall_seconds, 3) if wall_seconds > 0 else 0.0,
        "mean_ttft_seconds": round(statistics.fmean(ttfts), 5) if ttfts else 0.0,
        "mean_tpot_seconds": round(statistics.fmean(tpots), 6) if tpots else 0.0,
        "mean_latency_seconds": round(statistics.fmean(latencies), 4) if latencies else 0.0,
        "p50_latency_seconds": round(percentile(latencies, 0.50), 4),
        "p95_latency_seconds": round(percentile(latencies, 0.95), 4),
        "p99_latency_seconds": round(percentile(latencies, 0.99), 4),
        "kv_padding_waste_ratio": round(padding, 4),
        "tokens_per_second_per_request": round(
            (generated / wall_seconds) / max(len(requests), 1), 4
        ),
    }
    if metrics is not None:
        # TTFT here includes queueing, which for a burst workload is dominated by
        # the wait rather than by prefill. Report both so the two are separable.
        summary["mean_queue_wait_seconds"] = round(metrics.queue_wait.snapshot()["avg"], 5)
    if decode_slots:
        # Slots = (decode step, sequence) pairs consumed. Compare against the slots
        # that *had* to be spent: each request needs ``len(output)-1`` decode steps,
        # since its first token comes out of prefill. 1.0 means every slot produced
        # a token that was actually returned to a user; 0.5 means half the compute
        # was spent on sequences that had already finished.
        necessary = max(generated - len(requests), 0)
        summary["decode_slots"] = decode_slots
        summary["necessary_decode_slots"] = necessary
        summary["slot_efficiency"] = round(necessary / decode_slots, 4)
    return summary


# --------------------------------------------------------------------------- #
# Static batching baseline
# --------------------------------------------------------------------------- #
@dataclass
class BatchRun:
    """Result of a static-batch run, including the slots it consumed."""

    requests: list[GenerationRequest] = field(default_factory=list)
    decode_slots: int = 0


async def run_static_batch(
    backend: Backend,
    tokenizer: CharTokenizer,
    specs: list[tuple[str, int]],
    batch_size: int,
    temperature: float,
    rng: np.random.Generator,
    submitted_at: float,
) -> BatchRun:
    """Group requests, decode until **every** member finishes, then start the next group.

    This is the behaviour continuous batching replaces: a request that emitted EOS
    after 3 tokens still occupies its slot until the slowest member of its group is
    done. ``decode_slots`` counts those occupied slots, which is what exposes the
    waste even when wall-clock time happens to look similar.
    """
    run = BatchRun()
    for start in range(0, len(specs), batch_size):
        group = specs[start : start + batch_size]
        requests: list[GenerationRequest] = []
        for index, (prompt, max_tokens) in enumerate(group):
            request = GenerationRequest(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                id=f"static-{start + index}",
                submitted_at=submitted_at,
            )
            sequence = Sequence(
                id=request.id,
                prompt_tokens=tokenizer.encode(prompt),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            request.sequence = sequence
            logits = backend.prefill(sequence)
            token = sample_token(logits, temperature, 1.0, rng)
            sequence.output_tokens.append(token)
            request.first_token_at = time.time()
            requests.append(request)

        active = list(requests)
        while active:
            # Every member of the group holds its slot for the whole group, and
            # every member is included in the decode call — including the ones
            # that have already emitted EOS. Their compute is paid for and thrown
            # away. That is what "static" means, and it is why the slot efficiency
            # below is well under 1.
            run.decode_slots += len(active)
            seqs = [r.sequence for r in active if r.sequence is not None]
            logits_list = backend.decode(seqs)
            for request, logits in zip(active, logits_list, strict=True):
                sequence = request.sequence
                if sequence is None or request.finished:
                    continue  # compute spent, result discarded
                token = sample_token(logits, temperature, 1.0, rng)
                sequence.output_tokens.append(token)
                if token == EOS:
                    request.finish_reason = FinishReason.STOP
                    request.finished_at = time.time()
                elif len(sequence.output_tokens) >= request.max_tokens:
                    request.finish_reason = FinishReason.LENGTH
                    request.finished_at = time.time()
            if all(r.finished for r in active):
                break

        run.requests.extend(requests)
    return run


# --------------------------------------------------------------------------- #
# Main benchmark
# --------------------------------------------------------------------------- #
async def measure_decode_scaling(
    make_backend: Callable[[], Backend],
    batch_sizes: list[int],
    context_length: int = 8,
    steps: int = 12,
) -> list[dict[str, Any]]:
    """Isolate the *backend's* per-step batching from the scheduler's overhead.

    This measures how many tokens per second one decode step produces as the batch
    grows, with no queue, no metrics and no async in the way. The gap between these
    numbers and the end-to-end ones is the cost of the batcher plus, crucially, the
    per-step KV-cache gather/scatter — see the README's "known limitations".
    """
    results: list[dict[str, Any]] = []
    for size in batch_sizes:
        backend = make_backend()
        sequences = []
        for index in range(size):
            sequence = Sequence(
                id=f"scale-{index}",
                prompt_tokens=[((index + j) % 2000) + 4 for j in range(context_length)],
            )
            backend.prefill(sequence)
            sequence.output_tokens.append(5)
            sequences.append(sequence)

        # Warm up so the first-touch allocation cost is not charged to the timing.
        backend.decode(sequences)
        started = time.perf_counter()
        for _ in range(steps):
            backend.decode(sequences)
        elapsed = (time.perf_counter() - started) / steps

        results.append(
            {
                "batch_size": size,
                "context_length": context_length,
                "milliseconds_per_step": round(elapsed * 1000, 3),
                "tokens_per_second": round(size / elapsed, 2) if elapsed else 0.0,
                "kv_padding_waste_ratio": round(float(getattr(backend, "padding_waste", 0.0)), 4),
            }
        )

    baseline = results[0]["tokens_per_second"] if results else 0.0
    for row in results:
        row["speedup_vs_batch_1"] = (
            round(row["tokens_per_second"] / baseline, 3) if baseline else 0.0
        )
    return results


async def run_benchmark(
    backend: str = "numpy",
    requests: int = 64,
    max_tokens: int = 32,
    batch_sizes: list[int] | None = None,
    prompt_tokens: int = 32,
    seed: int = 7,
    output_dir: str | Path | None = "benchmarks/results",
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    batch_sizes = batch_sizes or [1, 8, 32]
    prompts = synthetic_prompts(requests, prompt_tokens, seed=seed)
    output_lengths = synthetic_output_lengths(requests, max_tokens, seed=seed)
    specs = list(zip(prompts, output_lengths, strict=True))
    scenarios: list[dict[str, Any]] = []

    def make_backend() -> Backend:
        return build_backend(backend, **(model_kwargs or {}))

    # Every scenario models the same event: a burst of requests that all arrive at
    # t=0. The anchor is taken *per scenario* — sharing one anchor across the run
    # would charge each scenario for the wall-clock time of all the earlier ones.
    # Anchoring submitted_at to the burst start is what makes TTFT and latency
    # comparable between the three strategies instead of flattering one of them.

    # --- 1. serial (batch = 1) ------------------------------------------ #
    serial_backend = make_backend()
    tokenizer = CharTokenizer()
    rng = np.random.default_rng(seed)
    burst_started = time.time()
    started = time.perf_counter()
    serial_results: list[GenerationRequest] = []
    for index, (prompt, budget) in enumerate(specs):
        request = GenerationRequest(
            prompt=prompt,
            max_tokens=budget,
            temperature=0.8,
            id=f"serial-{index}",
            submitted_at=burst_started,
        )
        sequence = Sequence(
            id=request.id,
            prompt_tokens=tokenizer.encode(prompt),
            max_tokens=budget,
            temperature=0.8,
        )
        request.sequence = sequence
        logits = serial_backend.prefill(sequence)
        token = sample_token(logits, 0.8, 1.0, rng)
        sequence.output_tokens.append(token)
        request.first_token_at = time.time()
        while len(sequence.output_tokens) < budget:
            logits_list = serial_backend.decode([sequence])
            token = sample_token(logits_list[0], 0.8, 1.0, rng)
            sequence.output_tokens.append(token)
        request.finish_reason = FinishReason.LENGTH
        request.finished_at = time.time()
        serial_results.append(request)
    serial_wall = time.perf_counter() - started
    scenarios.append(
        summarise(
            "serial (no batching)",
            1,
            serial_results,
            serial_wall,
            serial_backend,
            notes="one request at a time; the baseline everything else is compared against",
            # Serial spends exactly one step per required token: perfectly
            # efficient, but it never overlaps anything.
            decode_slots=sum(max(r.output_tokens - 1, 0) for r in serial_results),
        )
    )

    # --- 2. static batching --------------------------------------------- #
    static_backends: dict[int, Backend] = {}
    for size in batch_sizes:
        if size <= 1:
            continue
        static_backend = make_backend()
        static_backends[size] = static_backend
        burst_started = time.time()
        started = time.perf_counter()
        static_run = await run_static_batch(
            static_backend,
            tokenizer,
            specs,
            size,
            0.8,
            np.random.default_rng(seed),
            submitted_at=burst_started,
        )
        wall = time.perf_counter() - started
        scenarios.append(
            summarise(
                f"static batching (batch={size})",
                size,
                static_run.requests,
                wall,
                static_backend,
                notes="group decode-until-all-finish; early finishers hold their slot",
                decode_slots=static_run.decode_slots,
            )
        )

    # --- 3. continuous batching ----------------------------------------- #
    for size in batch_sizes:
        metrics = ServingMetrics().install()
        cont_backend = make_backend()
        batcher = ContinuousBatcher(
            cont_backend,
            tokenizer=tokenizer,
            max_batch_size=size,
            max_batch_tokens=1 << 20,
            metrics=metrics,
            seed=seed,
        )
        await batcher.start()
        burst_started = time.time()
        started = time.perf_counter()
        pending = [
            GenerationRequest(
                prompt=prompt,
                max_tokens=budget,
                temperature=0.8,
                id=f"cont-{index}",
                submitted_at=burst_started,
            )
            for index, (prompt, budget) in enumerate(specs)
        ]
        for request in pending:
            batcher.submit_nowait(request)
        while not all(r.finished for r in pending):
            await asyncio.sleep(0.001)
        wall = time.perf_counter() - started
        await batcher.stop()

        summary = summarise(
            f"continuous batching (batch={size})",
            size,
            pending,
            wall,
            cont_backend,
            notes="batch rebuilt every decode step; finished slots refilled immediately",
            metrics=metrics,
            decode_slots=sum(batcher.batch_size_history),
        )
        summary["mean_batch_size"] = round(batcher.stats()["mean_batch_size"], 3)
        scenarios.append(summary)

    serial = next(s for s in scenarios if s["batch_size"] == 1 and "serial" in s["label"])
    for scenario in scenarios:
        if scenario is serial:
            scenario["speedup_vs_serial"] = 1.0
            continue
        scenario["speedup_vs_serial"] = round(
            scenario["throughput_tokens_per_second"]
            / max(serial["throughput_tokens_per_second"], 1e-9),
            3,
        )

    # Isolate the backend's own batching from everything the batcher adds.
    decode_scaling = await measure_decode_scaling(
        make_backend, sorted({1, *batch_sizes}), context_length=max(prompt_tokens, 4)
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "config": {
            "backend": backend,
            "requests": requests,
            "max_tokens": max_tokens,
            "max_tokens_mean": round(statistics.fmean(output_lengths), 2),
            "prompt_tokens": prompt_tokens,
            "batch_sizes": batch_sizes,
            "seed": seed,
            "arrival_pattern": "burst (all requests arrive at t=0)",
        },
        "decode_scaling": decode_scaling,
        "scenarios": scenarios,
    }

    if output_dir:
        write_report(report, output_dir, "bench_serving")
    return report


async def run_workflow_benchmark(
    requests: int = 24,
    seconds_per_step: float = 0.25,
    out: str | Path | None = "benchmarks/results",
) -> dict[str, Any]:
    """Measure the DAG engine's scheduling speedup on an embarrassingly-parallel stage."""
    from ..workflow.engine import WorkflowEngine
    from ..workflow.executors import DryRunExecutor
    from ..workflow.spec import Workflow

    fan_out = requests
    document = {
        "apiVersion": "kubeai.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {"name": "fanout-bench"},
        "spec": {
            "entrypoint": "pipeline",
            "templates": [
                {
                    "name": "pipeline",
                    "dag": {
                        "tasks": [
                            *[
                                {"name": f"shard-{i:03d}", "template": "work"}
                                for i in range(fan_out)
                            ],
                            {
                                "name": "merge",
                                "template": "work",
                                "depends": [f"shard-{i:03d}" for i in range(fan_out)],
                            },
                        ]
                    },
                },
                {"name": "work", "container": {"image": "busybox", "command": ["true"]}},
            ],
        },
    }

    scenarios: list[dict[str, Any]] = []
    baseline: float | None = None
    for parallel in (1, 4, 8, 16):
        workflow = Workflow.from_dict(document)
        executor = DryRunExecutor(durations={"work": seconds_per_step})
        engine = WorkflowEngine(workflow, executor=executor, workdir=".kubeai/bench", max_parallel=parallel)
        run = await engine.run()
        duration = run.duration
        if baseline is None:
            baseline = duration
        scenarios.append(
            {
                "label": f"max_parallel={parallel}",
                "max_parallel": parallel,
                "tasks": len(run.steps),
                "duration_seconds": round(duration, 4),
                "sum_of_step_seconds": round(run.critical_path(), 4),
                "speedup": round(baseline / duration, 3) if duration else 0.0,
                "status": run.status.value,
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "tasks": fan_out + 1,
            "seconds_per_step": seconds_per_step,
            "shape": "fan-out then merge",
        },
        "scenarios": scenarios,
    }
    if out:
        write_report(report, out, "bench_workflow")
    return report


def write_report(report: dict[str, Any], output_dir: str | Path, prefix: str) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{prefix}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest = directory / f"{prefix}_latest.json"
    latest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


@dataclass
class BenchmarkSuite:
    """Convenience wrapper so callers can run everything in one go."""

    reports: list[dict[str, Any]] = field(default_factory=list)

    async def run_all(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.reports.append(await run_benchmark(**kwargs))
        self.reports.append(await run_workflow_benchmark())
        return self.reports
