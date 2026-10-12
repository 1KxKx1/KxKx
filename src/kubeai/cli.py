"""kubeai command line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__

EXIT_OK = 0
EXIT_FAILURE = 1


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _print_table(headers: list[str], rows: list[list[Any]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


def _emit(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------- #
# gpus
# --------------------------------------------------------------------------- #
def cmd_gpus(args: argparse.Namespace) -> int:
    from .scheduler.device import discover

    devices, source = discover(
        allow_simulation=not args.no_simulation, simulated_spec=args.simulate
    )
    print(f"accelerator source: {source}")
    if not devices:
        print("no accelerators found (and simulation disabled)")
        return EXIT_OK
    _print_table(
        ["idx", "type", "name", "total MiB", "used MiB", "free MiB", "util %", "temp", "domain"],
        [
            [
                d.index,
                d.device_type.value,
                d.name,
                d.memory_total_mib,
                d.memory_used_mib,
                d.memory_free_mib,
                d.utilization,
                d.temperature_c if d.temperature_c is not None else "-",
                d.interconnect_domain or "-",
            ]
            for d in devices
        ],
    )
    if args.json:
        _emit([d.to_dict() for d in devices], True)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# scheduler
# --------------------------------------------------------------------------- #
def cmd_schedule(args: argparse.Namespace) -> int:
    from .scheduler.simulate import build_fleet, compare_policies, generate_workload, run_simulation

    nodes = build_fleet(
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        memory_mib=args.gpu_memory_mib,
    )
    # Generate a workload the fleet can actually serve, so pending counts mean
    # "waiting for capacity" rather than "impossible".
    tasks = generate_workload(
        count=args.requests,
        seed=args.seed,
        max_memory_mib=args.gpu_memory_mib,
        max_devices=args.gpus_per_node,
    )

    if args.policy == "compare":
        reports = compare_policies(
            tasks, nodes, policies=("binpack", "spread", "topology-aware"),
            preemption=not args.no_preemption,
        )
    else:
        reports = [
            run_simulation(tasks, nodes, policy=args.policy, preemption=not args.no_preemption)
        ]

    _print_table(
        ["policy", "submitted", "completed", "preempted", "avg util", "peak util",
         "makespan s", "stranded MiB", "pending"],
        [
            [
                r.policy,
                r.submitted,
                r.completed,
                r.preempted,
                f"{r.avg_utilization:.3f}",
                f"{r.peak_utilization:.3f}",
                f"{r.makespan:.1f}",
                r.stranded_mib,
                r.pending,
            ]
            for r in reports
        ],
    )
    if args.json:
        _emit([r.to_dict() for r in reports], True)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# workflow
# --------------------------------------------------------------------------- #
def cmd_workflow_run(args: argparse.Namespace) -> int:
    from .workflow.engine import WorkflowEngine, WorkflowFailed
    from .workflow.spec import Workflow

    workflow = Workflow.load(args.file)
    engine = WorkflowEngine(
        workflow,
        executor=args.executor,
        workdir=args.workdir,
        max_parallel=args.parallel,
    )
    run = asyncio.run(engine.run())

    _print_table(
        ["task", "status", "exit", "seconds", "attempts"],
        [
            [name, run.steps[name].status.value, run.steps[name].exit_code,
             f"{run.steps[name].duration:.2f}", run.steps[name].attempts]
            for name in run.order
        ],
    )
    print(f"\nworkflow {run.name!r} {run.status.value} in {run.duration:.2f}s "
          f"({run.counts()})")
    if args.json:
        _emit(run.to_dict(), True)
    if args.show_output:
        for name in run.order:
            result = run.steps[name]
            if result.stdout:
                print(f"\n--- {name} stdout ---\n{result.stdout.rstrip()}")
    return EXIT_OK if run.status.value == "Succeeded" else EXIT_FAILURE


def cmd_workflow_compile(args: argparse.Namespace) -> int:
    from .workflow.compile import argo_yaml, manifests_yaml
    from .workflow.spec import Workflow

    workflow = Workflow.load(args.file)
    if args.format == "kubeai":
        text = workflow.to_yaml()
    elif args.format == "argo":
        text = argo_yaml(workflow)
    else:
        text = manifests_yaml(workflow)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out} ({len(text)} bytes)")
    else:
        print(text)
    return EXIT_OK


def cmd_workflow_validate(args: argparse.Namespace) -> int:
    from .workflow.spec import Workflow, WorkflowValidationError

    try:
        workflow = Workflow.load(args.file)
    except WorkflowValidationError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    print(f"ok: {workflow.name} ({len(workflow.templates)} templates, "
          f"entrypoint={workflow.entrypoint})")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# serving
# --------------------------------------------------------------------------- #
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .serving.gateway import GatewayConfig, create_app

    config = GatewayConfig(
        default_model=args.model,
        default_backend=args.backend,
        memory_budget_mib=args.memory_budget_mib,
        max_batch_size=args.max_batch_size,
        sim_gpus=args.simulate,
    )
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return EXIT_OK


def cmd_bench_serving(args: argparse.Namespace) -> int:
    from .serving.backends import TransformerConfig
    from .serving.bench import run_benchmark

    model_kwargs: dict[str, Any] = {}
    if args.backend.startswith("numpy"):
        model_kwargs["config"] = TransformerConfig(
            vocab_size=args.model_vocab, dim=args.model_dim, n_layers=args.model_layers
        )

    report = asyncio.run(
        run_benchmark(
            backend=args.backend,
            requests=args.requests,
            max_tokens=args.max_tokens,
            batch_sizes=[int(x) for x in args.batch_sizes.split(",")],
            prompt_tokens=args.prompt_tokens,
            seed=args.seed,
            output_dir=args.out,
            model_kwargs=model_kwargs,
        )
    )
    rows = []
    for scenario in report["scenarios"]:
        rows.append(
            [
                scenario["label"],
                scenario["batch_size"],
                f"{scenario['throughput_tokens_per_second']:.1f}",
                f"{scenario.get('speedup_vs_serial', 0.0):.2f}x",
                f"{scenario.get('slot_efficiency', 0.0):.3f}",
                f"{scenario['mean_ttft_seconds'] * 1000:.1f}",
                f"{scenario['mean_tpot_seconds'] * 1000:.2f}",
                f"{scenario['kv_padding_waste_ratio']:.3f}",
            ]
        )
    _print_table(
        ["scenario", "batch", "tok/s", "speedup", "slot eff", "TTFT ms", "TPOT ms", "KV waste"],
        rows,
    )
    print("\nbackend-only decode scaling (no batcher, no queue, no metrics):")
    _print_table(
        ["batch", "ms/step", "tok/s", "speedup"],
        [
            [
                row["batch_size"],
                f"{row['milliseconds_per_step']:.2f}",
                f"{row['tokens_per_second']:.1f}",
                f"{row['speedup_vs_batch_1']:.2f}x",
            ]
            for row in report.get("decode_scaling", [])
        ],
    )
    if args.json:
        _emit(report, True)
    return EXIT_OK


def cmd_bench_workflow(args: argparse.Namespace) -> int:
    from .serving.bench import run_workflow_benchmark

    report = asyncio.run(
        run_workflow_benchmark(
            requests=args.requests, seconds_per_step=args.seconds_per_step, out=args.out
        )
    )
    _print_table(
        ["scenario", "total s", "sum step s", "speedup"],
        [
            [
                s["label"],
                f"{s['duration_seconds']:.2f}",
                f"{s['sum_of_step_seconds']:.2f}",
                f"{s['speedup']:.2f}x",
            ]
            for s in report["scenarios"]
        ],
    )
    if args.json:
        _emit(report, True)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kubeai",
        description="Cloud-native AI platform: GPU scheduling, training/inference "
        "workflows and batched LLM serving.",
    )
    parser.add_argument("--version", action="version", version=f"kubeai {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    gpus = sub.add_parser("gpus", help="list accelerators discovered on this host")
    gpus.add_argument("--simulate", help="simulated fleet spec, e.g. nvidia-rtx4090:4")
    gpus.add_argument("--no-simulation", action="store_true")
    gpus.add_argument("--json", action="store_true")
    gpus.set_defaults(func=cmd_gpus)

    schedule = sub.add_parser("schedule", help="simulate workload placement")
    schedule.add_argument("--requests", type=int, default=200)
    schedule.add_argument("--nodes", type=int, default=4)
    schedule.add_argument("--gpus-per-node", type=int, default=4)
    schedule.add_argument("--gpu-memory-mib", type=int, default=24564)
    schedule.add_argument("--seed", type=int, default=7)
    schedule.add_argument(
        "--policy", default="compare",
        choices=["compare", "binpack", "spread", "topology-aware"],
    )
    schedule.add_argument("--no-preemption", action="store_true")
    schedule.add_argument("--json", action="store_true")
    schedule.set_defaults(func=cmd_schedule)

    workflow = sub.add_parser("workflow", help="run and compile DAG workflows")
    workflow_sub = workflow.add_subparsers(dest="workflow_command", required=True)

    run = workflow_sub.add_parser("run", help="execute a workflow locally")
    run.add_argument("file")
    run.add_argument("--executor", default="local", choices=["local", "docker", "dry-run"])
    run.add_argument("--workdir", default=".kubeai/runs")
    run.add_argument("--parallel", type=int, default=4)
    run.add_argument("--show-output", action="store_true")
    run.add_argument("--json", action="store_true")
    run.set_defaults(func=cmd_workflow_run)

    compile_ = workflow_sub.add_parser("compile", help="compile to Argo / Kubeflow")
    compile_.add_argument("file")
    compile_.add_argument("--format", default="argo", choices=["kubeai", "argo", "k8s"])
    compile_.add_argument("--out")
    compile_.set_defaults(func=cmd_workflow_compile)

    validate = workflow_sub.add_parser("validate", help="validate a workflow document")
    validate.add_argument("file")
    validate.set_defaults(func=cmd_workflow_validate)

    serve = sub.add_parser("serve", help="start the OpenAI-compatible gateway")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--model", default="kubeai-tiny")
    serve.add_argument("--backend", default="numpy")
    serve.add_argument("--max-batch-size", type=int, default=32)
    serve.add_argument("--memory-budget-mib", type=int, default=24000)
    serve.add_argument("--simulate", default="nvidia-rtx4090:4")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    bench = sub.add_parser("bench", help="benchmarks")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True)

    bench_serving = bench_sub.add_parser("serving", help="batched inference throughput")
    bench_serving.add_argument("--backend", default="numpy")
    bench_serving.add_argument("--requests", type=int, default=64)
    bench_serving.add_argument("--max-tokens", type=int, default=32)
    bench_serving.add_argument("--prompt-tokens", type=int, default=32)
    bench_serving.add_argument("--batch-sizes", default="1,8,32")
    bench_serving.add_argument("--seed", type=int, default=7)
    bench_serving.add_argument("--out", default="benchmarks/results")
    # Model size decides whether batching can pay off at all: if a step is too
    # small, per-step overhead dominates and the speedup is bounded by Python.
    bench_serving.add_argument("--model-dim", type=int, default=256)
    bench_serving.add_argument("--model-layers", type=int, default=4)
    bench_serving.add_argument("--model-vocab", type=int, default=23044)
    bench_serving.add_argument("--json", action="store_true")
    bench_serving.set_defaults(func=cmd_bench_serving)

    bench_workflow = bench_sub.add_parser("workflow", help="DAG parallelism speedup")
    bench_workflow.add_argument("--requests", type=int, default=24)
    bench_workflow.add_argument("--seconds-per-step", type=float, default=0.25)
    bench_workflow.add_argument("--out", default="benchmarks/results")
    bench_workflow.add_argument("--json", action="store_true")
    bench_workflow.set_defaults(func=cmd_bench_workflow)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    started = time.time()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        if "--traceback" in (argv or sys.argv):
            raise
        return EXIT_FAILURE
    finally:
        _ = started


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
