"""Workflow tests: validation, DAG execution, retries, gates and cluster compilation."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import yaml

from kubeai.workflow.compile import to_argo, to_kubernetes_manifests, to_pytorchjob
from kubeai.workflow.engine import WorkflowEngine, WorkflowFailed, run_workflow
from kubeai.workflow.executors import DryRunExecutor, StepStatus
from kubeai.workflow.spec import (
    ResourceSpec,
    Workflow,
    WorkflowValidationError,
    parse_cpu,
    parse_quantity_mib,
    substitute,
)


def workflow_dict(tasks: list[dict], templates: list[dict] | None = None, **spec_extra) -> dict:
    return {
        "apiVersion": "kubeai.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {"name": "test-flow", "namespace": "ai-platform"},
        "spec": {
            "entrypoint": "pipeline",
            "templates": [
                {"name": "pipeline", "dag": {"tasks": tasks}},
                *(templates or [{"name": "work", "container": {"image": "busybox", "command": ["true"]}}]),
            ],
            **spec_extra,
        },
    }


@pytest.fixture()
def fast_executor() -> DryRunExecutor:
    return DryRunExecutor(durations={"work": 0.01}, default_duration=0.005)


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    """A per-test run directory — never a hard-coded /tmp, which is not portable."""
    return tmp_path / "runs"


# --------------------------------------------------------------------------- #
# spec parsing
# --------------------------------------------------------------------------- #
def test_parse_cpu_and_memory_quantities():
    assert parse_cpu("2") == 2000
    assert parse_cpu("500m") == 500
    assert parse_quantity_mib("4Gi") == 4096
    assert parse_quantity_mib("512Mi") == 512
    assert parse_quantity_mib("1Ti") == 1024 * 1024


def test_resource_spec_reads_gpu_fields():
    spec = ResourceSpec.from_dict({"cpu": "4", "memory": "16Gi", "gpu": 2, "gpuMemoryMiB": 20480})
    assert spec.gpu == 2
    assert spec.gpu_memory_mib == 20480
    assert spec.cpu_millicores == 4000
    assert spec.memory_mib == 16384


def test_substitute_leaves_unknown_placeholders():
    assert substitute("hi {{name}}", {"name": "kai"}) == "hi kai"
    assert substitute("hi {{missing}}", {}) == "hi {{missing}}"


def test_roundtrip_through_yaml_preserves_structure():
    workflow = Workflow.from_dict(
        workflow_dict(
            [{"name": "a", "template": "work"}],
            [
                {"name": "work", "container": {"image": "busybox", "command": ["true"]},
                 "resources": {"gpu": 1, "gpuMemoryMiB": 1024}},
            ],
        )
    )
    reloaded = Workflow.from_yaml(workflow.to_yaml())
    assert reloaded.name == workflow.name
    assert reloaded.entrypoint == "pipeline"
    assert reloaded.template("work").resources.gpu == 1


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_validation_rejects_cycles():
    document = workflow_dict(
        [
            {"name": "a", "template": "work", "depends": ["b"]},
            {"name": "b", "template": "work", "depends": ["a"]},
        ]
    )
    with pytest.raises(WorkflowValidationError, match="cycle"):
        Workflow.from_dict(document)


def test_validation_rejects_unknown_template_reference():
    document = workflow_dict([{"name": "a", "template": "does-not-exist"}])
    with pytest.raises(WorkflowValidationError, match="unknown template"):
        Workflow.from_dict(document)


def test_validation_rejects_unknown_dependency():
    document = workflow_dict([{"name": "a", "template": "work", "depends": ["ghost"]}])
    with pytest.raises(WorkflowValidationError, match="unknown tasks"):
        Workflow.from_dict(document)


def test_validation_rejects_missing_entrypoint():
    document = workflow_dict([{"name": "a", "template": "work"}])
    document["spec"]["entrypoint"] = "nope"
    with pytest.raises(WorkflowValidationError, match="entrypoint"):
        Workflow.from_dict(document)


def test_validation_rejects_template_without_body():
    document = workflow_dict(
        [{"name": "a", "template": "empty"}],
        [{"name": "empty", "resources": {"cpu": "1"}}],
    )
    with pytest.raises(WorkflowValidationError, match="must define one of"):
        Workflow.from_dict(document)


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
async def test_dag_runs_tasks_in_dependency_order(fast_executor, workdir):
    workflow = Workflow.from_dict(
        workflow_dict(
            [
                {"name": "prepare", "template": "work"},
                {"name": "train", "template": "work", "depends": ["prepare"]},
                {"name": "evaluate", "template": "work", "depends": ["train"]},
            ]
        )
    )
    run = await WorkflowEngine(workflow, executor=fast_executor, workdir=workdir).run()
    assert run.status is StepStatus.SUCCEEDED
    assert run.order.index("prepare") < run.order.index("train") < run.order.index("evaluate")
    assert run.counts() == {"Succeeded": 3}


async def test_independent_tasks_can_run_concurrently(fast_executor, workdir):
    tasks = [{"name": f"shard-{i}", "template": "work"} for i in range(6)]
    workflow = Workflow.from_dict(workflow_dict(tasks))

    # A step long enough that scheduling noise cannot dominate the measurement, and a
    # fresh executor per run so accumulated attempt counters cannot leak between them.
    def slow() -> DryRunExecutor:
        return DryRunExecutor(durations={"work": 0.05}, default_duration=0.05)

    serial = await WorkflowEngine(
        workflow, executor=slow(), workdir=workdir, max_parallel=1
    ).run()
    parallel = await WorkflowEngine(
        workflow, executor=slow(), workdir=workdir, max_parallel=6
    ).run()

    assert serial.status is StepStatus.SUCCEEDED
    assert parallel.status is StepStatus.SUCCEEDED
    assert serial.duration >= 6 * 0.05 * 0.9, "a serial run should cost the sum of its steps"
    # Same work, materially less wall clock once the engine may overlap it.
    assert parallel.duration < serial.duration * 0.6


async def test_failure_propagates_as_skipped_dependents(workdir):
    executor = DryRunExecutor(failures={("train", 1), ("train", 2)})
    workflow = Workflow.from_dict(
        workflow_dict(
            [
                {"name": "prepare", "template": "work"},
                {"name": "train", "template": "work", "depends": ["prepare"]},
                {"name": "evaluate", "template": "work", "depends": ["train"]},
            ]
        )
    )
    run = await WorkflowEngine(workflow, executor=executor, workdir=workdir).run()
    assert run.steps["prepare"].status is StepStatus.SUCCEEDED
    assert run.steps["train"].status is StepStatus.FAILED
    assert run.steps["evaluate"].status is StepStatus.SKIPPED
    assert run.status is StepStatus.FAILED


async def test_retry_succeeds_on_a_later_attempt(workdir):
    executor = DryRunExecutor(failures={("train", 1)})
    document = workflow_dict(
        [{"name": "train", "template": "train-step"}],
        [
            {
                "name": "train-step",
                "container": {"image": "busybox", "command": ["true"]},
                "retryStrategy": {"limit": 2, "backoffSeconds": 0.001, "backoffFactor": 1},
            }
        ],
    )
    workflow = Workflow.from_dict(document)
    run = await WorkflowEngine(workflow, executor=executor, workdir=workdir).run()
    assert run.steps["train"].status is StepStatus.SUCCEEDED
    assert run.steps["train"].attempts == 2


async def test_when_gate_skips_a_step(fast_executor, workdir):
    document = workflow_dict(
        [
            {"name": "prepare", "template": "work"},
            {"name": "maybe", "template": "work", "depends": ["prepare"],
             "when": "{{tasks.prepare.status}} == Failed"},
        ]
    )
    workflow = Workflow.from_dict(document)
    run = await WorkflowEngine(workflow, executor=fast_executor, workdir=workdir).run()
    assert run.steps["maybe"].status is StepStatus.SKIPPED
    assert "when condition false" in run.steps["maybe"].stderr


async def test_when_gate_lets_a_step_run(fast_executor, workdir):
    document = workflow_dict(
        [
            {"name": "prepare", "template": "work"},
            {"name": "maybe", "template": "work", "depends": ["prepare"],
             "when": "{{tasks.prepare.status}} == Succeeded"},
        ]
    )
    workflow = Workflow.from_dict(document)
    run = await WorkflowEngine(workflow, executor=fast_executor, workdir=workdir).run()
    assert run.steps["maybe"].status is StepStatus.SUCCEEDED


async def test_parameters_flow_between_steps(workdir):
    """``{{tasks.produce.outputs.result}}`` must reach the downstream command."""
    script = "print('hello-from-script')"
    document = {
        "apiVersion": "kubeai.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {"name": "param-flow"},
        "spec": {
            "entrypoint": "pipeline",
            "arguments": {"parameters": [{"name": "who", "value": "kubeai"}]},
            "templates": [
                {
                    "name": "pipeline",
                    "dag": {
                        "tasks": [
                            {"name": "produce", "template": "producer"},
                            {
                                "name": "consume",
                                "template": "consumer",
                                "depends": ["produce"],
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "value",
                                            "value": "{{tasks.produce.outputs.result}}",
                                        }
                                    ]
                                },
                            },
                        ]
                    },
                },
                {
                    # A script template: `command` is the interpreter and the engine
                    # appends the written script path (Argo behaves the same way).
                    "name": "producer",
                    "script": {"image": "python", "source": script},
                    "outputs": {"result": "{{stdout}}"},
                },
                {
                    # A plain container template, so the argument substitution is
                    # tested independently of the script-file mechanism.
                    "name": "consumer",
                    "container": {
                        "image": "python",
                        "command": [sys.executable, "-c", "import sys; print('got:', sys.argv[1])"],
                        "args": ["{{inputs.parameters.value}}"],
                    },
                },
            ],
        },
    }
    workflow = Workflow.from_dict(document)
    run = await WorkflowEngine(workflow, executor="local", workdir=workdir).run()

    assert run.steps["produce"].status is StepStatus.SUCCEEDED
    assert run.steps["produce"].outputs["result"] == "hello-from-script"
    assert run.steps["consume"].status is StepStatus.SUCCEEDED
    assert "got: hello-from-script" in run.steps["consume"].stdout


async def test_local_executor_reports_a_failing_exit_code(workdir):
    document = workflow_dict(
        [{"name": "boom", "template": "failing"}],
        [{"name": "failing", "script": {"image": "python", "source": "raise SystemExit(3)"}}],
    )
    workflow = Workflow.from_dict(document)
    run = await WorkflowEngine(workflow, executor="local", workdir=workdir).run()
    assert run.steps["boom"].status is StepStatus.FAILED
    assert run.steps["boom"].exit_code == 3


def test_run_or_raise_raises_on_failure(workdir):
    document = workflow_dict(
        [{"name": "boom", "template": "failing"}],
        [{"name": "failing", "script": {"image": "python", "source": "raise SystemExit(1)"}}],
    )
    workflow = Workflow.from_dict(document)

    async def go():
        engine = WorkflowEngine(workflow, executor="local", workdir=workdir)
        with pytest.raises(WorkflowFailed):
            await engine.run_or_raise()

    asyncio.run(go())


def test_run_workflow_blocking_helper(workdir):
    workflow = Workflow.from_dict(workflow_dict([{"name": "a", "template": "work"}]))
    run = run_workflow(workflow, executor=DryRunExecutor(), workdir=workdir)
    assert run.status is StepStatus.SUCCEEDED


def test_workflow_loads_from_a_file(tmp_path):
    document = workflow_dict([{"name": "a", "template": "work"}])
    path = tmp_path / "flow.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    workflow = Workflow.load(path)
    assert workflow.name == "test-flow"


# --------------------------------------------------------------------------- #
# compilation
# --------------------------------------------------------------------------- #
def training_workflow() -> Workflow:
    document = {
        "apiVersion": "kubeai.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {"name": "llm-finetune", "namespace": "ai-platform"},
        "spec": {
            "entrypoint": "pipeline",
            "ttlSecondsAfterFinished": 3600,
            "arguments": {"parameters": [{"name": "baseModel", "value": "Qwen2.5-7B"}]},
            "templates": [
                {
                    "name": "pipeline",
                    "dag": {
                        "tasks": [
                            {"name": "prepare", "template": "data-prep"},
                            {"name": "train", "template": "train-job", "depends": ["prepare"]},
                            {"name": "evaluate", "template": "evaluate", "depends": ["train"]},
                        ]
                    },
                },
                {
                    "name": "data-prep",
                    "container": {"image": "python:3.12-slim", "command": ["python", "prep.py"]},
                    "resources": {"cpu": "4", "memory": "8Gi"},
                },
                {
                    "name": "train-job",
                    "container": {
                        "image": "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
                        "command": ["torchrun", "--nproc_per_node=2", "train.py"],
                        "env": {"MODEL": "{{workflow.parameters.baseModel}}"},
                    },
                    "resources": {"cpu": "16", "memory": "64Gi", "gpu": 2, "gpuMemoryMiB": 20480},
                    "retryStrategy": {"limit": 2},
                    "nodeSelector": {"nvidia.com/gpu.product": "A100"},
                },
                {
                    "name": "evaluate",
                    "script": {"image": "python:3.12-slim", "source": "print('eval')"},
                },
            ],
        },
    }
    return Workflow.from_dict(document)


def test_argo_compilation_produces_a_valid_resource():
    argo = to_argo(training_workflow())
    assert argo["apiVersion"] == "argoproj.io/v1alpha1"
    assert argo["kind"] == "Workflow"
    assert argo["metadata"]["name"] == "llm-finetune"
    assert argo["metadata"]["labels"]["kubeai.io/workflow"] == "llm-finetune"
    assert argo["spec"]["entrypoint"] == "pipeline"
    assert argo["spec"]["serviceAccountName"] == "kubeai-workflow"
    assert argo["spec"]["ttlStrategy"]["secondsAfterCompletion"] == 3600
    assert argo["spec"]["arguments"]["parameters"][0]["name"] == "baseModel"


def test_argo_dag_dependencies_are_rendered():
    argo = to_argo(training_workflow())
    dag = next(t for t in argo["spec"]["templates"] if t["name"] == "pipeline")["dag"]["tasks"]
    train = next(t for t in dag if t["name"] == "train")
    assert train["depends"] == "prepare"
    assert train["template"] == "train-job"


def test_argo_gpu_request_becomes_a_resource_limit():
    argo = to_argo(training_workflow())
    train = next(t for t in argo["spec"]["templates"] if t["name"] == "train-job")
    limits = train["container"]["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == 2
    assert limits["kubeai.io/gpu-memory-mib"] == "20480"
    assert train["nodeSelector"] == {"nvidia.com/gpu.product": "A100"}
    assert train["retryStrategy"]["limit"] == 2
    assert train["container"]["env"][0]["value"] == "{{workflow.parameters.baseModel}}"


def test_argo_script_template_keeps_its_source():
    argo = to_argo(training_workflow())
    evaluate = next(t for t in argo["spec"]["templates"] if t["name"] == "evaluate")
    assert evaluate["script"]["source"] == "print('eval')"


def test_npu_workflows_use_the_ascend_resource_key():
    document = workflow_dict(
        [{"name": "a", "template": "npu"}],
        [
            {
                "name": "npu",
                "container": {"image": "ascend/pytorch", "command": ["python", "train.py"]},
                "resources": {"accelerator": "npu", "gpu": 8},
            }
        ],
    )
    argo = to_argo(Workflow.from_dict(document))
    npu = next(t for t in argo["spec"]["templates"] if t["name"] == "npu")
    assert npu["container"]["resources"]["limits"]["huawei.com/Ascend910"] == 8


def test_pytorchjob_wires_replicas_and_world_size():
    job = to_pytorchjob(
        name="train",
        image="pytorch/pytorch:2.4.0",
        command=["torchrun", "train.py"],
        worker_replicas=3,
        gpu_per_replica=2,
    )
    assert job["apiVersion"] == "kubeflow.org/v1"
    assert job["kind"] == "PyTorchJob"
    replicas = job["spec"]["pytorchReplicaSpecs"]
    assert replicas["Worker"]["replicas"] == 3
    env = {e["name"]: e["value"] for e in replicas["Master"]["template"]["spec"]["containers"][0]["env"]}
    assert env["WORLD_SIZE"] == "4"
    assert env["MASTER_ADDR"] == "train-master-0"
    limits = replicas["Worker"]["template"]["spec"]["containers"][0]["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == 2


def test_manifests_bundle_includes_a_pytorchjob_for_training():
    manifests = to_kubernetes_manifests(training_workflow())
    kinds = [m["kind"] for m in manifests]
    assert kinds[0] == "Workflow"
    assert "PyTorchJob" in kinds


def test_compiled_argo_yaml_is_loadable():
    text = yaml.safe_dump(to_argo(training_workflow()))
    assert yaml.safe_load(text)["spec"]["entrypoint"] == "pipeline"
