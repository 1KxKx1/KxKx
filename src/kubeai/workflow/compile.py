"""Compile a kubeai workflow into resources a cluster can actually run.

``to_argo`` produces a real ``argoproj.io/v1alpha1`` ``Workflow``: GPU requests
become ``nvidia.com/gpu`` limits, ``retryStrategy``/``nodeSelector``/``outputs``
carry over, and DAG dependencies become ``depends``. ``to_pytorchjob`` produces a
Kubeflow ``PyTorchJob`` for distributed training, which is how a ``finetune``
template is expected to be deployed.
"""

from __future__ import annotations

from typing import Any

import yaml

from .spec import ContainerSpec, DAGTask, ResourceSpec, Template, Workflow

GPU_RESOURCE_KEY = "nvidia.com/gpu"
NPU_RESOURCE_KEY = "huawei.com/Ascend910"


def _accelerator_key(resources: ResourceSpec) -> str:
    return NPU_RESOURCE_KEY if resources.accelerator_type == "npu" else GPU_RESOURCE_KEY


def _argo_resources(resources: ResourceSpec) -> dict[str, dict[str, Any]]:
    limits: dict[str, Any] = {}
    requests: dict[str, Any] = {}

    if resources.cpu:
        requests["cpu"] = resources.cpu
        limits.setdefault("cpu", resources.cpu)
    if resources.memory:
        requests["memory"] = resources.memory
        limits.setdefault("memory", resources.memory)
    if resources.gpu:
        key = _accelerator_key(resources)
        limits[key] = resources.gpu
        requests[key] = resources.gpu
        if resources.gpu_memory_mib:
            # Not a scheduler primitive in Kubernetes; surfaced as an annotation
            # that our own scheduler reads back (see kubeai.scheduler).
            limits["kubeai.io/gpu-memory-mib"] = str(resources.gpu_memory_mib)

    out: dict[str, dict[str, Any]] = {}
    if requests:
        out["requests"] = requests
    if limits:
        out["limits"] = limits
    return out


def _argo_template(template: Template) -> dict[str, Any]:
    node: dict[str, Any] = {"name": template.name}

    if template.is_dag:
        node["dag"] = {"tasks": [_argo_dag_task(t) for t in template.tasks]}
    elif template.is_script:
        container = template.container or ContainerSpec()
        script: dict[str, Any] = {
            "image": container.image,
            "source": template.script,
        }
        if container.command:
            script["command"] = container.command
        if container.env:
            script["env"] = [{"name": k, "value": v} for k, v in container.env.items()]
        if template.resources != ResourceSpec():
            script["resources"] = _argo_resources(template.resources)
        node["script"] = script
    elif template.container:
        container = template.container
        spec: dict[str, Any] = {"image": container.image}
        if container.command:
            spec["command"] = container.command
        if container.args:
            spec["args"] = container.args
        if container.env:
            spec["env"] = [{"name": k, "value": v} for k, v in container.env.items()]
        if container.working_dir:
            spec["workingDir"] = container.working_dir
        if template.resources != ResourceSpec():
            spec["resources"] = _argo_resources(template.resources)
        node["container"] = spec

    if template.retry.limit:
        node["retryStrategy"] = {
            "limit": template.retry.limit,
            "retryPolicy": template.retry.retry_policy,
            "backoff": {
                "duration": f"{int(template.retry.backoff_seconds)}s",
                "factor": int(template.retry.backoff_factor),
                "maxDuration": f"{int(template.retry.max_backoff_seconds)}s",
            },
        }
    if template.timeout_seconds:
        node["activeDeadlineSeconds"] = template.timeout_seconds
    if template.node_selector:
        node["nodeSelector"] = template.node_selector
    if template.outputs:
        node["outputs"] = {
            "parameters": [
                {"name": key, "valueFrom": {"path": _output_path(value)}}
                for key, value in template.outputs.items()
            ]
        }
    if template.inputs:
        node["inputs"] = {
            "parameters": [{"name": name} for name in template.inputs]
        }
    return node


def _output_path(declaration: str) -> str:
    """Translate a kubeai output declaration into an Argo output path.

    ``{{file:/work/metrics.json}}`` -> ``/work/metrics.json``; anything else falls
    back to a conventional location so the emitted document stays valid YAML.
    """
    declaration = declaration.strip()
    if declaration.startswith("{{file:") and declaration.endswith("}}"):
        return declaration[len("{{file:") : -2].strip()
    return "/tmp/kubeai-output"


def _argo_dag_task(task: DAGTask) -> dict[str, Any]:
    node: dict[str, Any] = {"name": task.name, "template": task.template}
    if task.depends:
        node["depends"] = " && ".join(task.depends) if len(task.depends) > 1 else task.depends[0]
    if task.when:
        node["when"] = task.when
    if task.arguments:
        node["arguments"] = {
            "parameters": [{"name": k, "value": v} for k, v in task.arguments.items()]
        }
    return node


def to_argo(
    workflow: Workflow,
    service_account: str | None = "kubeai-workflow",
    artifact_repository: str | None = None,
    image_pull_secrets: list[str] | None = None,
) -> dict[str, Any]:
    workflow.validate()

    labels = {
        "app.kubernetes.io/name": "kubeai",
        "app.kubernetes.io/managed-by": "kubeai-platform",
        "kubeai.io/workflow": workflow.name,
        **workflow.labels,
    }
    spec: dict[str, Any] = {
        "entrypoint": workflow.entrypoint,
        "templates": [_argo_template(t) for t in workflow.templates],
    }
    if workflow.arguments:
        spec["arguments"] = {
            "parameters": [{"name": k, "value": v} for k, v in workflow.arguments.items()]
        }
    if workflow.ttl_seconds:
        spec["ttlStrategy"] = {"secondsAfterCompletion": workflow.ttl_seconds}
    if service_account:
        spec["serviceAccountName"] = service_account
    if artifact_repository:
        spec["artifactRepositoryRef"] = {"configMap": artifact_repository}
    if image_pull_secrets:
        spec["imagePullSecrets"] = [{"name": name} for name in image_pull_secrets]

    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {"name": workflow.name, "namespace": workflow.namespace, "labels": labels},
        "spec": spec,
    }


def argo_yaml(workflow: Workflow, **kwargs: Any) -> str:
    return yaml.safe_dump(to_argo(workflow, **kwargs), sort_keys=False, allow_unicode=True)


# --------------------------------------------------------------------------- #
# Kubeflow
# --------------------------------------------------------------------------- #
def to_pytorchjob(
    name: str,
    image: str,
    command: list[str],
    args: list[str] | None = None,
    namespace: str = "ai-platform",
    master_replicas: int = 1,
    worker_replicas: int = 3,
    gpu_per_replica: int = 1,
    gpu_memory_mib: int = 0,
    cpu: str = "8",
    memory: str = "32Gi",
    env: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """A Kubeflow ``PyTorchJob`` for distributed fine-tuning.

    ``nnodes``/``nproc_per_node`` are wired to the replica counts so ``torchrun``
    sizes itself correctly without the user doing arithmetic.
    """
    resources = ResourceSpec(
        cpu=cpu, memory=memory, gpu=gpu_per_replica, gpu_memory_mib=gpu_memory_mib
    )
    argo_resources = _argo_resources(resources)
    env_list = [{"name": k, "value": v} for k, v in (env or {}).items()]
    env_list += [
        {"name": "MASTER_ADDR", "value": f"{name}-master-0"},
        {"name": "MASTER_PORT", "value": "29500"},
        {"name": "WORLD_SIZE", "value": str(master_replicas + worker_replicas)},
    ]
    container = {
        "image": image,
        "command": command,
        **({"args": args} if args else {}),
        "env": env_list,
        "resources": argo_resources,
    }
    replica_spec = {
        "replicas": 1,
        "restartPolicy": "OnFailure",
        "template": {"spec": {"containers": [container]}},
    }
    worker_spec = {
        "replicas": worker_replicas,
        "restartPolicy": "OnFailure",
        "template": {"spec": {"containers": [container]}},
    }
    return {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {"app.kubernetes.io/managed-by": "kubeai-platform", **(labels or {})},
        },
        "spec": {
            "runPolicy": {"cleanPodPolicy": "Running"},
            "pytorchReplicaSpecs": {"Master": replica_spec, "Worker": worker_spec},
        },
    }


def to_kubernetes_manifests(workflow: Workflow) -> list[dict[str, Any]]:
    """Everything needed to run ``workflow`` on a cluster, in apply order.

    A training workflow additionally gets a ``PyTorchJob`` for its ``finetune`` /
    ``train`` template so the heavy step is not a single pod.
    """
    workflow.validate()
    manifests: list[dict[str, Any]] = [to_argo(workflow)]

    train_template = next(
        (
            t
            for t in workflow.templates
            if t.container and t.resources.gpu > 0 and "train" in t.name.lower()
        ),
        None,
    )
    if train_template and train_template.container:
        manifests.append(
            to_pytorchjob(
                name=f"{workflow.name}-train",
                namespace=workflow.namespace,
                image=train_template.container.image,
                command=train_template.container.command or ["torchrun"],
                args=train_template.container.args,
                gpu_per_replica=train_template.resources.gpu,
                gpu_memory_mib=train_template.resources.gpu_memory_mib,
                cpu=train_template.resources.cpu,
                memory=train_template.resources.memory,
                labels={"kubeai.io/workflow": workflow.name},
            )
        )
    return manifests


def manifests_yaml(workflow: Workflow) -> str:
    return yaml.safe_dump_all(
        to_kubernetes_manifests(workflow), sort_keys=False, allow_unicode=True
    )
