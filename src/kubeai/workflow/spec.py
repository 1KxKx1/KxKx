"""Workflow specification model.

The schema deliberately mirrors a *subset* of Argo Workflows so that

* a pipeline a data scientist writes here is readable by anyone who knows Argo,
* ``kubeai.workflow.compile`` can emit a real ``argoproj.io/v1alpha1`` resource
  without a lossy translation layer, and
* the same document runs locally through ``kubeai.workflow.engine`` for fast
  iteration (no cluster, no image pull) and on a cluster unchanged.

Example
-------
.. code-block:: yaml

    apiVersion: kubeai.io/v1alpha1
    kind: Workflow
    metadata:
      name: llm-finetune
    spec:
      entrypoint: pipeline
      arguments:
        parameters:
          - name: baseModel
            value: Qwen/Qwen2.5-7B-Instruct
      templates:
        - name: pipeline
          dag:
            tasks:
              - name: prepare
                template: data-prep
              - name: train
                template: train
                depends: prepare
        - name: train
          container:
            image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
            command: [torchrun, --nproc_per_node=2, train.py]
            resources:
              gpu: 2
              gpuMemoryMiB: 20480
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PARAM_PATTERN = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


class WorkflowValidationError(ValueError):
    """Raised when a workflow document is structurally invalid."""


@dataclass
class RetryStrategy:
    limit: int = 0
    retry_policy: str = "OnFailure"  # OnFailure | Always | OnError
    backoff_seconds: float = 1.0
    backoff_factor: float = 2.0
    max_backoff_seconds: float = 30.0

    @classmethod
    def from_dict(cls, data: dict | None) -> RetryStrategy:
        if not data:
            return cls()
        return cls(
            limit=int(data.get("limit", 0)),
            retry_policy=str(data.get("retryPolicy", data.get("retry_policy", "OnFailure"))),
            backoff_seconds=float(data.get("backoffSeconds", data.get("backoff_seconds", 1.0))),
            backoff_factor=float(data.get("backoffFactor", data.get("backoff_factor", 2.0))),
            max_backoff_seconds=float(
                data.get("maxBackoffSeconds", data.get("max_backoff_seconds", 30.0))
            ),
        )

    def delay_for(self, attempt: int) -> float:
        """Seconds to wait before ``attempt`` (1-based)."""
        return min(self.backoff_seconds * (self.backoff_factor ** max(attempt - 1, 0)),
                   self.max_backoff_seconds)


@dataclass
class ResourceSpec:
    cpu: str = "1"
    memory: str = "2Gi"
    gpu: int = 0
    gpu_memory_mib: int = 0
    #: gpu | npu — which accelerator pool the task should land in.
    accelerator_type: str = "gpu"

    @classmethod
    def from_dict(cls, data: dict | None) -> ResourceSpec:
        data = data or {}
        return cls(
            cpu=str(data.get("cpu", "1")),
            memory=str(data.get("memory", "2Gi")),
            gpu=int(data.get("gpu", data.get("nvidia.com/gpu", 0)) or 0),
            gpu_memory_mib=int(
                data.get("gpuMemoryMiB", data.get("gpu_memory_mib", 0)) or 0
            ),
            accelerator_type=str(
                data.get("accelerator", data.get("acceleratorType", "gpu"))
            ).lower(),
        )

    @property
    def cpu_millicores(self) -> int:
        return parse_cpu(self.cpu)

    @property
    def memory_mib(self) -> int:
        return parse_quantity_mib(self.memory)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"cpu": self.cpu, "memory": self.memory}
        if self.gpu:
            out["gpu"] = self.gpu
        if self.gpu_memory_mib:
            out["gpuMemoryMiB"] = self.gpu_memory_mib
        return out


def parse_cpu(value: str) -> int:
    """``"2"`` -> 2000, ``"500m"`` -> 500 (millicores)."""
    text = str(value).strip()
    if text.endswith("m"):
        return int(float(text[:-1]))
    return int(float(text) * 1000)


_UNITS_MIB = {
    "ki": 1 / 1024,
    "mi": 1,
    "gi": 1024,
    "ti": 1024 * 1024,
    "k": 1000 / (1024 * 1024),
    "m": 1_000_000 / (1024 * 1024),
    "g": 1000 * 1000 * 1000 / (1024 * 1024),
}


def parse_quantity_mib(value: str) -> int:
    """Kubernetes-style quantity to MiB. ``"4Gi"`` -> 4096, ``"512Mi"`` -> 512."""
    text = str(value).strip()
    match = re.fullmatch(r"([0-9.]+)\s*([a-zA-Z]*)", text)
    if not match:
        raise WorkflowValidationError(f"cannot parse quantity {value!r}")
    number, unit = float(match.group(1)), match.group(2).lower()
    if unit in _UNITS_MIB:
        return int(number * _UNITS_MIB[unit])
    if unit == "":
        return int(number / (1024 * 1024))  # bare bytes
    raise WorkflowValidationError(f"unknown unit in quantity {value!r}")


@dataclass
class ContainerSpec:
    image: str = "python:3.12-slim"
    command: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    working_dir: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> ContainerSpec:
        if not isinstance(data, dict):
            raise WorkflowValidationError("container must be a mapping")
        return cls(
            image=str(data.get("image", "python:3.12-slim")),
            command=[str(x) for x in _as_list(data.get("command"))],
            args=[str(x) for x in _as_list(data.get("args"))],
            env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
            working_dir=data.get("workingDir"),
        )

    def argv(self) -> list[str]:
        return [*self.command, *self.args]

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"image": self.image}
        if self.command:
            out["command"] = self.command
        if self.args:
            out["args"] = self.args
        if self.env:
            out["env"] = self.env
        if self.working_dir:
            out["workingDir"] = self.working_dir
        return out


@dataclass
class Template:
    """A reusable step: a container invocation or a script body."""

    name: str
    container: ContainerSpec | None = None
    script: str | None = None
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    retry: RetryStrategy = field(default_factory=RetryStrategy)
    timeout_seconds: int | None = None
    node_selector: dict[str, str] = field(default_factory=dict)
    #: Declared output parameters, e.g. ``{"result": "{{stdout}}"}``.
    outputs: dict[str, str] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    #: Populated for ``dag`` templates only.
    tasks: list[DAGTask] = field(default_factory=list)

    @property
    def is_dag(self) -> bool:
        return bool(self.tasks)

    @property
    def is_script(self) -> bool:
        return self.script is not None

    @classmethod
    def from_dict(cls, data: dict) -> Template:
        if "name" not in data:
            raise WorkflowValidationError("template requires a name")
        template = cls(name=str(data["name"]))
        template.resources = ResourceSpec.from_dict(data.get("resources"))
        template.retry = RetryStrategy.from_dict(data.get("retryStrategy"))
        if data.get("timeout"):
            template.timeout_seconds = int(_parse_duration(data["timeout"]))
        template.node_selector = {
            str(k): str(v) for k, v in (data.get("nodeSelector") or {}).items()
        }
        template.outputs = {str(k): str(v) for k, v in (data.get("outputs") or {}).items()}
        template.inputs = [str(p) for p in _as_list((data.get("inputs") or {}).get("parameters"))]

        if "dag" in data:
            dag = data["dag"] or {}
            tasks = dag.get("tasks") or []
            if not tasks:
                raise WorkflowValidationError(f"dag template {template.name!r} has no tasks")
            template.tasks = [DAGTask.from_dict(t) for t in tasks]
        elif "container" in data:
            template.container = ContainerSpec.from_dict(data["container"])
        elif "script" in data:
            script = data["script"]
            if isinstance(script, dict):
                template.container = ContainerSpec.from_dict(script)
                template.script = str(script.get("source", ""))
            else:
                template.script = str(script)
        else:
            raise WorkflowValidationError(
                f"template {template.name!r} must define one of: dag, container, script"
            )
        return template

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"name": self.name}
        if self.is_dag:
            out["dag"] = {"tasks": [t.to_dict() for t in self.tasks]}
        elif self.is_script:
            container = self.container.to_dict() if self.container else {}
            container["source"] = self.script
            out["script"] = container
        elif self.container:
            out["container"] = self.container.to_dict()
        if self.resources != ResourceSpec():
            out["resources"] = self.resources.to_dict()
        if self.retry.limit:
            out["retryStrategy"] = {
                "limit": self.retry.limit,
                "retryPolicy": self.retry.retry_policy,
                "backoffSeconds": self.retry.backoff_seconds,
                "backoffFactor": self.retry.backoff_factor,
            }
        if self.timeout_seconds:
            out["timeout"] = f"{self.timeout_seconds}s"
        if self.node_selector:
            out["nodeSelector"] = self.node_selector
        if self.outputs:
            out["outputs"] = self.outputs
        return out


@dataclass
class DAGTask:
    name: str
    template: str
    depends: list[str] = field(default_factory=list)
    when: str | None = None
    arguments: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> DAGTask:
        if "name" not in data or "template" not in data:
            raise WorkflowValidationError("dag task requires name and template")
        depends = _as_list(data.get("depends") or data.get("dependencies"))
        arguments: dict[str, str] = {}
        for param in _as_list((data.get("arguments") or {}).get("parameters")):
            if isinstance(param, dict) and "name" in param:
                arguments[str(param["name"])] = str(param.get("value", ""))
        return cls(
            name=str(data["name"]),
            template=str(data["template"]),
            depends=[str(d) for d in depends],
            when=data.get("when"),
            arguments=arguments,
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"name": self.name, "template": self.template}
        if self.depends:
            out["depends"] = self.depends
        if self.when:
            out["when"] = self.when
        if self.arguments:
            out["arguments"] = {
                "parameters": [{"name": k, "value": v} for k, v in self.arguments.items()]
            }
        return out


@dataclass
class Workflow:
    name: str
    entrypoint: str
    templates: list[Template] = field(default_factory=list)
    namespace: str = "ai-platform"
    arguments: dict[str, str] = field(default_factory=dict)
    ttl_seconds: int | None = None
    labels: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def template(self, name: str) -> Template:
        for t in self.templates:
            if t.name == name:
                return t
        raise WorkflowValidationError(f"template {name!r} not found in workflow {self.name!r}")

    @property
    def entry(self) -> Template:
        return self.template(self.entrypoint)

    def validate(self) -> None:
        """Fail fast on the mistakes that would otherwise surface mid-run."""
        if not self.name:
            raise WorkflowValidationError("workflow requires a name")
        if not self.entrypoint:
            raise WorkflowValidationError("workflow requires an entrypoint")
        names = [t.name for t in self.templates]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise WorkflowValidationError(f"duplicate template names: {sorted(duplicates)}")
        if self.entrypoint not in names:
            raise WorkflowValidationError(f"entrypoint {self.entrypoint!r} is not a template")

        for template in self.templates:
            if not template.is_dag:
                continue
            task_names = [t.name for t in template.tasks]
            dupes = {n for n in task_names if task_names.count(n) > 1}
            if dupes:
                raise WorkflowValidationError(
                    f"dag {template.name!r} has duplicate task names: {sorted(dupes)}"
                )
            known = set(task_names)
            for task in template.tasks:
                unknown = [d for d in task.depends if d not in known]
                if unknown:
                    raise WorkflowValidationError(
                        f"dag {template.name!r} task {task.name!r} depends on unknown "
                        f"tasks {unknown}"
                    )
                if task.template not in names:
                    raise WorkflowValidationError(
                        f"dag {template.name!r} task {task.name!r} references unknown "
                        f"template {task.template!r}"
                    )
            self._assert_acyclic(template)

    @staticmethod
    def _assert_acyclic(template: Template) -> None:
        graph = {t.name: list(t.depends) for t in template.tasks}
        state: dict[str, int] = {}

        def visit(node: str, stack: list[str]) -> None:
            if state.get(node) == 2:
                return
            if state.get(node) == 1:
                cycle = " -> ".join([*stack, node])
                raise WorkflowValidationError(f"cycle detected in dag {template.name!r}: {cycle}")
            state[node] = 1
            for dep in graph.get(node, []):
                visit(dep, stack + [node])
            state[node] = 2

        for node in graph:
            visit(node, [])

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict) -> Workflow:
        if not isinstance(data, dict):
            raise WorkflowValidationError("workflow document must be a mapping")
        metadata = data.get("metadata") or {}
        spec = data.get("spec") or {}
        arguments: dict[str, str] = {}
        for param in _as_list((spec.get("arguments") or {}).get("parameters")):
            if isinstance(param, dict) and "name" in param:
                arguments[str(param["name"])] = str(param.get("value", ""))
        templates = [Template.from_dict(t) for t in (spec.get("templates") or [])]
        workflow = cls(
            name=str(metadata.get("name", "")),
            namespace=str(metadata.get("namespace", "ai-platform")),
            entrypoint=str(spec.get("entrypoint", "")),
            templates=templates,
            arguments=arguments,
            ttl_seconds=int(spec["ttlSecondsAfterFinished"])
            if spec.get("ttlSecondsAfterFinished")
            else None,
            labels={str(k): str(v) for k, v in (metadata.get("labels") or {}).items()},
        )
        workflow.validate()
        return workflow

    @classmethod
    def load(cls, path: str | Path) -> Workflow:
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_yaml(text)

    @classmethod
    def from_yaml(cls, text: str) -> Workflow:
        data = yaml.safe_load(text)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return {
            "apiVersion": "kubeai.io/v1alpha1",
            "kind": "Workflow",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                **({"labels": self.labels} if self.labels else {}),
            },
            "spec": {
                "entrypoint": self.entrypoint,
                **(
                    {
                        "arguments": {
                            "parameters": [
                                {"name": k, "value": v} for k, v in self.arguments.items()
                            ]
                        }
                    }
                    if self.arguments
                    else {}
                ),
                **({"ttlSecondsAfterFinished": self.ttl_seconds} if self.ttl_seconds else {}),
                "templates": [t.to_dict() for t in self.templates],
            },
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _parse_duration(value: Any) -> float:
    """``"30s"``/``"5m"``/``90`` -> seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.fullmatch(r"([0-9.]+)\s*([smhd]?)", text)
    if not match:
        raise WorkflowValidationError(f"cannot parse duration {value!r}")
    number, unit = float(match.group(1)), match.group(2)
    return number * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def substitute(text: str, values: dict[str, str]) -> str:
    """Replace ``{{ key }}`` placeholders; unknown keys are left untouched."""
    if not isinstance(text, str):
        return text

    def repl(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return values.get(key, match.group(0))

    return PARAM_PATTERN.sub(repl, text)
