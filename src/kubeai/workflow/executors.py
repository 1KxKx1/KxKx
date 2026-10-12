"""Step executors.

A workflow is portable across executors: the same document runs on a laptop
(:class:`LocalExecutor`), inside containers (:class:`DockerExecutor`) or on a
cluster (compiled to Argo — see ``kubeai.workflow.compile``). CI uses
:class:`DryRunExecutor`, which is deterministic and needs no images.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

from .spec import ContainerSpec, Template, substitute


class StepStatus(str, Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    SKIPPED = "Skipped"
    ERRORED = "Errored"

    @property
    def terminal(self) -> bool:
        return self in (StepStatus.SUCCEEDED, StepStatus.FAILED, StepStatus.SKIPPED,
                        StepStatus.ERRORED)

    @property
    def successful(self) -> bool:
        return self is StepStatus.SUCCEEDED


@dataclass
class StepResult:
    task: str
    status: StepStatus
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    duration: float = 0.0
    attempts: int = 1
    executor: str = "local"

    @property
    def ok(self) -> bool:
        return self.status is StepStatus.SUCCEEDED

    def to_dict(self, stdout_limit: int = 4000) -> dict:
        return {
            "task": self.task,
            "status": self.status.value,
            "exitCode": self.exit_code,
            "durationSeconds": round(self.duration, 3),
            "attempts": self.attempts,
            "executor": self.executor,
            "outputs": self.outputs,
            "stdout": self.stdout[-stdout_limit:],
            "stderr": self.stderr[-stdout_limit:],
        }


class Executor(Protocol):
    name: str

    async def execute(
        self,
        template: Template,
        task_name: str,
        workdir: Path,
        context: dict[str, str],
    ) -> StepResult: ...


def _materialise(
    container: ContainerSpec | None,
    template: Template,
    workdir: Path,
    context: dict[str, str],
) -> list[str]:
    """Return the argv to run, writing out a ``script`` body when present.

    Placeholders are substituted into the script source, not just into the argv —
    Argo does the same, and a pipeline that reads ``{{inputs.parameters.shard}}``
    inside its own script is the common case.
    """
    container = container or ContainerSpec()
    argv = [str(x) for x in container.argv()]

    if template.is_script:
        source = substitute(template.script or "", context)
        path = workdir / "main.py"
        path.write_text(source, encoding="utf-8")
        if argv:
            # `command` is the interpreter/driver; append the script we just wrote.
            argv = [*argv, path.name]
        else:
            argv = [sys.executable, path.name]
    return argv


_PYTHON_ALIASES = {"python", "python3", "python3.10", "python3.11", "python3.12", "py"}


def _resolve_interpreter(argv: list[str]) -> list[str]:
    """Point a bare ``python``/``python3`` at the interpreter running kubeai.

    A workflow document has to name its interpreter so that Argo knows how to run
    the ``script`` body on a cluster (``command: [python]``). Locally that would
    resolve through PATH to whatever Python happens to be installed — a different
    interpreter without kubeai in it. Since the local executor exists precisely to
    let the same document run on a laptop, resolving the alias to the current
    interpreter is the behaviour that makes the two paths agree.
    """
    if not argv:
        return argv
    head = os.path.basename(argv[0]).lower()
    if head in _PYTHON_ALIASES:
        return [sys.executable, *argv[1:]]
    return argv


class LocalExecutor:
    """Run steps as local subprocesses.

    This is the fast-iteration path: no image build, no pull, no cluster. It is
    also what the bundled examples execute through, so ``kubeai workflow run``
    does real work on a laptop.
    """

    name = "local"

    async def execute(
        self,
        template: Template,
        task_name: str,
        workdir: Path,
        context: dict[str, str],
    ) -> StepResult:
        workdir.mkdir(parents=True, exist_ok=True)
        container = template.container
        argv = _resolve_interpreter(_materialise(container, template, workdir, context))
        if not argv:
            return StepResult(
                task=task_name,
                status=StepStatus.ERRORED,
                exit_code=-1,
                stderr="template produced no command to run",
                executor=self.name,
            )

        # `command`/`args` carry the same placeholders the script body does.
        argv = [substitute(arg, context) for arg in argv]

        env = dict(os.environ)
        env.update(
            {k: substitute(v, context) for k, v in (container.env if container else {}).items()}
        )
        env["KUBEAI_TASK"] = task_name

        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(workdir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timeout = template.timeout_seconds
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            timed_out = False
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stdout_b, stderr_b = b"", f"step exceeded timeout of {timeout}s".encode()
            timed_out = True

        duration = time.monotonic() - started
        stdout = stdout_b.decode("utf-8", "replace")
        stderr = stderr_b.decode("utf-8", "replace")
        exit_code = -1 if timed_out else (proc.returncode or 0)

        outputs = _resolve_outputs(template, stdout, workdir)
        return StepResult(
            task=task_name,
            status=StepStatus.SUCCEEDED if exit_code == 0 else StepStatus.FAILED,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            outputs=outputs,
            duration=duration,
            executor=self.name,
        )


def _resolve_outputs(template: Template, stdout: str, workdir: Path) -> dict[str, str]:
    """Resolve declared outputs: ``{{stdout}}`` and ``{{file:path}}`` are supported."""
    resolved: dict[str, str] = {}
    for key, spec in template.outputs.items():
        spec = spec.strip()
        if spec in ("{{stdout}}", "stdout"):
            resolved[key] = stdout.strip().splitlines()[-1] if stdout.strip() else ""
        elif spec.startswith("{{file:") and spec.endswith("}}"):
            rel = spec[len("{{file:") : -2].strip()
            path = workdir / rel
            resolved[key] = path.read_text(encoding="utf-8").strip() if path.exists() else ""
        else:
            resolved[key] = spec
    return resolved


class DockerExecutor(LocalExecutor):
    """Run each step in a container via ``docker run``.

    Kept intentionally thin — the point is to keep the *same* workflow document
    portable between "run it on my laptop" and "run it on the cluster".
    """

    name = "docker"

    def __init__(self, docker_bin: str = "docker", network: str | None = None) -> None:
        self.docker_bin = docker_bin
        self.network = network

    async def execute(
        self,
        template: Template,
        task_name: str,
        workdir: Path,
        context: dict[str, str],
    ) -> StepResult:
        workdir.mkdir(parents=True, exist_ok=True)
        container = template.container or ContainerSpec()
        docker_argv = [self.docker_bin, "run", "--rm", "-v", f"{workdir}:/work", "-w", "/work"]
        if self.network:
            docker_argv += ["--network", self.network]
        if template.resources.gpu:
            docker_argv += ["--gpus", str(template.resources.gpu)]
        for key, value in (container.env or {}).items():
            docker_argv += ["-e", f"{key}={value}"]
        docker_argv += [container.image, *container.argv()]

        proxy = Template(
            name=template.name,
            container=ContainerSpec(image=container.image, command=docker_argv, args=[]),
            resources=template.resources,
            timeout_seconds=template.timeout_seconds,
            outputs=template.outputs,
        )
        if template.is_script:
            proxy.script = template.script
        result = await super().execute(proxy, task_name, workdir, context)
        result.executor = self.name
        return result


@dataclass
class DryRunExecutor:
    """Deterministic, instant executor for CI and demos.

    ``durations`` maps a template name to a simulated runtime; ``failures`` lists
    ``(task_name, attempt)`` pairs that should fail, which lets the retry logic be
    exercised without flakiness.
    """

    name: str = "dry-run"
    durations: dict[str, float] = field(default_factory=dict)
    default_duration: float = 0.01
    failures: set[tuple[str, int]] = field(default_factory=set)
    attempts_seen: dict[str, int] = field(default_factory=dict)

    async def execute(
        self,
        template: Template,
        task_name: str,
        workdir: Path,
        context: dict[str, str],
    ) -> StepResult:
        workdir.mkdir(parents=True, exist_ok=True)
        attempt = self.attempts_seen.get(task_name, 0) + 1
        self.attempts_seen[task_name] = attempt
        await asyncio.sleep(self.durations.get(template.name, self.default_duration))

        if (task_name, attempt) in self.failures:
            return StepResult(
                task=task_name,
                status=StepStatus.FAILED,
                exit_code=1,
                stderr=f"simulated failure on attempt {attempt}",
                duration=self.default_duration,
                attempts=attempt,
                executor=self.name,
            )

        outputs = {
            key: f"<dry-run {task_name}:{key}>" for key in template.outputs
        }
        return StepResult(
            task=task_name,
            status=StepStatus.SUCCEEDED,
            exit_code=0,
            stdout=f"[dry-run] template={template.name} task={task_name} argv={shlex.join(template.container.argv()) if template.container else ''}",
            outputs=outputs,
            duration=self.durations.get(template.name, self.default_duration),
            attempts=attempt,
            executor=self.name,
        )


EXECUTORS: dict[str, type] = {
    "local": LocalExecutor,
    "docker": DockerExecutor,
    "dry-run": DryRunExecutor,
}


def build_executor(kind: str, **kwargs) -> Executor:
    if kind not in EXECUTORS:
        raise ValueError(f"unknown executor {kind!r}; choose from {sorted(EXECUTORS)}")
    return EXECUTORS[kind](**kwargs)
