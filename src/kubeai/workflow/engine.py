"""DAG execution engine.

Resolves the entrypoint, runs ready tasks concurrently up to a parallelism
budget, honours ``depends`` / ``when`` gates, retries with exponential backoff and
threads parameters between steps (``{{tasks.train.outputs.modelPath}}``).

The engine is deliberately storage-agnostic: every step gets a directory under
the run directory, which is what the local executor bind-mounts and what an
artifact repository would back on a cluster.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .executors import Executor, StepResult, StepStatus, build_executor
from .spec import DAGTask, Template, Workflow, WorkflowValidationError, substitute

log = logging.getLogger(__name__)


class WorkflowFailed(RuntimeError):
    """Raised by :meth:`WorkflowEngine.run_or_raise` when the run did not succeed."""


@dataclass
class WorkflowRun:
    name: str
    run_id: str
    status: StepStatus
    started_at: float
    finished_at: float = 0.0
    steps: dict[str, StepResult] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    parameters: dict[str, str] = field(default_factory=dict)
    executor: str = "local"

    @property
    def duration(self) -> float:
        return max(self.finished_at - self.started_at, 0.0)

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.steps.values():
            counts[result.status.value] = counts.get(result.status.value, 0) + 1
        return counts

    def critical_path(self) -> float:
        """Wall-clock time of the slowest chain — the number an operator tunes."""
        return sum(r.duration for r in self.steps.values())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "runId": self.run_id,
            "status": self.status.value,
            "executor": self.executor,
            "durationSeconds": round(self.duration, 3),
            "sumOfStepSeconds": round(self.critical_path(), 3),
            "counts": self.counts(),
            "parameters": self.parameters,
            "order": self.order,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
        }


class WorkflowEngine:
    def __init__(
        self,
        workflow: Workflow,
        executor: Executor | str = "local",
        workdir: str | Path = ".kubeai/runs",
        max_parallel: int = 4,
        parameters: dict[str, str] | None = None,
    ) -> None:
        workflow.validate()
        self.workflow = workflow
        self.executor: Executor = (
            build_executor(executor) if isinstance(executor, str) else executor
        )
        self.max_parallel = max(1, max_parallel)
        self.parameters = {**workflow.arguments, **(parameters or {})}
        self.run_id = uuid.uuid4().hex[:12]
        self.run_dir = Path(workdir) / f"{workflow.name}-{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._results: dict[str, StepResult] = {}
        self._order: list[str] = []
        self._semaphore = asyncio.Semaphore(self.max_parallel)

    # ------------------------------------------------------------------ #
    # Context for parameter substitution
    # ------------------------------------------------------------------ #
    def _context(self, task: DAGTask | None) -> dict[str, str]:
        context = {
            "workflow.name": self.workflow.name,
            "workflow.namespace": self.workflow.namespace,
            "workflow.uid": self.run_id,
            "workflow.parameters": "",
        }
        for key, value in self.parameters.items():
            context[f"workflow.parameters.{key}"] = value
            context[f"inputs.parameters.{key}"] = value
        for name, result in self._results.items():
            context[f"tasks.{name}.status"] = result.status.value
            for out_key, out_value in result.outputs.items():
                context[f"tasks.{name}.outputs.{out_key}"] = out_value
        if task is not None:
            for key, value in task.arguments.items():
                context[f"inputs.parameters.{key}"] = substitute(value, context)
        return context

    # ------------------------------------------------------------------ #
    # Gates
    # ------------------------------------------------------------------ #
    @staticmethod
    def _evaluate_when(expression: str, context: dict[str, str]) -> bool:
        """A deliberately small subset of Argo's ``when`` expressions.

        Supports ``{{a.b}} == value``, ``!=``, and the bare ``Succeeded``/``Failed``
        shorthand. Anything unparseable evaluates to ``True`` so a typo degrades to
        "run the step" rather than silently skipping work.
        """
        expr = substitute(expression, context).strip()
        if not expr:
            return True
        if expr in {"Succeeded", "true", "True"}:
            return True
        if expr in {"Failed", "false", "False"}:
            return False
        for operator in ("==", "!="):
            if operator in expr:
                left, _, right = expr.partition(operator)
                left, right = left.strip(), right.strip().strip("\"'")
                return (left == right) if operator == "==" else (left != right)
        return True

    def _dependencies_satisfied(self, task: DAGTask) -> bool:
        return all(dep in self._results for dep in task.depends)

    def _dependency_blocked(self, task: DAGTask) -> bool:
        """True when an upstream step did not succeed, so this one must skip."""
        for dep in task.depends:
            result = self._results.get(dep)
            if result is not None and not result.ok:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    async def _run_task_with_retries(
        self, task: DAGTask, template: Template, workdir: Path
    ) -> StepResult:
        retry = template.retry
        attempts = 0
        last: StepResult | None = None

        while attempts <= retry.limit:
            attempts += 1
            context = self._context(task)
            async with self._semaphore:
                result = await self.executor.execute(template, task.name, workdir, context)
            result.attempts = attempts
            result.task = task.name
            last = result

            if result.ok:
                return result

            should_retry = attempts <= retry.limit and retry.retry_policy in ("OnFailure", "Always")
            if not should_retry:
                return result

            delay = retry.delay_for(attempts)
            log.warning(
                "task %s failed (attempt %d/%d), retrying in %.2fs",
                task.name,
                attempts,
                retry.limit + 1,
                delay,
            )
            await asyncio.sleep(delay)

        assert last is not None
        return last

    async def _execute_dag(self, template: Template) -> dict[str, StepResult]:
        pending: dict[str, DAGTask] = {t.name: t for t in template.tasks}
        running: dict[str, asyncio.Task] = {}

        while pending or running:
            # Admit everything that is ready right now.
            ready: list[DAGTask] = []
            for name, task in list(pending.items()):
                if name in running or not self._dependencies_satisfied(task):
                    continue
                ready.append(task)
                del pending[name]

            for task in ready:
                if self._dependency_blocked(task):
                    self._results[task.name] = StepResult(
                        task=task.name,
                        status=StepStatus.SKIPPED,
                        stderr="upstream dependency did not succeed",
                        executor=self.executor.name,
                    )
                    self._order.append(task.name)
                    continue
                if task.when and not self._evaluate_when(task.when, self._context(task)):
                    self._results[task.name] = StepResult(
                        task=task.name,
                        status=StepStatus.SKIPPED,
                        stderr=f"when condition false: {task.when}",
                        executor=self.executor.name,
                    )
                    self._order.append(task.name)
                    continue

                step_template = self.workflow.template(task.template)
                workdir = self.run_dir / task.name
                running[task.name] = asyncio.create_task(
                    self._run_task_with_retries(task, step_template, workdir)
                )
                self._order.append(task.name)

            if not running:
                if pending:
                    # Nothing runnable and nothing running => unsatisfiable graph.
                    for name in pending:
                        self._results[name] = StepResult(
                            task=name,
                            status=StepStatus.ERRORED,
                            exit_code=-1,
                            stderr="dependency graph is unsatisfiable",
                            executor=self.executor.name,
                        )
                        self._order.append(name)
                    pending.clear()
                break

            done, _ = await asyncio.wait(running.values(), return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                name = next(n for n, t in running.items() if t is finished)
                del running[name]
                self._results[name] = finished.result()

        return self._results

    async def run(self) -> WorkflowRun:
        started = time.time()
        entry = self.workflow.entry

        try:
            if entry.is_dag:
                await self._execute_dag(entry)
            else:
                workdir = self.run_dir / entry.name
                result = await self._run_task_with_retries(
                    DAGTask(name=entry.name, template=entry.name), entry, workdir
                )
                self._results[entry.name] = result
                self._order.append(entry.name)
        except Exception as exc:  # noqa: BLE001 — surfaced as an Errored run
            log.exception("workflow %s crashed", self.workflow.name)
            self._results.setdefault(
                "__engine__",
                StepResult(
                    task="__engine__",
                    status=StepStatus.ERRORED,
                    exit_code=-1,
                    stderr=f"{type(exc).__name__}: {exc}",
                    executor=self.executor.name,
                ),
            )

        if any(r.status is StepStatus.ERRORED for r in self._results.values()):
            status = StepStatus.ERRORED
        elif any(r.status is StepStatus.FAILED for r in self._results.values()):
            status = StepStatus.FAILED
        else:
            status = StepStatus.SUCCEEDED

        return WorkflowRun(
            name=self.workflow.name,
            run_id=self.run_id,
            status=status,
            started_at=started,
            finished_at=time.time(),
            steps=dict(self._results),
            order=list(self._order),
            parameters=dict(self.parameters),
            executor=self.executor.name,
        )

    async def run_or_raise(self) -> WorkflowRun:
        run = await self.run()
        if run.status is not StepStatus.SUCCEEDED:
            failed = [n for n, r in run.steps.items() if not r.ok]
            raise WorkflowFailed(f"workflow {run.name!r} finished {run.status.value}: {failed}")
        return run


def run_workflow(
    workflow: Workflow | str | Path,
    executor: Executor | str = "local",
    workdir: str | Path = ".kubeai/runs",
    max_parallel: int = 4,
    parameters: dict[str, str] | None = None,
) -> WorkflowRun:
    """Blocking convenience wrapper around :class:`WorkflowEngine`."""
    if isinstance(workflow, (str, Path)):
        workflow = Workflow.load(workflow)
    if not isinstance(workflow, Workflow):
        raise WorkflowValidationError("workflow must be a Workflow, path or YAML string")
    engine = WorkflowEngine(
        workflow,
        executor=executor,
        workdir=workdir,
        max_parallel=max_parallel,
        parameters=parameters,
    )
    return asyncio.run(engine.run())
