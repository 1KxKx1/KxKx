"""Continuous (in-flight) batching scheduler.

The classic mistake is *static* batching: collect N requests, run them together
until **all** finish, then start the next batch. That wastes the accelerator
twice over — a request that finishes early holds its slot idle, and a request
that arrives just after a batch starts waits for the whole batch to drain.

Continuous batching fixes both. The batch is rebuilt on **every decode step**:

* a sequence that emits EOS is retired immediately and its slot is handed to a
  queued request on the very next step;
* new arrivals join the running batch without waiting for anyone else.

This module implements that loop, plus the streaming, cancellation and metrics a
service needs around it. Backend work runs in a worker thread so NumPy/BLAS can
use the cores without blocking the event loop that serves HTTP.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

from .backends import Backend, CharTokenizer, EOS, Sequence
from .metrics import ServingMetrics

log = logging.getLogger(__name__)


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    ERROR = "error"


class QueueFull(RuntimeError):
    """Raised when the admission queue is at capacity (backpressure)."""


@dataclass
class GenerationRequest:
    """One client request and everything we learn while serving it."""

    prompt: str
    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    stream: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- runtime state --- #
    sequence: Sequence | None = None
    output_text: str = ""
    finish_reason: FinishReason | None = None
    error: str | None = None

    submitted_at: float = field(default_factory=time.time)
    prefill_at: float | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    finished_at: float | None = None
    batch_size_seen: int = 0

    _subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)
    _cancel: bool = False

    # ------------------------------------------------------------------ #
    @property
    def output_tokens(self) -> int:
        return len(self.sequence.output_tokens) if self.sequence else 0

    @property
    def prompt_tokens(self) -> int:
        return len(self.sequence.prompt_tokens) if self.sequence else 0

    @property
    def latency(self) -> float:
        end = self.finished_at or time.time()
        return end - self.submitted_at

    @property
    def ttft(self) -> float | None:
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.submitted_at

    @property
    def tpot(self) -> float | None:
        """Mean inter-token latency, excluding the first token."""
        if self.first_token_at is None or self.output_tokens < 2:
            return None
        end = self.finished_at or time.time()
        return (end - self.first_token_at) / (self.output_tokens - 1)

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None

    def cancel(self) -> None:
        self._cancel = True

    # --- streaming ------------------------------------------------------ #
    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def _publish(self, event: dict | None) -> None:
        for queue in self._subscribers:
            queue.put_nowait(event)

    def to_dict(self, include_text: bool = True) -> dict:
        payload = {
            "id": self.id,
            "finish_reason": self.finish_reason.value if self.finish_reason else None,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "latency_seconds": round(self.latency, 4),
            "ttft_seconds": round(self.ttft, 4) if self.ttft is not None else None,
            "tpot_seconds": round(self.tpot, 6) if self.tpot is not None else None,
            "batch_size_seen": self.batch_size_seen,
        }
        if include_text:
            payload["text"] = self.output_text
        if self.error:
            payload["error"] = self.error
        return payload


def sample_token(
    logits: np.ndarray, temperature: float, top_p: float, rng: np.random.Generator
) -> int:
    """Greedy when ``temperature <= 0``, otherwise temperature + nucleus sampling."""
    if temperature <= 0:
        return int(np.argmax(logits))

    scaled = logits.astype(np.float64) / max(temperature, 1e-6)
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()

    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        # Keep the smallest prefix whose mass reaches top_p.
        cutoff = int(np.searchsorted(cumulative, top_p) + 1)
        keep = order[:cutoff]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        total = probs.sum()
        probs = probs / total if total > 0 else probs

    return int(rng.choice(len(probs), p=probs))


class ContinuousBatcher:
    """Runs a :class:`~kubeai.serving.backends.Backend` with in-flight batching."""

    def __init__(
        self,
        backend: Backend,
        tokenizer: CharTokenizer | None = None,
        max_batch_size: int = 32,
        max_batch_tokens: int = 4096,
        max_queue_size: int = 2048,
        metrics: ServingMetrics | None = None,
        seed: int = 1234,
        use_thread: bool = True,
        idle_sleep: float = 0.0005,
        max_prefills_per_step: int = 4,
        throughput_window: float = 5.0,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.backend = backend
        # Prefer the backend's own tokenizer: its vocabulary must match the
        # embedding table, or token ids would index out of range.
        self.tokenizer = tokenizer or getattr(backend, "tokenizer", None) or CharTokenizer()
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.max_queue_size = max_queue_size
        self.metrics = (metrics or ServingMetrics()).install()
        self.use_thread = use_thread
        self.idle_sleep = idle_sleep
        self.max_prefills_per_step = max_prefills_per_step
        self.throughput_window = throughput_window

        self._rng = np.random.default_rng(seed)
        self._waiting: deque[GenerationRequest] = deque()
        self._active: list[GenerationRequest] = []
        self._running = False
        self._task: asyncio.Task | None = None
        self._step_lock = asyncio.Lock()

        # throughput accounting
        self._token_events: deque[tuple[float, int]] = deque()
        self._steps = 0
        self._admitted = 0
        self.batch_size_history: list[int] = []

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="kubeai-batcher")

    async def stop(self, drain: bool = False, timeout: float = 10.0) -> None:
        self._running = False
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        for request in [*self._active, *self._waiting]:
            if not request.finished:
                self._finish(request, FinishReason.CANCELLED, "batcher stopped")
        self._active.clear()
        self._waiting.clear()

    async def __aenter__(self) -> ContinuousBatcher:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------ #
    # Submission
    # ------------------------------------------------------------------ #
    def submit_nowait(self, request: GenerationRequest) -> GenerationRequest:
        if len(self._waiting) >= self.max_queue_size:
            raise QueueFull(
                f"inference queue is full ({len(self._waiting)}/{self.max_queue_size})"
            )
        self._waiting.append(request)
        self.metrics.queue_depth.set(len(self._waiting))
        return request

    async def submit(self, request: GenerationRequest) -> GenerationRequest:
        if not self._running:
            await self.start()
        self.submit_nowait(request)
        while not request.finished:
            await asyncio.sleep(0.0005)
        return request

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerationRequest:
        request = GenerationRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            **kwargs,
        )
        return await self.submit(request)

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #
    async def _run_loop(self) -> None:
        try:
            while self._running:
                admitted = await self._admit()
                if not self._active:
                    if not admitted:
                        await asyncio.sleep(self.idle_sleep)
                    continue
                await self._decode_step()
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        except Exception:  # pragma: no cover - defensive
            log.exception("batcher loop crashed")
            for request in [*self._active, *self._waiting]:
                if not request.finished:
                    self._finish(request, FinishReason.ERROR, "batcher crashed")
            self._active.clear()
            self._waiting.clear()

    async def _admit(self) -> int:
        """Prefill queued requests while the batch has room. Returns how many joined."""
        admitted = 0
        while (
            self._waiting
            and len(self._active) < self.max_batch_size
            and admitted < self.max_prefills_per_step
        ):
            used = sum(r.output_tokens + r.prompt_tokens for r in self._active)
            candidate = self._waiting[0]
            if used >= self.max_batch_tokens and self._active:
                break
            self._waiting.popleft()
            self.metrics.queue_depth.set(len(self._waiting))
            self.metrics.queue_wait.observe(time.time() - candidate.submitted_at)

            if candidate._cancel:
                self._finish(candidate, FinishReason.CANCELLED, "cancelled before prefill")
                continue

            prompt = candidate.prompt
            if not prompt:
                # A bare generation request: seed with BOS so the model has context.
                prompt = ""
            sequence = Sequence(
                id=candidate.id,
                prompt_tokens=self.tokenizer.encode(prompt),
                max_tokens=candidate.max_tokens,
                temperature=candidate.temperature,
                top_p=candidate.top_p,
                stop=candidate.stop,
            )
            candidate.sequence = sequence
            candidate.prefill_at = time.time()

            try:
                logits = await self._run(self.backend.prefill, sequence)
            except Exception as exc:  # noqa: BLE001 — a bad request must not kill the loop
                self._finish(candidate, FinishReason.ERROR, f"prefill failed: {exc}")
                continue

            self.metrics.prompt_tokens_total.inc(sequence.prompt_tokens and len(sequence.prompt_tokens) or 0)

            token = sample_token(logits, candidate.temperature, candidate.top_p, self._rng)
            sequence.output_tokens.append(token)
            candidate.first_token_at = time.time()
            self._emit(candidate, token, first=True)

            if self._should_stop(candidate):
                self._finish(candidate, *self._stop_reason(candidate))
            else:
                self._active.append(candidate)
            admitted += 1

        if self._active:
            self.metrics.active_sequences.set(len(self._active))
        return admitted

    async def _decode_step(self) -> None:
        batch = list(self._active)
        if not batch:
            return

        sequences = [r.sequence for r in batch if r.sequence is not None]
        try:
            logits_list = await self._run(self.backend.decode, sequences)
        except Exception as exc:  # noqa: BLE001
            for request in batch:
                self._finish(request, FinishReason.ERROR, f"decode failed: {exc}")
            self._active = [r for r in self._active if not r.finished]
            return

        self._steps += 1
        self.batch_size_history.append(len(batch))
        self.metrics.batch_size.observe(len(batch))

        now = time.time()
        for request, logits in zip(batch, logits_list, strict=True):
            sequence = request.sequence
            if sequence is None or request.finished:
                continue
            request.batch_size_seen = max(request.batch_size_seen, len(batch))

            token = sample_token(logits, request.temperature, request.top_p, self._rng)
            sequence.output_tokens.append(token)
            sequence.last_token_at = now
            self._emit(request, token)

            if self._should_stop(request):
                self._finish(request, *self._stop_reason(request))

        for request in [r for r in self._active if r.finished]:
            if request.sequence is not None:
                self.backend.release(request.sequence)

        self._active = [r for r in self._active if not r.finished]
        self.metrics.active_sequences.set(len(self._active))

        if self._active:
            average_batch = sum(self.batch_size_history[-20:]) / min(
                len(self.batch_size_history), 20
            )
            self.metrics.tokens_per_second.set(self.throughput())
            _ = average_batch

        self.metrics.kv_padding_waste.set(float(getattr(self.backend, "padding_waste", 0.0)))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def is_idle(self) -> bool:
        """True when nothing is queued and nothing is decoding."""
        return not self._active and not self._waiting

    async def _run(self, fn: Callable, *args: Any) -> Any:
        if self.use_thread:
            return await asyncio.to_thread(fn, *args)
        return fn(*args)

    def _emit(self, request: GenerationRequest, token: int, first: bool = False) -> None:
        text = self.tokenizer.decode([token])
        request.output_text += text
        self.metrics.generation_tokens_total.inc()
        now = time.time()
        self._token_events.append((now, 1))

        if first and request.first_token_at is not None:
            self.metrics.ttft.observe(request.first_token_at - request.submitted_at)
        elif request.last_token_at is not None:
            # Per-request inter-token latency: how long *this* request waited for
            # its next token, which is what a user actually experiences.
            self.metrics.tpot.observe(max(now - request.last_token_at, 0.0))
        request.last_token_at = now

        if request.stream:
            request._publish(
                {
                    "id": request.id,
                    "token": text,
                    "index": request.output_tokens - 1,
                }
            )

    def _should_stop(self, request: GenerationRequest) -> bool:
        sequence = request.sequence
        if sequence is None:
            return True
        if request._cancel:
            return True
        if len(sequence.output_tokens) >= request.max_tokens:
            return True
        if sequence.output_tokens and sequence.output_tokens[-1] == EOS:
            return True
        if request.stop and any(
            request.output_text.endswith(marker) for marker in request.stop if marker
        ):
            return True
        return False

    def _stop_reason(self, request: GenerationRequest) -> tuple[FinishReason, str | None]:
        sequence = request.sequence
        if request._cancel:
            return FinishReason.CANCELLED, "client cancelled"
        if sequence and sequence.output_tokens and sequence.output_tokens[-1] == EOS:
            return FinishReason.STOP, None
        if request.stop and any(
            request.output_text.endswith(marker) for marker in request.stop if marker
        ):
            return FinishReason.STOP, "stop sequence matched"
        return FinishReason.LENGTH, None

    def _finish(
        self, request: GenerationRequest, reason: FinishReason, error: str | None = None
    ) -> None:
        request.finish_reason = reason
        request.error = error
        request.finished_at = time.time()
        self.metrics.e2e.observe(request.latency)
        status = {
            FinishReason.STOP: "succeeded",
            FinishReason.LENGTH: "succeeded",
            FinishReason.CANCELLED: "cancelled",
            FinishReason.ERROR: "failed",
        }[reason]
        self.metrics.requests_total.inc(status=status)
        if request.sequence is not None:
            request.sequence.finished = True
            request.sequence.finish_reason = reason.value
        if request.stream:
            request._publish(
                {
                    "id": request.id,
                    "token": "",
                    "index": request.output_tokens,
                    "finish_reason": reason.value,
                    "done": True,
                }
            )
            request._publish(None)

    def throughput(self, window: float | None = None) -> float:
        """Generation tokens/second over a sliding window."""
        window = window or self.throughput_window
        cutoff = time.time() - window
        while self._token_events and self._token_events[0][0] < cutoff:
            self._token_events.popleft()
        if not self._token_events:
            return 0.0
        span = max(time.time() - self._token_events[0][0], 1e-6)
        return len(self._token_events) / span

    def stats(self) -> dict:
        history = self.batch_size_history[-200:]
        return {
            "backend": self.backend.name,
            "max_batch_size": self.max_batch_size,
            "max_batch_tokens": self.max_batch_tokens,
            "waiting": len(self._waiting),
            "active": len(self._active),
            "steps": self._steps,
            "mean_batch_size": round(sum(history) / len(history), 3) if history else 0.0,
            "max_batch_size_seen": max(history) if history else 0,
            "generation_tokens_per_second": round(self.throughput(), 2),
            "backend_info": self.backend.info(),
            "metrics": self.metrics.summary(),
        }
