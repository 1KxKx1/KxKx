"""Serving tests: tokenizer, transformer backend, continuous batching, registry, metrics."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from kubeai.serving.backends import (
    BOS,
    UNK,
    CharTokenizer,
    EchoBackend,
    NumpyTransformerBackend,
    Sequence,
    TransformerConfig,
    build_backend,
)
from kubeai.serving.batcher import (
    ContinuousBatcher,
    FinishReason,
    GenerationRequest,
    QueueFull,
    sample_token,
)
from kubeai.serving.metrics import MetricsRegistry, ServingMetrics
from kubeai.serving.registry import (
    InsufficientMemory,
    ModelNotFound,
    ModelRegistry,
    ModelSpec,
)

TINY = TransformerConfig(vocab_size=256, dim=32, n_layers=1, n_heads=2, max_seq=128)


@pytest.fixture()
def backend() -> NumpyTransformerBackend:
    return NumpyTransformerBackend(TINY)


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #
def test_tokenizer_round_trips_ascii():
    tokenizer = CharTokenizer()
    text = "gpu scheduling 101"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenizer_round_trips_cjk():
    tokenizer = CharTokenizer()
    text = "云原生推理平台"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenizer_emits_bos_and_marks_unknown_characters():
    tokenizer = CharTokenizer(vocab_size=64)
    ids = tokenizer.encode("a")
    assert ids[0] == BOS
    assert tokenizer._encode_char("\u4e00") == UNK  # CJK outside a tiny vocabulary


def test_tokenizer_never_exceeds_the_vocabulary():
    tokenizer = CharTokenizer(vocab_size=64)
    for char in "abcXYZ019 云原生":
        assert tokenizer._encode_char(char) < 64


# --------------------------------------------------------------------------- #
# transformer backend
# --------------------------------------------------------------------------- #
def test_prefill_returns_logits_over_the_vocabulary(backend):
    sequence = Sequence(id="s1", prompt_tokens=backend.tokenizer.encode("hello world"))
    logits = backend.prefill(sequence)
    assert logits.shape == (TINY.vocab_size,)
    assert sequence.prefilled
    assert len(sequence.cache) == TINY.n_layers


def test_decode_advances_the_cache_by_one_token(backend):
    sequence = Sequence(id="s1", prompt_tokens=backend.tokenizer.encode("hello"))
    backend.prefill(sequence)
    before = sequence.cache[0][0].shape[1]
    sequence.output_tokens.append(7)
    logits = backend.decode([sequence])
    after = sequence.cache[0][0].shape[1]
    assert after == before + 1
    assert logits[0].shape == (TINY.vocab_size,)


def test_decode_handles_sequences_with_different_history_lengths(backend):
    lengths = [3, 9, 17]
    sequences = []
    for index, length in enumerate(lengths):
        tokens = (np.arange(length) % (TINY.vocab_size - 4) + 4).tolist()
        sequence = Sequence(id=f"s{index}", prompt_tokens=tokens)
        backend.prefill(sequence)
        sequence.output_tokens.append(5)
        sequences.append(sequence)

    logits = backend.decode(sequences)
    assert len(logits) == len(lengths)
    # Each sequence's cache must contain exactly its own history plus the new token.
    for sequence, length in zip(sequences, lengths, strict=True):
        assert sequence.cache[0][0].shape[1] == length + 1


def test_batched_decode_matches_unbatched_results(backend):
    """Batching must not change the maths — only how it is scheduled."""
    prompts = [[4, 5, 6], [7, 8, 9]]

    def run(sequences_in_one_batch: bool) -> list[np.ndarray]:
        model = NumpyTransformerBackend(TINY)
        sequences = []
        for index, tokens in enumerate(prompts):
            sequence = Sequence(id=f"s{index}", prompt_tokens=tokens, output_tokens=[11])
            model.prefill(sequence)
            sequences.append(sequence)
        if sequences_in_one_batch:
            return model.decode(sequences)
        return [model.decode([s])[0] for s in sequences]

    batched = run(True)
    unbatched = run(False)
    for a, b in zip(batched, unbatched, strict=True):
        # Padding is masked out, so results agree to float tolerance.
        assert np.allclose(a, b, atol=1e-3)


def test_backend_reports_its_footprint_and_padding_waste(backend):
    sequences = []
    for length in (2, 40):
        tokens = (np.arange(length) % (TINY.vocab_size - 4) + 4).tolist()
        sequence = Sequence(id=str(length), prompt_tokens=tokens)
        backend.prefill(sequence)
        sequence.output_tokens.append(5)
        sequences.append(sequence)
    backend.decode(sequences)

    assert backend.footprint_mib >= 0
    assert 0.0 <= backend.padding_waste < 1.0
    assert backend.info()["parameters"] > 0


def test_backend_is_deterministic_for_a_fixed_seed():
    first = NumpyTransformerBackend(TINY).prefill(
        Sequence(id="a", prompt_tokens=[4, 5, 6, 7])
    )
    second = NumpyTransformerBackend(TINY).prefill(
        Sequence(id="a", prompt_tokens=[4, 5, 6, 7])
    )
    assert np.allclose(first, second)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        build_backend("nope")


def test_backend_accepts_a_mapping_config():
    """POST /v1/models/load sends JSON, so a dict must be accepted for `config`."""
    backend = build_backend(
        "numpy", config={"vocab_size": 64, "dim": 16, "n_layers": 1, "n_heads": 2}
    )
    assert isinstance(backend, NumpyTransformerBackend)
    assert backend.config.vocab_size == 64
    assert backend.config.dim == 16
    # The tokenizer must follow the model's vocabulary, not the default one.
    assert backend.tokenizer.vocab_size == 64


def test_backend_options_accept_camel_case():
    """The documented HTTP payload uses camelCase; it must not be a TypeError."""
    backend = build_backend("vllm", baseUrl="http://vllm:8000/v1", model="qwen")
    assert backend.base_url == "http://vllm:8000/v1"
    assert backend.model == "qwen"


def test_unknown_backend_option_raises_a_clear_error():
    with pytest.raises(ValueError, match="invalid options for backend 'echo'"):
        build_backend("echo", nonExistentOption=1)


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def test_greedy_sampling_picks_the_argmax():
    logits = np.array([0.1, 5.0, 0.2], dtype=np.float32)
    assert sample_token(logits, temperature=0.0, top_p=1.0, rng=np.random.default_rng(0)) == 1


def test_top_p_restricts_sampling_to_the_nucleus():
    logits = np.array([100.0, 0.0, 0.0], dtype=np.float32)
    rng = np.random.default_rng(0)
    picks = {sample_token(logits, temperature=1.0, top_p=0.5, rng=rng) for _ in range(20)}
    assert picks == {0}


# --------------------------------------------------------------------------- #
# continuous batching
# --------------------------------------------------------------------------- #
async def test_all_requests_complete_and_are_batched_together(backend):
    metrics = ServingMetrics().install()
    batcher = ContinuousBatcher(backend, max_batch_size=8, metrics=metrics)
    await batcher.start()
    try:
        requests = [
            GenerationRequest(prompt=f"prompt number {i}", max_tokens=6, temperature=0.0)
            for i in range(16)
        ]
        for request in requests:
            batcher.submit_nowait(request)
        while not all(r.finished for r in requests):
            await asyncio.sleep(0.001)
    finally:
        await batcher.stop()

    assert all(r.finish_reason is FinishReason.LENGTH for r in requests)
    assert all(r.output_tokens == 6 for r in requests)
    # The whole point: more than one sequence decoded per step.
    assert max(batcher.batch_size_history) > 1
    assert metrics.requests_total.value(status="succeeded") == 16
    assert metrics.generation_tokens_total.value() == 96


async def test_finished_sequences_free_their_slot_for_queued_work(backend):
    """A freed slot is reused immediately — the queued request does not wait for the batch.

    Under *static* batching the queued request would block until every member of
    the group finished, i.e. forever here, because ``stuck`` never completes.
    """
    batcher = ContinuousBatcher(backend, max_batch_size=2, max_prefills_per_step=2)
    await batcher.start()
    try:
        # One request that ends quickly, one that deliberately never ends.
        short_lived = GenerationRequest(prompt="finishes fast", max_tokens=4, temperature=0.0)
        stuck = GenerationRequest(
            prompt="never finishes " * 3, max_tokens=1_000_000, temperature=0.0
        )
        batcher.submit_nowait(short_lived)
        batcher.submit_nowait(stuck)

        deadline = time.time() + 20
        while not short_lived.finished and time.time() < deadline:
            await asyncio.sleep(0.002)
        assert short_lived.finished, "the first request never completed"

        queued = GenerationRequest(prompt="queued", max_tokens=2, temperature=0.0)
        batcher.submit_nowait(queued)
        deadline = time.time() + 20
        while not queued.finished and time.time() < deadline:
            await asyncio.sleep(0.002)

        assert queued.finished, "the queued request never got the freed slot"
        assert queued.output_tokens == 2
        # Proof it did not wait for the batch: the other member is still decoding.
        assert not stuck.finished
    finally:
        await batcher.stop()


async def test_streaming_publishes_tokens_then_a_terminator(backend):
    batcher = ContinuousBatcher(backend, max_batch_size=1)
    await batcher.start()
    try:
        request = GenerationRequest(prompt="stream me", max_tokens=4, temperature=0.0, stream=True)
        queue = request.subscribe()
        batcher.submit_nowait(request)

        events = []
        while True:
            event = await asyncio.wait_for(queue.get(), timeout=10)
            if event is None:
                break
            events.append(event)

        tokens = [e for e in events if not e.get("done")]
        assert len(tokens) == 4
        assert [e["index"] for e in tokens] == [0, 1, 2, 3]
        # The last non-sentinel event is the terminator; the None sentinel that
        # follows it is what ends the iteration and is intentionally not recorded.
        assert events[-1]["done"] is True
        assert events[-1]["finish_reason"] == "length"
    finally:
        await batcher.stop()


async def test_cancellation_stops_generation(backend):
    batcher = ContinuousBatcher(backend, max_batch_size=1)
    await batcher.start()
    try:
        request = GenerationRequest(prompt="cancel me", max_tokens=1000, temperature=0.0)
        batcher.submit_nowait(request)
        await asyncio.sleep(0.02)
        request.cancel()
        deadline = time.time() + 5
        while not request.finished and time.time() < deadline:
            await asyncio.sleep(0.002)
        assert request.finish_reason is FinishReason.CANCELLED
        assert request.output_tokens < 1000
    finally:
        await batcher.stop()


async def test_queue_rejects_beyond_capacity(backend):
    batcher = ContinuousBatcher(backend, max_queue_size=1)
    batcher.submit_nowait(GenerationRequest(prompt="a", max_tokens=1))
    with pytest.raises(QueueFull):
        batcher.submit_nowait(GenerationRequest(prompt="b", max_tokens=1))


async def test_generate_helper_returns_finished_request(backend):
    async with ContinuousBatcher(backend, max_batch_size=2) as batcher:
        request = await asyncio.wait_for(
            batcher.generate("hello", max_tokens=3, temperature=0.0), timeout=15
        )
        assert request.finished
        assert request.output_tokens == 3


async def test_batcher_stats_expose_utilisation(backend):
    async with ContinuousBatcher(backend, max_batch_size=4) as batcher:
        await asyncio.wait_for(batcher.generate("stats", max_tokens=2, temperature=0.0), timeout=15)
        stats = batcher.stats()
        assert stats["steps"] > 0
        assert stats["max_batch_size"] == 4
        assert stats["backend"] == "numpy-transformer"
        assert "generation_tokens_per_second" in stats


def test_batch_size_must_be_positive(backend):
    with pytest.raises(ValueError, match="max_batch_size"):
        ContinuousBatcher(backend, max_batch_size=0)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
async def test_registry_loads_unloads_and_reports(backend):
    registry = ModelRegistry(memory_budget_mib=1024)
    registry.register(ModelSpec(id="tiny", backend="numpy", footprint_mib=64,
                                backend_options={"config": TINY}))
    model = await registry.load("tiny")
    assert model.spec.id == "tiny"
    assert registry.used_mib == 64
    assert registry.available_mib == 960
    assert await registry.unload("tiny") is True
    assert registry.used_mib == 0
    await registry.shutdown()


async def test_registry_evicts_lru_to_make_room(backend):
    registry = ModelRegistry(memory_budget_mib=200)
    for name in ("a", "b", "c"):
        registry.register(
            ModelSpec(id=name, backend="numpy", footprint_mib=100,
                      backend_options={"config": TINY})
        )
    await registry.load("a")
    await asyncio.sleep(0.01)
    await registry.load("b")
    assert set(registry.loaded) == {"a", "b"}

    await registry.load("c")
    assert set(registry.loaded) == {"b", "c"}, "least recently used model should be evicted"
    assert registry.evictions == 1
    await registry.shutdown()


async def test_registry_refuses_when_the_budget_cannot_fit(backend):
    registry = ModelRegistry(memory_budget_mib=100)
    registry.register(
        ModelSpec(id="huge", backend="numpy", footprint_mib=5000,
                  backend_options={"config": TINY})
    )
    with pytest.raises(InsufficientMemory):
        await registry.load("huge")
    await registry.shutdown()


async def test_registry_does_not_evict_pinned_models(backend):
    registry = ModelRegistry(memory_budget_mib=200)
    registry.register(
        ModelSpec(id="pinned", backend="numpy", footprint_mib=100, pinned=True,
                  backend_options={"config": TINY})
    )
    # 150 does not fit in the 100MiB left over, and the only thing holding memory
    # is pinned, so the load must fail rather than evict it.
    registry.register(
        ModelSpec(id="other", backend="numpy", footprint_mib=150,
                  backend_options={"config": TINY})
    )
    await registry.load("pinned")
    with pytest.raises(InsufficientMemory):
        await registry.load("other")
    assert "pinned" in registry.loaded
    await registry.shutdown()


async def test_registry_rejects_unknown_model():
    registry = ModelRegistry(memory_budget_mib=100)
    with pytest.raises(ModelNotFound):
        await registry.load("ghost")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_counter_gauge_and_histogram_render_prometheus_text():
    registry = MetricsRegistry("test")
    counter = registry.counter("requests_total", "Requests.", ("status",))
    gauge = registry.gauge("queue_depth", "Queued requests.")
    histogram = registry.histogram("latency_seconds", "Latency.", buckets=(0.1, 1.0))

    counter.inc(status="ok")
    counter.inc(2, status="ok")
    gauge.set(5)
    histogram.observe(0.05)
    histogram.observe(2.0)

    text = registry.render()
    assert "# TYPE test_requests_total counter" in text
    assert 'test_requests_total{status="ok"} 3.0' in text
    assert "# TYPE test_queue_depth gauge" in text
    assert "test_queue_depth 5.0" in text
    assert "# TYPE test_latency_seconds histogram" in text
    assert 'test_latency_seconds_bucket{le="0.1"} 1' in text
    assert 'test_latency_seconds_bucket{le="+Inf"} 2' in text
    assert "test_latency_seconds_count 2" in text


def test_metrics_registry_is_idempotent():
    registry = MetricsRegistry("test")
    first = registry.counter("x_total", "X.")
    second = registry.counter("x_total", "X.")
    first.inc()
    assert second.value() == 1.0


async def test_serving_metrics_capture_latency(backend):
    metrics = ServingMetrics().install()
    async with ContinuousBatcher(backend, max_batch_size=2, metrics=metrics) as batcher:
        await asyncio.wait_for(batcher.generate("m", max_tokens=3, temperature=0.0), timeout=15)
    summary = metrics.summary()
    assert summary["requests"]["succeeded"] == 1
    assert summary["ttft_seconds"]["count"] == 1
    assert summary["generation_tokens"] == 3


def test_echo_backend_smoke():
    echo = EchoBackend()
    sequence = Sequence(id="e", prompt_tokens=[1, 2, 3])
    logits = echo.prefill(sequence)
    assert logits.shape == (EchoBackend.VOCAB,)
    assert len(echo.decode([sequence])) == 1
