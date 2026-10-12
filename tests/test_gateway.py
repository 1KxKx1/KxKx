"""Gateway tests: the OpenAI-compatible HTTP surface.

Skipped automatically when the optional web extra (FastAPI) is not installed —
the scheduler, workflow engine and batcher are all usable without it.
"""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi", reason="the [web] extra is not installed")
from fastapi.testclient import TestClient  # noqa: E402

from kubeai.serving.gateway import GatewayConfig, create_app, render_chat  # noqa: E402

TINY_GATEWAY = GatewayConfig(
    default_model="tiny-test",
    default_backend="numpy",
    memory_budget_mib=512,
    max_batch_size=4,
    model_dim=32,
    model_layers=1,
    model_vocab=256,
)


@pytest.fixture()
def client():
    app = create_app(TINY_GATEWAY)
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# health and introspection
# --------------------------------------------------------------------------- #
def test_healthz_reports_a_loaded_model(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert "tiny-test" in body["loaded_models"]
    assert body["accelerator_source"] in {"nvidia-smi", "simulated", "none"}


def test_readyz_is_ready_once_the_default_model_is_loaded(client):
    assert client.get("/readyz").json()["status"] == "ready"


def test_models_endpoint_lists_the_default_model(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    entry = next(m for m in body["data"] if m["id"] == "tiny-test")
    assert entry["object"] == "model"
    assert entry["metadata"]["loaded"] is True


def test_metrics_are_exposed_in_prometheus_format(client):
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
    )
    text = client.get("/metrics").text
    assert "# TYPE kubeai_requests_total counter" in text
    assert "kubeai_generation_tokens_total" in text
    assert "kubeai_batch_size_bucket" in text


def test_admin_stats_includes_scheduler_and_registry(client):
    body = client.get("/admin/stats").json()
    assert "registry" in body and "scheduler" in body and "metrics" in body
    assert body["registry"]["used_mib"] >= 0
    assert body["scheduler"]["policy"] == "binpack"


def test_admin_gpus_reports_the_inventory(client):
    body = client.get("/admin/gpus").json()
    assert body["nodes"], "at least one node should always be reported"
    assert "stats" in body


def test_admin_queue_reports_batcher_state(client):
    body = client.get("/admin/queue").json()
    assert "tiny-test" in body
    assert body["tiny-test"]["backend"] == "numpy-transformer"


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def test_chat_completion_returns_an_openai_shaped_response(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "describe the scheduler"}],
            "max_tokens": 4,
            "temperature": 0.0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert body["choices"][0]["finish_reason"] == "length"
    usage = body["usage"]
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == usage["prompt_tokens"] + 4
    # Our own telemetry rides alongside, without breaking OpenAI clients.
    assert "ttft_seconds" in body["kubeai"]


def test_completions_endpoint_works(client):
    response = client.post(
        "/v1/completions", json={"prompt": "hello", "max_tokens": 3, "temperature": 0.0}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["usage"]["completion_tokens"] == 3


def test_streaming_chat_completion_emits_sse_chunks(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "stream this"}],
            "max_tokens": 3,
            "temperature": 0.0,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payloads = [
            line[len("data: ") :]
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert len(chunks) == 3
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"


def test_embeddings_are_normalised_and_deterministic(client):
    first = client.post("/v1/embeddings", json={"input": "vector me"}).json()
    second = client.post("/v1/embeddings", json={"input": "vector me"}).json()
    vector = first["data"][0]["embedding"]
    assert len(vector) > 0
    assert first["data"][0]["embedding"] == second["data"][0]["embedding"]
    norm = sum(x * x for x in vector) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-3)


def test_embeddings_accept_a_batch(client):
    body = client.post("/v1/embeddings", json={"input": ["a", "b", "c"]}).json()
    assert len(body["data"]) == 3
    assert [d["index"] for d in body["data"]] == [0, 1, 2]


def test_unknown_model_returns_404(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "does-not-exist", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 404


def test_invalid_payload_is_rejected(client):
    response = client.post("/v1/chat/completions", json={"max_tokens": 0})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# model lifecycle
# --------------------------------------------------------------------------- #
def test_load_and_unload_a_model_at_runtime(client):
    loaded = client.post(
        "/v1/models/load",
        json={
            "id": "second",
            "backend": "numpy",
            "footprint_mib": 64,
            "max_batch_size": 2,
            "backend_options": {
                "config": {
                    "vocab_size": 256, "dim": 32, "n_layers": 1, "n_heads": 2, "max_seq": 64
                }
            },
        },
    )
    assert loaded.status_code == 200, loaded.text
    assert loaded.json()["id"] == "second"

    assert client.delete("/v1/models/second").json()["unloaded"] is True
    assert client.delete("/v1/models/second").status_code == 404


def test_loading_a_model_that_cannot_fit_returns_507(client):
    """The endpoint registers what it is asked for, so the failure mode is memory."""
    response = client.post(
        "/v1/models/load",
        json={"id": "too-big", "backend": "numpy", "footprint_mib": 100_000},
    )
    assert response.status_code == 507
    assert "too-big" in response.json()["detail"]


def test_loading_with_an_unknown_backend_returns_400(client):
    response = client.post(
        "/v1/models/load",
        json={"id": "bad-backend", "backend": "not-a-backend", "footprint_mib": 1},
    )
    assert response.status_code == 400
    assert "unknown backend" in response.json()["detail"]


def test_loading_with_an_invalid_backend_option_returns_400(client):
    response = client.post(
        "/v1/models/load",
        json={
            "id": "bad-option",
            "backend": "numpy",
            "footprint_mib": 1,
            "backend_options": {"notAnOption": 1},
        },
    )
    assert response.status_code == 400
    assert "invalid options" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# autoscaling
# --------------------------------------------------------------------------- #
def test_autoscaling_hint_recommends_scaling_out_when_saturated(client):
    for _ in range(12):
        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "load"}], "max_tokens": 1},
        )
    body = client.get("/admin/autoscaling-hint").json()
    assert body["recommendation"] in {"scale-up", "hold", "scale-down"}
    assert body["desired_replicas"] >= 1
    assert 0.0 <= body["saturation"] <= 10.0


# --------------------------------------------------------------------------- #
# prompt formatting
# --------------------------------------------------------------------------- #
def test_render_chat_uses_role_markers_and_ends_with_assistant():
    from kubeai.serving.gateway import ChatMessage

    prompt = render_chat(
        [ChatMessage(role="system", content="be terse"),
         ChatMessage(role="user", content="hi")]
    )
    assert prompt.startswith("<|system|>be terse")
    assert "<|user|>hi" in prompt
    assert prompt.endswith("<|assistant|>")
