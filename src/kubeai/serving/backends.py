"""Inference backends.

A backend owns the weights and the per-sequence KV cache; the batcher owns
scheduling. The interface is deliberately the same shape as vLLM's internals —
``prefill`` once, then one ``decode`` step for the whole running batch — because
that is the split that makes *continuous* batching possible: a sequence that
finishes mid-flight frees its slot, and a queued sequence takes it on the very
next step instead of waiting for the batch to drain.

``NumpyTransformerBackend`` is a real (small, randomly initialised) decoder-only
transformer. The weights are random on purpose: it makes the repository
self-contained, keeps the numbers reproducible, and — since we are measuring
*throughput*, not quality — costs nothing. Swap in ``VLLMBackend`` for real
weights.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
PAD, BOS, EOS, UNK = 0, 1, 2, 3
_ASCII_START = 4
_ASCII_CODEPOINT_START = 32
_ASCII_COUNT = 2048                      # 32 .. 2079 — Latin, Greek, Cyrillic, ...
_CJK_START = _ASCII_START + _ASCII_COUNT
_CJK_CODEPOINT_START = 0x4E00            # CJK Unified Ideographs
_CJK_COUNT = 0x9FFF - 0x4E00 + 1         # the whole block, 20992 code points
DEFAULT_VOCAB_SIZE = _CJK_START + _CJK_COUNT


class CharTokenizer:
    """A tiny, dependency-free, reversible character tokenizer.

    Covers printable Latin/Greek/Cyrillic plus the complete CJK Unified Ideographs
    block, which keeps prompts and generated text readable in demos. It is
    intentionally *not* a sub-word tokenizer: a production deployment swaps it for
    the model's own tokenizer (see :class:`VLLMBackend`).
    """

    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        ids = [BOS]
        for ch in text:
            ids.append(self._encode_char(ch))
        return ids

    def _encode_char(self, ch: str) -> int:
        cp = ord(ch)
        if _ASCII_CODEPOINT_START <= cp < _ASCII_CODEPOINT_START + _ASCII_COUNT:
            token = _ASCII_START + (cp - _ASCII_CODEPOINT_START)
        elif _CJK_CODEPOINT_START <= cp < _CJK_CODEPOINT_START + _CJK_COUNT:
            token = _CJK_START + (cp - _CJK_CODEPOINT_START)
        else:
            return UNK
        # A small model must not be handed token ids beyond its embedding table.
        return token if token < self.vocab_size else UNK

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for token in ids:
            if token in (PAD, BOS, EOS, UNK):
                continue
            if _ASCII_START <= token < _CJK_START:
                out.append(chr(token - _ASCII_START + _ASCII_CODEPOINT_START))
            elif _CJK_START <= token < _CJK_START + _CJK_COUNT:
                out.append(chr(token - _CJK_START + _CJK_CODEPOINT_START))
        return "".join(out)


# --------------------------------------------------------------------------- #
# Sequence
# --------------------------------------------------------------------------- #
@dataclass
class Sequence:
    """One in-flight generation request, as the backend sees it."""

    id: str
    prompt_tokens: list[int]
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)

    output_tokens: list[int] = field(default_factory=list)
    #: Per-layer ``(k, v)`` arrays of shape ``(heads, T, head_dim)``.
    cache: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    prefilled: bool = False
    finished: bool = False
    finish_reason: str | None = None

    first_token_at: float | None = None
    last_token_at: float | None = None

    @property
    def context_length(self) -> int:
        return len(self.prompt_tokens) + len(self.output_tokens)

    def text(self, tokenizer: CharTokenizer) -> str:
        return tokenizer.decode(self.output_tokens)


# --------------------------------------------------------------------------- #
# Backend interface
# --------------------------------------------------------------------------- #
class Backend(ABC):
    """Weights + KV cache management."""

    name: str = "backend"

    @abstractmethod
    def prefill(self, sequence: Sequence) -> np.ndarray:
        """Run the prompt through the model; return next-token logits ``(V,)``."""

    @abstractmethod
    def decode(self, sequences: list[Sequence]) -> list[np.ndarray]:
        """Advance every sequence by one token; return logits ``(V,)`` per sequence."""

    def release(self, sequence: Sequence) -> None:  # noqa: B027 — optional hook
        """Free the sequence's cache resources."""

    @property
    def footprint_mib(self) -> int:
        return 0

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "footprint_mib": self.footprint_mib}


# --------------------------------------------------------------------------- #
# Numpy transformer
# --------------------------------------------------------------------------- #
@dataclass
class TransformerConfig:
    vocab_size: int = DEFAULT_VOCAB_SIZE
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: int = 4
    max_seq: int = 1024
    seed: int = 1234
    dtype: str = "float32"

    @property
    def head_dim(self) -> int:
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        return self.dim // self.n_heads


def _layer_norm(x: np.ndarray, gain: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gain + bias


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def _project(
    hidden: np.ndarray, matrix: np.ndarray, batch: int, tokens: int, heads: int, head_dim: int
) -> np.ndarray:
    """``(B, T, D) @ (D, D)`` -> ``(B, heads, T, head_dim)``.

    A module-level function rather than a closure over the loop body: a closure
    that reads the loop variable is correct only as long as it is called in the
    same iteration, which is exactly the kind of assumption that breaks later.
    """
    out = hidden @ matrix
    return out.reshape(batch, tokens, heads, head_dim).transpose(0, 2, 1, 3)


class NumpyTransformerBackend(Backend):
    """Decoder-only transformer with per-sequence KV caches, in NumPy.

    The decode path pads every sequence's cache to the longest cache in the batch
    before the batched attention. That padding is real, measurable wasted work —
    and it is exactly the inefficiency that PagedAttention (vLLM) was invented to
    remove. ``padding_waste`` reports it, which is why the metric exists.
    """

    name = "numpy-transformer"

    def __init__(self, config: TransformerConfig | None = None) -> None:
        self.config = config or TransformerConfig()
        self.tokenizer = CharTokenizer(self.config.vocab_size)
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        dt = np.dtype(cfg.dtype)

        def randn(*shape: int, scale: float = 0.02) -> np.ndarray:
            return (rng.standard_normal(shape) * scale).astype(dt)

        self.token_embedding = randn(cfg.vocab_size, cfg.dim)
        self.position_embedding = randn(cfg.max_seq, cfg.dim)
        self.layers = [
            {
                "ln1_g": np.ones(cfg.dim, dt), "ln1_b": np.zeros(cfg.dim, dt),
                "wq": randn(cfg.dim, cfg.dim), "wk": randn(cfg.dim, cfg.dim),
                "wv": randn(cfg.dim, cfg.dim), "wo": randn(cfg.dim, cfg.dim),
                "ln2_g": np.ones(cfg.dim, dt), "ln2_b": np.zeros(cfg.dim, dt),
                "w1": randn(cfg.dim, cfg.dim * cfg.mlp_ratio),
                "b1": np.zeros(cfg.dim * cfg.mlp_ratio, dt),
                "w2": randn(cfg.dim * cfg.mlp_ratio, cfg.dim),
                "b2": np.zeros(cfg.dim, dt),
            }
            for _ in range(cfg.n_layers)
        ]
        self.final_g = np.ones(cfg.dim, dt)
        self.final_b = np.zeros(cfg.dim, dt)
        self.lm_head = randn(cfg.dim, cfg.vocab_size)

        # telemetry
        self.steps = 0
        self.tokens_processed = 0
        self.padded_slots = 0
        self.used_slots = 0

    # -- weights ---------------------------------------------------------- #
    @property
    def parameter_count(self) -> int:
        total = self.token_embedding.size + self.position_embedding.size + self.lm_head.size
        total += sum(v.size for layer in self.layers for v in layer.values())
        return int(total)

    @property
    def footprint_mib(self) -> int:
        bytes_per = np.dtype(self.config.dtype).itemsize
        return int(self.parameter_count * bytes_per / (1024 * 1024))

    def info(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "backend": self.name,
            "vocab_size": cfg.vocab_size,
            "dim": cfg.dim,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "parameters": self.parameter_count,
            "footprint_mib": self.footprint_mib,
            "steps": self.steps,
            "tokens_processed": self.tokens_processed,
            "padding_waste": round(self.padding_waste, 4),
        }

    @property
    def padding_waste(self) -> float:
        if self.padded_slots == 0:
            return 0.0
        return max(1.0 - (self.used_slots / self.padded_slots), 0.0)

    # -- forward ---------------------------------------------------------- #
    def _forward(
        self,
        token_ids: np.ndarray,          # (B, T)
        past: list[tuple[np.ndarray, np.ndarray, int]],
        # per-sequence (k, v, valid_len); k/v are (layers, heads, Tp, head_dim)
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], int]:
        """Returns ``(logits, k_per_layer, v_per_layer, pad_len)``.

        ``k_per_layer[l]`` / ``v_per_layer[l]`` have shape ``(B, heads, pad_len+T, head_dim)``.
        The first ``pad_len`` slots are padded: callers must strip them per sequence
        using each sequence's own length (see :meth:`decode`).
        """
        cfg = self.config
        batch, new_tokens = token_ids.shape
        pad_len = max((p[2] for p in past), default=0)
        heads, head_dim = cfg.n_heads, cfg.head_dim
        dt = np.dtype(cfg.dtype)

        # --- embed ------------------------------------------------------- #
        positions = np.stack(
            [np.arange(past[i][2], past[i][2] + new_tokens) for i in range(batch)]
        )  # (B, T)
        positions = np.clip(positions, 0, cfg.max_seq - 1)
        x = self.token_embedding[token_ids] + self.position_embedding[positions]  # (B,T,D)

        # --- padded past cache ------------------------------------------- #
        k_past = [np.zeros((batch, heads, pad_len, head_dim), dt) for _ in range(cfg.n_layers)]
        v_past = [np.zeros((batch, heads, pad_len, head_dim), dt) for _ in range(cfg.n_layers)]
        valid = np.zeros((batch, 1, 1, pad_len), dtype=bool)
        for i in range(batch):
            length = past[i][2]
            valid[i, 0, 0, :length] = True
            for layer in range(cfg.n_layers):
                if length:
                    k_past[layer][i, :, :length, :] = past[i][0][layer]
                    v_past[layer][i, :, :length, :] = past[i][1][layer]

        self.padded_slots += batch * pad_len
        self.used_slots += sum(p[2] for p in past)

        new_k: list[np.ndarray] = []
        new_v: list[np.ndarray] = []

        # Mask for the *new* tokens: causal, plus the valid past. Both sides are
        # broadcast to (B, 1, T, .) so a prefill (pad_len == 0) and a decode step
        # build the same shaped mask.
        causal = np.tril(np.ones((new_tokens, new_tokens), dtype=bool))  # (T,T)
        past_mask = np.broadcast_to(valid, (batch, 1, new_tokens, pad_len))
        new_mask = np.broadcast_to(causal[None, None, :, :], (batch, 1, new_tokens, new_tokens))
        attn_mask = np.concatenate([past_mask, new_mask], axis=-1)  # (B,1,T,Tp+T)

        for layer_index, layer in enumerate(self.layers):
            residual = x
            h = _layer_norm(x, layer["ln1_g"], layer["ln1_b"])

            q = _project(h, layer["wq"], batch, new_tokens, heads, head_dim)
            k_new = _project(h, layer["wk"], batch, new_tokens, heads, head_dim)
            v_new = _project(h, layer["wv"], batch, new_tokens, heads, head_dim)

            k_all = np.concatenate([k_past[layer_index], k_new], axis=2)
            v_all = np.concatenate([v_past[layer_index], v_new], axis=2)

            scores = (q @ k_all.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
            scores = np.where(attn_mask, scores, np.float32(-1e30))
            context = _softmax(scores, axis=-1) @ v_all  # (B,heads,T,head_dim)
            context = context.transpose(0, 2, 1, 3).reshape(batch, new_tokens, cfg.dim)

            x = residual + context @ layer["wo"]

            residual = x
            h2 = _layer_norm(x, layer["ln2_g"], layer["ln2_b"])
            hidden = np.maximum(h2 @ layer["w1"] + layer["b1"], 0)
            x = residual + hidden @ layer["w2"] + layer["b2"]

            new_k.append(k_all)
            new_v.append(v_all)

        x = _layer_norm(x, self.final_g, self.final_b)
        logits = x @ self.lm_head  # (B,T,V)
        return logits, new_k, new_v, pad_len

    # -- Backend API ------------------------------------------------------ #
    def prefill(self, sequence: Sequence) -> np.ndarray:
        tokens = np.asarray([sequence.prompt_tokens], dtype=np.int64)  # (1,L)
        if tokens.shape[1] > self.config.max_seq:
            tokens = tokens[:, -self.config.max_seq :]
        logits, k_layers, v_layers, _pad = self._forward(tokens, [self._empty_past()])
        # pad_len is 0 for a prefill, so the cache is exactly this sequence's.
        sequence.cache = [
            (k_layers[layer][0], v_layers[layer][0]) for layer in range(self.config.n_layers)
        ]
        sequence.prefilled = True
        self.steps += 1
        self.tokens_processed += tokens.shape[1]
        return logits[0, -1, :]

    def _empty_past(self) -> tuple[np.ndarray, np.ndarray, int]:
        cfg = self.config
        empty = np.zeros((cfg.n_layers, cfg.n_heads, 0, cfg.head_dim), dtype=np.dtype(cfg.dtype))
        return empty, empty, 0

    def decode(self, sequences: list[Sequence]) -> list[np.ndarray]:
        if not sequences:
            return []
        cfg = self.config
        last_tokens = np.asarray(
            [
                [seq.output_tokens[-1] if seq.output_tokens else seq.prompt_tokens[-1]]
                for seq in sequences
            ],
            dtype=np.int64,
        )  # (B,1)

        lengths: list[int] = []
        past: list[tuple[np.ndarray, np.ndarray, int]] = []
        empty_k = np.zeros((cfg.n_layers, cfg.n_heads, 0, cfg.head_dim), dtype=np.dtype(cfg.dtype))
        for seq in sequences:
            if seq.cache:
                k = np.stack([c[0] for c in seq.cache])
                v = np.stack([c[1] for c in seq.cache])
                length = k.shape[2]
            else:
                k, v, length = empty_k, empty_k, 0
            lengths.append(length)
            past.append((k, v, length))

        logits, k_layers, v_layers, pad_len = self._forward(last_tokens, past)

        # Strip the batch padding: a sequence owns [0:length] of the padded history
        # plus the freshly computed token at the end.
        for i, seq in enumerate(sequences):
            length = lengths[i]
            seq.cache = [
                (
                    np.concatenate(
                        [k_layers[layer][i, :, :length, :], k_layers[layer][i, :, pad_len:, :]],
                        axis=1,
                    ),
                    np.concatenate(
                        [v_layers[layer][i, :, :length, :], v_layers[layer][i, :, pad_len:, :]],
                        axis=1,
                    ),
                )
                for layer in range(cfg.n_layers)
            ]

        self.steps += 1
        self.tokens_processed += len(sequences)
        return [logits[i, -1, :] for i in range(len(sequences))]

    def release(self, sequence: Sequence) -> None:
        sequence.cache = []


# --------------------------------------------------------------------------- #
# Other backends
# --------------------------------------------------------------------------- #
class EchoBackend(Backend):
    """Deterministic toy backend used by smoke tests and the CLI demo."""

    name = "echo"
    VOCAB = 64

    def __init__(self, delay_ms: float = 0.0) -> None:
        self.delay_ms = delay_ms

    def _logits(self, sequence: Sequence) -> np.ndarray:
        logits = np.zeros(self.VOCAB, dtype=np.float32)
        seed = (sum(sequence.prompt_tokens) + len(sequence.output_tokens)) % self.VOCAB
        logits[seed] = 10.0
        logits[(seed + 1) % self.VOCAB] = 5.0
        return logits

    def prefill(self, sequence: Sequence) -> np.ndarray:
        sequence.prefilled = True
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        return self._logits(sequence)

    def decode(self, sequences: list[Sequence]) -> list[np.ndarray]:
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        return [self._logits(seq) for seq in sequences]


class VLLMBackend(Backend):
    """Proxy to a running vLLM (or any OpenAI-compatible) server.

    This is the production path: keep the batching, scheduling and metrics of
    kubeai, and delegate the heavy GPU work to vLLM over HTTP. ``prefill`` and
    ``decode`` collapse into a single blocking ``generate`` call per request, so
    this backend ignores the batcher's step interface by design.
    """

    name = "vllm"

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "default",
                 api_key: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.tokenizer = CharTokenizer()

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode())

    def prefill(self, sequence: Sequence) -> np.ndarray:  # pragma: no cover - network
        raise NotImplementedError("VLLMBackend serves through generate_openai()")

    def decode(self, sequences: list[Sequence]) -> list[np.ndarray]:  # pragma: no cover
        raise NotImplementedError("VLLMBackend serves through generate_openai()")

    def generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            data = self._post(
                "/chat/completions",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                },
            )
        except urllib.error.URLError as exc:  # pragma: no cover - network
            raise RuntimeError(f"vLLM backend unreachable at {self.base_url}: {exc}") from exc
        return data["choices"][0]["message"]["content"]


BACKENDS: dict[str, type[Backend]] = {
    "numpy": NumpyTransformerBackend,
    "numpy-transformer": NumpyTransformerBackend,
    "echo": EchoBackend,
    "vllm": VLLMBackend,
}


def build_backend(kind: str, **kwargs: Any) -> Backend:
    if kind not in BACKENDS:
        raise ValueError(f"unknown backend {kind!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[kind](**kwargs)
