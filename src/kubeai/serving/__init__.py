"""OpenAI-compatible serving with continuous batching."""

from .backends import (
    BACKENDS,
    Backend,
    CharTokenizer,
    EchoBackend,
    NumpyTransformerBackend,
    Sequence,
    TransformerConfig,
    VLLMBackend,
    build_backend,
)
from .batcher import (
    ContinuousBatcher,
    FinishReason,
    GenerationRequest,
    QueueFull,
    sample_token,
)
from .metrics import MetricsRegistry, ServingMetrics
from .registry import InsufficientMemory, LoadedModel, ModelNotFound, ModelRegistry, ModelSpec

# The HTTP layer is an optional extra: the batcher, backends and registry are
# useful on their own (CLI benchmarks, embedded use), so a missing FastAPI must
# not make ``import kubeai.serving`` fail.
try:  # pragma: no cover - depends on the installed extras
    from .gateway import DEFAULT_MODEL, GatewayConfig, create_app, render_chat

    HTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    DEFAULT_MODEL = "kubeai-tiny"
    GatewayConfig = None  # type: ignore[assignment]
    create_app = None  # type: ignore[assignment]
    render_chat = None  # type: ignore[assignment]
    HTTP_AVAILABLE = False

__all__ = [
    "BACKENDS",
    "Backend",
    "CharTokenizer",
    "ContinuousBatcher",
    "DEFAULT_MODEL",
    "EchoBackend",
    "FinishReason",
    "GatewayConfig",
    "GenerationRequest",
    "HTTP_AVAILABLE",
    "InsufficientMemory",
    "LoadedModel",
    "MetricsRegistry",
    "ModelNotFound",
    "ModelRegistry",
    "ModelSpec",
    "NumpyTransformerBackend",
    "QueueFull",
    "Sequence",
    "ServingMetrics",
    "TransformerConfig",
    "VLLMBackend",
    "build_backend",
    "create_app",
    "render_chat",
    "sample_token",
]
