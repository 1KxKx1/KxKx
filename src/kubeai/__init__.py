"""kubeai — a cloud-native AI platform.

Three cooperating subsystems:

``kubeai.scheduler``
    Discover GPUs/NPUs/CPUs and place training and serving workloads onto them
    with bin-packing, topology awareness and priority preemption.

``kubeai.workflow``
    A DAG engine for model training / inference pipelines that runs locally and
    compiles to real Argo Workflow and Kubeflow PyTorchJob resources.

``kubeai.serving``
    An OpenAI-compatible inference gateway with continuous batching, a
    memory-aware model registry and Prometheus metrics.
"""

__version__ = "0.4.0"

__all__ = ["__version__"]
