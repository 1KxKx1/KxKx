# kubeai — a cloud-native AI platform

**A GPU-aware scheduler, a DAG workflow engine that compiles to Argo Workflows, and an
OpenAI-compatible inference gateway with continuous batching — one Python package, no cluster
required to run or test it.**

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-000000)](https://docs.astral.sh/ruff/)
[![Tests](https://img.shields.io/badge/tests-93%20passing-brightgreen)](tests)

[![CI](https://github.com/1KxKx1/KxKx/actions/workflows/ci.yml/badge.svg)](https://github.com/1KxKx1/KxKx/actions/workflows/ci.yml)

---

## Why this exists

Serving a model and training a model are two different scheduling problems that usually get
solved by two different teams with two different tools. In practice they share one scarce
resource — accelerator memory — and the interesting failures happen at the boundary:

* a fine-tune leaves 4 GiB stranded on every card, and no serving replica can use it;
* the serving queue grows while the CPU graph looks idle, because batching is the bottleneck;
* batch requests that finished early keep holding slots, so throughput *falls* as you add load;
* a pipeline runs on a laptop but has to be rewritten by hand before it can run on Kubernetes.

kubeai keeps the three concerns — **placement**, **pipeline**, **serving** — in one package so
the scheduling decisions, the workflow definitions and the inference path can share a model of
the hardware.

---

## What it does

| Subsystem | Responsibility |
|---|---|
| `kubeai.scheduler` | Discovers GPUs/NPUs via `nvidia-smi` (with a deterministic simulated fleet for CI), then places workloads with bin-packing, NVLink-topology awareness, fragmentation-aware scoring and priority preemption. |
| `kubeai.workflow` | A DAG engine with retries, timeouts, `when` gates and parameter passing. The same document runs locally for fast iteration and compiles to a real `argoproj.io/v1alpha1` Workflow and a Kubeflow `PyTorchJob`. |
| `kubeai.serving` | An OpenAI-compatible gateway: continuous batching, a memory-budgeted model registry with LRU eviction, streaming, Prometheus metrics and a queue-depth autoscaling signal. |

---

## Measured results

All numbers below are produced by this repository on the machine listed, and are reproducible
with `make bench`. The serving backend is a NumPy decoder-only transformer (28.7 M parameters,
vocabulary 23 044) so the benchmark needs no model download and no GPU.

**Host** — `Windows-11-10.0.26200`, AMD64 Family 25 (AMD Ryzen, 8 cores), Python 3.12.14.
Accelerator: `NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB` (detected and reported by
`kubeai gpus`, but not used by the NumPy backend).

### 1. Continuous batching vs static batching

32 requests arriving as a burst, output lengths drawn from 7–48 tokens (mean 26.5), prompt
8 tokens, `max_tokens=48`.

| Scenario | Throughput | vs serial | Decode-slot efficiency | p95 latency |
|---|---:|---:|---:|---:|
| serial (no batching) | 146.0 tok/s | 1.00× | 1.000 | 5.37 s |
| static batching (batch=8) | 115.7 tok/s | 0.79× | 0.622 | 7.05 s |
| static batching (batch=32) | 101.8 tok/s | 0.70× | 0.543 | 7.64 s |
| continuous batching (batch=8) | 169.4 tok/s | 1.16× | 1.000 | 4.82 s |
| **continuous batching (batch=32)** | **180.9 tok/s** | **1.24×** | **1.000** | **4.66 s** |

Two things are worth pointing at:

* **Static batching is slower than no batching at all** (0.70×). Grouping requests and decoding
  until the slowest member finishes means every early finisher keeps burning a slot: slot
  efficiency collapses to 0.54. With *uniform* output lengths the two strategies look identical —
  variable finish times are what expose the difference, which is why the benchmark generates them.
* **Continuous batching is 1.78× faster than static batching** at the same batch size
  (180.9 / 101.8) while holding slot efficiency at 1.000, and it is the only batched strategy
  that beats serial at all.

`decode-slot efficiency` is the fraction of `(decode step, sequence)` slots that produced a token
actually returned to a user. Each request needs `len(output) − 1` decode steps, because its first
token comes out of prefill; 1.000 means nothing was spent on work that had already finished.

**When batching pays, and when it does not.** The gap over *serial* depends on how much work each
step contains. With this 28.7 M-parameter model at mean 26.5 output tokens it is 1.24×; rerun the
same benchmark with a smaller model (`--model-dim 256 --model-layers 4`) and continuous batching
lands slightly *below* serial (0.96×), because the fixed per-step cost — sampling, metrics, and the
KV-cache gather/scatter described below — is no longer amortised over enough arithmetic. The
advantage over **static** batching, by contrast, held in every run I made (1.40× in that same
smaller-model run), because it comes from not wasting slots rather than from per-step efficiency.
That is why CI asserts the structural claims and only *reports* the serial ratio.

### 2. The batching effect in isolation (backend only)

The same model, decoded directly — no batcher, no queue, no metrics:

| Batch size | ms / step | tokens / second | vs batch=1 |
|---:|---:|---:|---:|
| 1 | 5.87 | 170.4 | 1.00× |
| 8 | 27.45 | 291.4 | 1.71× |
| 32 | 90.60 | 353.2 | 2.07× |

The gap between this table and the one above is what the scheduling layer costs: per-step KV-cache
gather/scatter, sampling, metrics and the async loop. See [Known limitations](#known-limitations).

### 3. DAG parallelism

26 tasks (25-way fan-out, then a merge), 0.25 s of simulated work per task:

| `max_parallel` | Wall clock | Sum of step time | Speedup |
|---:|---:|---:|---:|
| 1 | 6.40 s | 6.25 s | 1.00× |
| 4 | 1.82 s | 6.25 s | 3.52× |
| 8 | 1.03 s | 6.25 s | 6.22× |
| 16 | 0.76 s | 6.25 s | **8.40×** |

Charts are regenerated by `make charts` into [`benchmarks/results/`](benchmarks/results).

---

## Architecture

```
                        ┌──────────────────────────────────────────────┐
   OpenAI SDK ─────────▶│  kubeai.serving                              │
   LangChain  ─────────▶│                                              │
   eval harness ───────▶│  FastAPI gateway  /v1/chat/completions       │
                        │        │                                     │
                        │        ▼                                     │
                        │  ContinuousBatcher ── step ──▶ Backend       │
                        │   • rebuilds the batch every decode step     │
                        │   • retires finished sequences immediately   │
                        │   • admits queued work into freed slots      │
                        │        │                                     │
                        │        ▼                                     │
                        │  ModelRegistry  (memory budget + LRU evict)  │
                        └────────────┬─────────────────────────────────┘
                                     │ placement
                                     ▼
   kubeai CLI ──┐        ┌──────────────────────────────────────────────┐
   Argo        ─┼───────▶│  kubeai.scheduler                            │
   Kubeflow    ─┘        │  discover() ──▶ Node[] (GPU/NPU/CPU)         │
                        │  filter ─▶ score ─▶ bind ─▶ preempt           │
                        │  policies: binpack | spread | topology-aware │
                        └────────────┬─────────────────────────────────┘
                                     │ runs the steps
                                     ▼
                        ┌──────────────────────────────────────────────┐
                        │  kubeai.workflow                             │
                        │  Workflow (YAML) ─▶ DAG engine ─▶ Executor    │
                        │      │                    local | docker |    │
                        │      │                    dry-run            │
                        │      └─▶ compile ─▶ Argo Workflow CR         │
                        │                 └─▶ Kubeflow PyTorchJob      │
                        └──────────────────────────────────────────────┘
```

---

## Quick start

No cluster, no GPU, no model download.

```bash
git clone https://github.com/1KxKx1/KxKx
cd kubeai-platform
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,web]"
```

### See the hardware it found

```bash
kubeai gpus
```
```
accelerator source: nvidia-smi
idx  type  name                                total MiB  used MiB  free MiB  util %  temp  domain
---  ----  ----------------------------------  ---------  --------  --------  ------  ----  ------
0    gpu   NVIDIA GeForce RTX 4060 Laptop GPU  8188       1836      6352      27      46    -
```

With no accelerator present it prints the simulated fleet it fell back to and says so, so a CI
failure is never mistaken for "the GPU was busy".

### Run the whole platform as one pipeline

```bash
kubeai workflow run examples/selftest-pipeline.yaml --show-output
```

This executes a real DAG — accelerator inventory, a three-policy scheduling comparison and a
serving benchmark fan out in parallel, then merge into a report. Nothing is mocked.

### Compare scheduling policies

```bash
kubeai schedule --policy compare
```
```
policy          submitted  completed  preempted  avg util  peak util  makespan s  stranded MiB  pending
--------------  ---------  ---------  ---------  --------  ---------  ----------  ------------  -------
binpack         200        200        0          0.730     0.925      661.0       0             0
spread          200        200        1          0.646     0.920      661.0       0             9
topology-aware  200        200        0          0.673     0.925      661.0       0             18
```

200 tasks (serving replicas, fine-tunes, pre-trains and batch inference) over a simulated
4-node × 4× 24 GiB fleet, at 1-second resolution. Bin-packing reaches the highest average
utilisation here; `spread` preempts once where a serving replica needed a slot a batch job was
holding. Each policy gets its own copy of the fleet, so the numbers are the policy's and not the
previous run's leftovers.

### Serve models over an OpenAI-compatible API

```bash
kubeai serve --port 8080
```

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "explain continuous batching"}],
  "max_tokens": 32,
  "stream": true
}'
```

Any OpenAI client works unchanged:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="kubeai-tiny",
    messages=[{"role": "user", "content": "hello"}],
).choices[0].message.content)
```

Point it at real weights by registering a vLLM-backed model instead of the built-in one:

```bash
curl -s localhost:8080/v1/models/load -H 'Content-Type: application/json' -d '{
  "id": "qwen2.5-7b-instruct",
  "backend": "vllm",
  "footprint_mib": 16384,
  "max_batch_size": 64,
  "backend_options": {"baseUrl": "http://vllm:8000/v1", "model": "Qwen/Qwen2.5-7B-Instruct"}
}'
```

### Author a pipeline once, run it locally, deploy it to Kubernetes

```bash
kubeai workflow validate examples/training-pipeline.yaml
kubeai workflow compile  examples/training-pipeline.yaml --format argo --out out/argo.yaml
kubeai workflow compile  examples/training-pipeline.yaml --format k8s  --out out/cluster.yaml
```

The compile step is not a template render — it produces resources a cluster accepts, including
GPU requests translated to `nvidia.com/gpu` limits, the DAG as Argo `depends`, retry policies as
`retryStrategy`, and a Kubeflow `PyTorchJob` for the distributed training step:

```yaml
- name: train-job
  container:
    image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
    command: [torchrun, --nproc_per_node=2, -m, kubeai.pipelines.train]
    resources:
      limits:
        nvidia.com/gpu: 2
        kubeai.io/gpu-memory-mib: '40960'
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-A100-SXM4-40GB
  retryStrategy:
    limit: 3
    retryPolicy: OnFailure
    backoff: {duration: 30s, factor: 2, maxDuration: 600s}
```

---

## Scheduling

The loop mirrors the Kubernetes scheduler — filter, score, bind — with two changes that matter for
AI workloads: the allocation unit is accelerator **memory** (several small models share one card),
and **preemption is a first-class path** so training can make progress on a busy fleet.

| Policy | Behaviour | Use when |
|---|---|---|
| `binpack` | Fills the most-occupied device that still fits, with a penalty for leaving a leftover too small to reuse | You want whole nodes kept free for large jobs |
| `spread` | Favours the least-loaded *node*, not just the emptiest device | You want resilience and fewer noisy neighbours |
| `topology-aware` | Bin-packing plus an NVLink/HCCS island bonus, so a 4-GPU job lands on one fabric | Multi-GPU training where all-reduce dominates |

Preemption only evicts work that is **strictly lower priority and marked preemptible**, and the
victim is requeued rather than dropped. Jobs that are expensive to restart (training) stay
resident while serving replicas absorb the eviction.

Workloads are described with the same vocabulary Kubernetes uses:

```python
from kubeai.scheduler import Node, Scheduler, ResourceRequest, TaskRequest, TopologyPreference

job = TaskRequest(
    name="qwen-finetune",
    resources=ResourceRequest(count=2, memory_mib_per_device=40960, cpu_millicores=16000),
    priority=10,
    topology=TopologyPreference.SAME_DOMAIN,   # keep the pair inside one NVLink island
    preemptible=False,                          # do not evict me to serve a chat request
    node_selector={"nvidia.com/gpu.product": "NVIDIA-A100-SXM4-40GB"},
)
scheduler = Scheduler(nodes, policy="topology-aware")
scheduler.add_task(job)
placements, unscheduled = scheduler.schedule_once()
```

---

## Observability and autoscaling

`GET /metrics` exposes Prometheus-format series. `/admin/stats`, `/admin/gpus` and `/admin/queue`
give the registry, accelerator inventory and per-model queue depth as JSON.

```
kubeai_queue_depth                          pending requests
kubeai_active_sequences                     sequences currently decoding
kubeai_batch_size_bucket                    sequences per decode step
kubeai_time_to_first_token_seconds_bucket   TTFT histogram
kubeai_time_per_output_token_seconds_bucket inter-token latency histogram
kubeai_kv_padding_waste_ratio               padded KV slots / total KV slots
kubeai_generation_tokens_per_second         rolling throughput
```

Scaling on **queue depth and TTFT**, not CPU, is what keeps an inference deployment responsive —
a replica that is fully batched can look idle on CPU while users wait. `GET
/admin/autoscaling-hint` answers with a `scale-up` / `hold` / `scale-down` recommendation, and
[`deploy/manifests/keda-scaledobject.yaml`](deploy/manifests/keda-scaledobject.yaml) wires that
into KEDA. Alerting rules ship in the Helm chart.

---

## Deploy

```bash
helm install kubeai deploy/helm/kubeai -n ai-platform --create-namespace \
  --set serviceMonitor.enabled=true --set autoscaling.enabled=true

docker build -t kubeai-platform:0.4.0 .
docker run --rm -p 8080:8080 kubeai-platform:0.4.0
```

The chart ships a Deployment (non-root, readiness on `/readyz` so traffic only reaches pods with a
loaded model), HPA, PDB, ServiceMonitor, PrometheusRule, a model-registry ConfigMap and the RBAC
that lets Argo run training workflows. [`deploy/manifests/`](deploy/manifests) holds the pieces that
live outside the chart: namespace, Kubernetes `PriorityClass`es mirroring the scheduler's own
priority bands, the GPU inventory, and the KEDA trigger.

---

## Project layout

```
src/kubeai/
├── cli.py                 kubeai gpus | schedule | workflow | serve | bench
├── scheduler/
│   ├── device.py          nvidia-smi + topology discovery, simulated fleet
│   ├── model.py           Node, DeviceSlot, TaskRequest, Placement
│   ├── policies.py        binpack / spread / topology-aware scoring
│   ├── scheduler.py       filter -> score -> bind -> preempt
│   └── simulate.py        discrete-event workload simulation
├── workflow/
│   ├── spec.py            Workflow / Template / DAGTask (Argo-shaped subset)
│   ├── engine.py          async DAG execution, retries, when-gates, parameters
│   ├── executors.py       local | docker | dry-run
│   └── compile.py         -> Argo Workflow CR, -> Kubeflow PyTorchJob
└── serving/
    ├── backends.py        tokenizer, KV cache, NumPy transformer, vLLM proxy
    ├── batcher.py         continuous batching loop, streaming, cancellation
    ├── registry.py        memory budget, LRU eviction, placement
    ├── metrics.py         Prometheus registry
    ├── gateway.py         OpenAI-compatible FastAPI app
    └── bench.py           serial / static / continuous benchmark harness
```

---

## Testing

```bash
make test          # or: pytest -q
make lint          # ruff check
```

93 tests covering scheduling policies and preemption, DAG execution semantics (cycles, retries,
`when` gates, cross-step parameters), KV-cache correctness against padding, batcher behaviour
under cancellation and back-pressure, registry eviction, metric exposition, and the full
end-to-end self-test pipeline. A further 18 tests in `tests/test_gateway.py` exercise the
OpenAI-compatible HTTP surface; they skip automatically when the optional `[web]` extra is absent,
because the scheduler, workflow engine and batcher are all usable without FastAPI.

CI additionally runs the self-test DAG end to end, compiles both example pipelines, boots the
container and calls `/v1/chat/completions`, renders the Helm chart, checks that every `--json`
command emits parseable JSON, and asserts the two structural batching claims — continuous beats
static at the same batch size, and static slot efficiency stays below 0.95. That last assertion is
the one that matters: it is what fails if the static baseline ever starts admitting work mid-group
again, which is precisely the bug it was written for.

---

## Design notes

**Why continuous batching is rebuilt every step.** The batch is a scheduling decision, not a
configuration value. Rebuilding it each decode step is what lets a sequence that emitted EOS free
its slot immediately and a queued request join without waiting for anyone else. The benchmark above
is the argument: same batch size, same model, 1.78× the throughput of decode-until-all-finish.

**Why the workflow schema is an Argo subset.** If the in-cluster representation is a translation
target rather than the source of truth, the two drift. Keeping the document Argo-shaped means
`compile` is a rename-free mapping, and a data scientist's local run and the cluster run execute the
same graph.

**Why the scheduler owns memory, not devices.** One 24 GiB card can host three 7 GiB models. Whole-
device allocation either wastes 3 GiB per card or forces every workload to be the same size.

---

## Known limitations

This is a working platform, not a finished product. The honest list:

* **Per-step KV-cache gather/scatter is the main performance ceiling.** Caches are padded to the
  longest sequence in the batch and concatenated every step; the backend's own batching gains
  2.07× at batch 32, but end-to-end only 1.24× survives. That overhead is exactly the problem
  PagedAttention (vLLM) was designed to remove, and it is the first thing I would replace — with
  fixed-size blocks and a block table, so a step touches only the blocks it needs.
* **Prefill is not batched.** Requests are prefilled one at a time (up to
  `max_prefills_per_step`), so on prefill-heavy traffic TTFT is worse than it needs to be. Chunked
  prefill is the standard fix.
* **The NumPy backend is a benchmark harness, not a model server.** Weights are random on purpose:
  it makes the repository self-contained and the numbers reproducible. Real weights go through
  `VLLMBackend`.
* **The scheduler is in-process.** It places work onto a modelled fleet; it does not yet run a
  controller loop against the Kubernetes API or implement a device plugin. The GPU inventory
  manifest is the seam where that would attach.
* **The simulated fleet is deterministic, not statistically faithful.** It is a test fixture for
  scheduling logic, not a capacity-planning tool.
* **No multi-node collective fault tolerance.** A node failure mid-`torchrun` restarts the step via
  `retryStrategy`; it does not resume from a checkpoint.

## What I would build next

1. Paged KV cache with block tables — the single largest throughput win available.
2. Chunked, batched prefill to bring TTFT down under load.
3. A real device plugin + controller reconciling against `nvidia.com/gpu` allocations.
4. Speculative decoding behind the same `Backend` interface, measured with the same harness.
5. Replay-driven scheduling eval: feed a production trace instead of a synthetic distribution.

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
