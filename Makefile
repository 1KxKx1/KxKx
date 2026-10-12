.PHONY: help install install-dev test lint fmt bench bench-serving bench-workflow \
        workflow-run workflow-validate compile-argo compile-k8s serve gpus schedule \
        docker clean all

PY ?= .venv/Scripts/python.exe
ifeq ($(OS),)
PY := python
endif
ifeq ($(shell uname -s 2>/dev/null),Linux)
PY := python
endif

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	$(PY) -m pip install -e .

install-dev:  ## Install with the torch, metrics and dev extras
	$(PY) -m pip install -e ".[dev,web]"

test:  ## Run the test suite
	$(PY) -m pytest -q

test-cov:  ## Run tests with a coverage report
	$(PY) -m pytest -q --cov=kubeai --cov-report=term-missing

lint:  ## Lint and import-sort check
	$(PY) -m ruff check src tests benchmarks

fmt:  ## Auto-fix lint issues
	$(PY) -m ruff check --fix src tests benchmarks

gpus:  ## Show the accelerators discovered on this host
	$(PY) -m kubeai gpus

schedule:  ## Compare scheduling policies on a synthetic workload
	$(PY) -m kubeai schedule --policy compare

serve:  ## Start the OpenAI-compatible gateway
	$(PY) -m kubeai serve --port 8080

workflow-validate:  ## Validate the bundled example workflows
	$(PY) -m kubeai workflow validate examples/selftest-pipeline.yaml
	$(PY) -m kubeai workflow validate examples/training-pipeline.yaml

workflow-run:  ## Run the self-test DAG end to end (no images required)
	$(PY) -m kubeai workflow run examples/selftest-pipeline.yaml --show-output

compile-argo:  ## Compile the training pipeline to an Argo Workflow
	$(PY) -m kubeai workflow compile examples/training-pipeline.yaml \
		--format argo --out deploy/manifests/argo-training-pipeline.yaml

compile-k8s:  ## Compile the training pipeline to a full cluster bundle
	$(PY) -m kubeai workflow compile examples/training-pipeline.yaml \
		--format k8s --out deploy/manifests/cluster-training-bundle.yaml

bench: bench-serving bench-workflow  ## Run every benchmark

bench-serving:  ## Measure continuous vs static batching throughput
	$(PY) -m kubeai bench serving --requests 48 --max-tokens 24 --batch-sizes 1,4,16 --json

bench-workflow:  ## Measure DAG parallelism speedup
	$(PY) -m kubeai bench workflow --requests 24 --json

charts:  ## Render benchmark result charts
	$(PY) benchmarks/plot_results.py

docker:  ## Build the container image
	docker build -t kubeai-platform:0.4.0 .

clean:  ## Remove caches and run artefacts
	rm -rf .pytest_cache .ruff_cache .kubeai build dist *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

all: lint test bench  ## Lint, test and benchmark
