.DEFAULT_GOAL := help
SHELL := /bin/bash

# ============================================================
# Help
# ============================================================

.PHONY: help
help: ## List all targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ============================================================
# Development
# ============================================================

.PHONY: install
install: ## Install all dependencies (JS + Python)
	yarn install
	uv sync --all-groups

.PHONY: dev-docs
dev-docs: ## Start Docusaurus dev server
	yarn dev:docs

# ============================================================
# Python (Make owns this — Yarn does NOT touch Python)
# ============================================================

PYTHON_SRC := packages/phspectra/src/ packages/phspectra/tests/

.PHONY: lint-python
lint-python: ## Lint Python code with flake8 + pylint
	uv run flake8 $(PYTHON_SRC)
	uv run pylint $(PYTHON_SRC)

.PHONY: format-python
format-python: ## Format Python code with black + isort
	uv run isort $(PYTHON_SRC)
	uv run black $(PYTHON_SRC)

.PHONY: format-python-check
format-python-check: ## Check Python formatting with black + isort
	uv run isort --check-only $(PYTHON_SRC)
	uv run black --check $(PYTHON_SRC)

.PHONY: docstyle-python
docstyle-python: ## Check docstring style with pydocstyle
	uv run pydocstyle packages/phspectra/src/

.PHONY: test-python
test-python: ## Run Python tests with pytest
	uv run python -m pytest -m "not slow"

.PHONY: test-python-smoke
test-python-smoke: ## Run slow smoke tests (GRS FITS, needs network + astropy)
	uv run --group smoke python -m pytest -m slow

.PHONY: typecheck-python
typecheck-python: ## Type-check Python code with mypy
	uv run mypy packages/phspectra/src/

# ============================================================
# CI
# ============================================================

.PHONY: ci-python
ci-python: lint-python format-python-check docstyle-python test-python typecheck-python ## Run full Python CI pipeline

.PHONY: ci
ci: ci-python ## Full CI pipeline (JS + Python)
	yarn ci

# ============================================================
# Infrastructure
# ============================================================

INFRA_DIR := packages/infrastructure
PHSPECTRA_DIR := packages/phspectra
WORKER_DIR := $(INFRA_DIR)/lambda/worker

.PHONY: build-phspectra-wheel
build-phspectra-wheel: ## Build phspectra wheel for Lambda worker
	cd $(PHSPECTRA_DIR) && uv build --wheel --out-dir ../../$(WORKER_DIR)/

.PHONY: build-lambdas
build-lambdas: build-phspectra-wheel ## Build Lambda Docker images (via CDK)
	cd $(INFRA_DIR) && yarn build

.PHONY: synth
synth: build-phspectra-wheel ## Synthesise CloudFormation template
	cd $(INFRA_DIR) && ENVIRONMENT=development yarn cdk synth

.PHONY: deploy
deploy: build-phspectra-wheel ## Deploy stack to AWS (ENVIRONMENT=development by default)
	cd $(INFRA_DIR) && ENVIRONMENT=$${ENVIRONMENT:-development} yarn cdk deploy --require-approval broadening

.PHONY: diff
diff: build-phspectra-wheel ## Show CDK diff against deployed stack
	cd $(INFRA_DIR) && ENVIRONMENT=$${ENVIRONMENT:-development} yarn cdk diff

.PHONY: upload-test-cube
upload-test-cube: ## Upload GRS test field FITS to S3
	aws s3 cp tests/fixtures/grs-test-field.fits s3://phspectra-development-data/cubes/grs-test-field.fits

.PHONY: upload-beta-sweep
upload-beta-sweep: ## Upload beta sweep manifest to S3
	@echo '{"cube_key":"cubes/grs-test-field.fits","survey":"grs","beta_values":[3,4,5,6,7,8]}' | \
		aws s3 cp - s3://phspectra-development-data/manifests/grs-beta-sweep.json
