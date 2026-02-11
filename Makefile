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
	cd packages/phspectra && uv sync --dev

.PHONY: dev-docs
dev-docs: ## Start Docusaurus dev server
	yarn dev:docs

# ============================================================
# Python (Make owns this — Yarn does NOT touch Python)
# ============================================================

.PHONY: lint-python
lint-python: ## Lint Python code with flake8 + pylint
	cd packages/phspectra && uv run flake8 src/ tests/
	cd packages/phspectra && uv run pylint src/ tests/

.PHONY: format-python
format-python: ## Format Python code with black + isort
	cd packages/phspectra && uv run isort src/ tests/
	cd packages/phspectra && uv run black src/ tests/

.PHONY: format-python-check
format-python-check: ## Check Python formatting with black + isort
	cd packages/phspectra && uv run isort --check-only src/ tests/
	cd packages/phspectra && uv run black --check src/ tests/

.PHONY: docstyle-python
docstyle-python: ## Check docstring style with pydocstyle
	cd packages/phspectra && uv run pydocstyle src/

.PHONY: test-python
test-python: ## Run Python tests with pytest
	cd packages/phspectra && uv run python -m pytest -m "not slow"

.PHONY: test-python-smoke
test-python-smoke: ## Run slow smoke tests (GRS FITS, needs network + astropy)
	cd packages/phspectra && uv run --group smoke python -m pytest -m slow

.PHONY: typecheck-python
typecheck-python: ## Type-check Python code with mypy
	cd packages/phspectra && uv run mypy src/

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
