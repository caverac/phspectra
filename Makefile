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
lint-python: ## Lint Python code with ruff
	cd packages/phspectra && uv run ruff check .

.PHONY: lint-python-fix
lint-python-fix: ## Lint + fix Python code with ruff
	cd packages/phspectra && uv run ruff check --fix .

.PHONY: format-python
format-python: ## Format Python code with ruff
	cd packages/phspectra && uv run ruff format .

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
ci-python: lint-python test-python typecheck-python ## Run full Python CI pipeline

.PHONY: ci
ci: ci-python ## Full CI pipeline (JS + Python)
	yarn ci
