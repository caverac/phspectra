# Contributing to PHSpectra

Thanks for your interest in contributing! This document covers the setup, workflow, and code quality standards for the project.

## Prerequisites

- [mise](https://mise.jdx.dev/) (manages Node 22, Python 3.11, uv)
- [Yarn 4](https://yarnpkg.com/) (bundled via Corepack)

## Setup

```bash
mise install
yarn install
uv sync --all-groups
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

The last command installs git hooks that run automatically on every commit. This is **required** before making any changes.

## Development workflow

1. Create a branch from `main`
2. Make your changes
3. Commit using [Conventional Commits](#commit-messages)
4. Open a pull request against `main`
5. CI must pass before merging

## Pre-commit hooks

Every commit is checked automatically by [pre-commit](https://pre-commit.com/). The hooks enforce all of the project's code quality standards so that issues are caught locally before reaching CI.

### Python hooks

| Hook       | What it does                      |
| ---------- | --------------------------------- |
| black      | Formats code (line length 120)    |
| isort      | Sorts imports (black-compatible)  |
| flake8     | Lints for style and common errors |
| pylint     | Static analysis                   |
| pydocstyle | Enforces NumPy-style docstrings   |
| mypy       | Strict type checking              |

### JS/TS hooks

| Hook     | What it does                     |
| -------- | -------------------------------- |
| prettier | Formats code                     |
| eslint   | Lints with zero warnings allowed |
| tsc      | TypeScript type checking         |

### Commit message hook

| Hook       | What it does                                     |
| ---------- | ------------------------------------------------ |
| commitlint | Validates Conventional Commit format (see below) |

If a hook fails, fix the issue and re-stage your changes. Black and isort auto-fix files in place, so you just need to `git add` the corrected files and commit again.

## Commit messages

All commits must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type(scope): description
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

Examples:

```
feat(phspectra): add multi-peak detection
fix(benchmarks): handle empty spectra in comparison
docs: update installation instructions
```

Commit messages drive automated releases — `feat` triggers a minor version bump, `fix` triggers a patch. A `BREAKING CHANGE` footer triggers a major bump.

## Code quality standards

This project enforces strict code quality. All of the following must pass in CI before a PR can be merged:

- **100% test coverage** enforced in the ci pipeline for all code
- **Zero lint warnings** across both Python and JS/TS
- **Strict mypy** with no untyped definitions
- **NumPy-style docstrings** on all public modules
- **120-character line length** for Python (configured in black, isort, pylint)

## Running checks manually

```bash
# Python
uv run black --check packages/phspectra/src/
uv run isort --check-only packages/phspectra/src/
uv run flake8 packages/phspectra/src/
uv run pylint packages/phspectra/src/
uv run pydocstyle packages/phspectra/src/
uv run mypy packages/phspectra/src/
uv run python -m pytest packages/phspectra/tests/ -v

# JS/TS
yarn lint
yarn format
yarn typecheck

# Full build
yarn build
```

## Project structure

```
packages/
  phspectra/       Core Python library
  benchmarks/      Benchmark CLI (phspectra vs GaussPy+)
  docs/            Docusaurus documentation site
  infrastructure/  AWS CDK stack
  pre-print/       LaTeX source for the paper
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
