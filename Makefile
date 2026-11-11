# Local quality gate (there is no CI). Run `make check` before every push.
# Repo lives on an external volume: uv cannot hardlink across disks, so copy.
export UV_LINK_MODE := copy
.PHONY: help install check lint format typecheck test licenses secrets doctor docker demo release-history public-source vulnerabilities

help:  ## List targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

install:  ## Install pinned dependencies from uv.lock
	uv sync --frozen

check: lint typecheck test licenses secrets  ## Everything below, in order
	@echo "make check: all passed"

lint:  ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint and formatting
	uv run ruff check --fix .
	uv run ruff format .

typecheck:  ## mypy on src/
	uv run mypy

test:  ## Offline tests (live tests are excluded by default)
	uv run pytest

licenses:  ## Screen installed license metadata (redistribution still requires review)
	uv run pip-licenses --partial-match --fail-on="GPL;AGPL;SSPL;Commons Clause;UNKNOWN" > /dev/null
	@echo "licenses: metadata screen passed; review distribution notices separately"

secrets:  ## Scan git history and staged changes for secrets (needs `brew install gitleaks`)
	@command -v gitleaks >/dev/null || { echo "gitleaks missing: brew install gitleaks"; exit 1; }
	gitleaks git --no-banner --redact --log-opts=--all .
	gitleaks git --no-banner --redact --pre-commit --staged .

doctor:  ## Check local setup
	uv run fraudlab doctor

docker:  ## Build the offline image (works with Docker or Podman)
	docker build -t fraud-decision-lab .
	@echo "Run it: docker run --rm fraud-decision-lab doctor"

demo:  ## Offline demo: rules + XGBoost on synthetic data, then the HTML report (no keys)
	uv run fraudlab demo

release-history:  ## Flag historical results/data before changing repository visibility
	python3 scripts/check_release_history.py

public-source:  ## Prepare a source-only candidate; set PUBLIC_SOURCE_OUT outside this repository
	@test -n "$(PUBLIC_SOURCE_OUT)" || { echo "Set PUBLIC_SOURCE_OUT to a new directory outside this repository"; exit 1; }
	python3 scripts/prepare_public_source.py --out "$(PUBLIC_SOURCE_OUT)"

vulnerabilities:  ## Optional cached-advisory scan; never installs a tool or permits network
	.venv/bin/python -B scripts/check_vulnerabilities_offline.py
