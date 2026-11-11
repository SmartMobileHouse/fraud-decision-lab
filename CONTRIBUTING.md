# Contributing to fraud-decision-lab

Read [`README.md`](README.md) and [`CLAUDE.md`](CLAUDE.md) first. The hard
rules in `CLAUDE.md` (no invented numbers, offline by default, no secrets or
data in git, respect dataset rights, consistent feature schema) apply to
every change.

## Prerequisites

| Tool | Why | Install (macOS) |
|---|---|---|
| `uv` | Python 3.12 and pinned dependencies from `uv.lock` | `brew install uv` |
| `make` | Runs the local checks | Xcode Command Line Tools |
| `gitleaks` | Secret scan in `make check` | `brew install gitleaks` |
| Docker or Podman | Optional: build the offline image | `brew install --cask docker` or `brew install podman` |

## Workflow

```bash
uv sync --frozen
git switch -c feat/short-slug
# ... make your change, with tests ...
make format
make check          # must pass: there is no hosted CI
```

- **Conventional Commits:** `feat`, `fix`, `docs`, `test`, `refactor`,
  `build`, `chore`. Branches are named `type/slug`.
- **Tests:** correctness/security fixes need focused regression coverage; routine
  documentation edits do not need new tests. Tests must not reach the network. A
  test that calls a real paid API is marked `@pytest.mark.live` and is never
  run by default.
- **Providers:** check the provider's current official API docs before
  changing an adapter, and keep provider-specific logic inside its adapter.
- **Dependencies:** review compatibility and redistribution notices, including
  transitive and bundled code. `make licenses` screens metadata; it cannot prove
  legal compatibility. `make vulnerabilities` uses an installed `pip-audit` with
  network blocked and can use only locally cached advisory responses. Missing
  tools, cache misses or unavailable services mean **not verified**, not clean.
- **Release:** use [RELEASE.md](RELEASE.md). Local quality checks do not approve
  dataset rights, hidden GitHub artifacts, repository visibility or publication.

## Issues and pull requests

Use the issue forms (task, bug, tracking, decision) and the pull request
template. Keep pull requests small; a task holds at most 3 of them.
