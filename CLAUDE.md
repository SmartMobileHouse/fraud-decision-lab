# CLAUDE.md — working rules for this repository

Fraud Decision Lab benchmarks AI decision models (TypeSafe Jev, OpenAI
Decisions, Cloudflare Clef / Clef-flash, a structured-output LLM) against
XGBoost and rules for payment-fraud ranking. Python 3.12, uv, Typer, httpx,
XGBoost, Plotly. MIT licensed.

## Commands

```bash
uv sync --frozen        # install exactly what uv.lock pins
uv run fraudlab doctor  # offline setup check
make check              # ruff, mypy, pytest, licenses, gitleaks — must pass before any push
make format             # auto-fix lint/format
make docker             # build the offline image (Docker or Podman)
```

## Hard rules

- **Never invent numbers.** Results, metrics and costs come only from real
  runs with a run ID. Missing results are shown as `not_run`,
  `permission_blocked` or `failed`, never as placeholder numbers.
- **Offline by default.** No code path may call a provider without
  `--allow-external-requests`. Tests must never reach the network; the
  autouse guard in `tests/conftest.py` enforces this. Paid-API tests are
  marked `live` and are excluded by default.
- **Never commit or print secrets or data.** `.env`, `data/external/`,
  `data/processed/`, `outputs/`, raw provider traces and planning documents
  stay out of git; `tests/unit/test_repo_hygiene.py` enforces this. Report
  secrets by name and length only.
- **Respect dataset rights.** A dataset config with
  `external_processing_allowed: false` (for example IEEE-CIS) must never be
  sent to a provider, whatever the CLI flags say. Publication is separate:
  reports/exports require `publication_allowed` and a documented approval basis.
  Never infer publication rights from processing permission or aggregate format.
- **Same configured feature schema.** Providers receive the shared question and
  rounded payload through their own request formats; local models use original
  dataframe values. Disclose these differences. Changing rendering or questions
  requires a version bump and new prediction evidence where affected.
- **MIT-compatible distribution.** Review actual dependency obligations,
  including transitive and bundled components. MIT/BSD/Apache, ISC, PSF and
  applicable MPL components can coexist; MPL is not automatically incompatible.
  `make licenses` is a metadata screen, not a legal compatibility determination.
- **Honest spending and evidence.** Token-price calculations and unknown-cost
  reservations are not invoices or guaranteed billing caps. Preserve all attempts
  and immutable experiment identity. Do not rewrite historical evidence to make
  incomplete measurements look complete.
- **Verify provider APIs against current official docs** before writing or
  changing an adapter. Keep provider differences behind the common adapter
  interface.

## Conventions

- Conventional Commits: `feat`, `fix`, `docs`, `test`, `refactor`, `build`,
  `chore`. Branches are named `type/slug`.
- Keep commit authorship as configured in the repo; add AI assistance only as
  a `Co-Authored-By` trailer.
- Short, plain sentences in docs and issues. Explain project terms on first
  use.
