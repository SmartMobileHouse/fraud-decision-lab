# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them
privately through GitHub: **Security → Report a vulnerability** on this
repository, when private vulnerability reporting is enabled. If that option is
unavailable, contact Smart Mobile House privately through its website and ask
for a secure reporting channel before sending sensitive details. Include a
sanitised reproduction and expected impact. We aim to reply within 5 working days.

## What counts

- A way for secrets (API keys in `.env`) to appear in logs, reports, outputs
  or the Docker image.
- A way for a dataset marked `external_processing_allowed: false` to be sent
  to an external provider.
- A code path that calls a paid provider without `--allow-external-requests`,
  or bypasses the configured estimated spending guard.
- A way for unapproved dataset content or derived results to reach reports or
  exports, including aggregate IEEE-CIS results pending publication rights review.

## How the project protects you

- Offline CLI defaults; live calls need explicit flags, keys and a spending
  estimate. This estimate is not a guaranteed provider billing cap. Unknown usage,
  retries and provider pricing differences require separate accounting.
- `.env` is gitignored and excluded from the Docker image. Credential checks
  report names and lengths. Detailed traces remain private and require review
  before sharing, even when request headers were not recorded.
- `make check` provides a manually invoked secret scan and license metadata
  screen. It does not run automatically on every push and is not a vulnerability
  database or dataset-rights review.
- Offline tests block network connections. Real-API tests are excluded by default.
- Dataset publication permission is separate from external-processing permission.

The lab does not connect to payment execution systems or authorize financial
actions. Prompt injection and manipulated transaction fields can affect model
scores; this research does not claim adversarial robustness or production-grade
fraud controls. See [RELEASE.md](RELEASE.md) for the remaining publication gates.
