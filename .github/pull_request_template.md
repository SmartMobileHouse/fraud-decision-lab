## Summary

<!-- 1–3 sentences. What changed and why. Skip the "what the diff says" — that's
     visible in the diff. Focus on the why. -->

## Type

<!-- Pick the Conventional Commits type that matches your branch prefix and
     commit messages. -->

- [ ] `feat` — new capability
- [ ] `fix` — bug fix
- [ ] `docs` — documentation only
- [ ] `test` — tests only
- [ ] `refactor` — code change with no behaviour change
- [ ] `build` — dependencies, Docker, Makefile
- [ ] `chore` — anything else with no behaviour change

## Project checks

<!-- Tick the boxes that apply. If a row applies but you have a reason to skip
     it, leave it unticked and explain in the summary above. -->

- [ ] **No data or secrets.** No dataset rows, `.env` contents, raw provider
  traces or planning documents are committed. Secrets appear only by name and
  length.
- [ ] **No invented numbers.** Every result, metric or cost in code, docs or
  the report comes from a real run, cited by run ID.
- [ ] **Offline stays offline.** No new code path reaches the network without
  `--allow-external-requests`; tests stay offline (paid-API tests are marked
  `live`).
- [ ] **Dataset rights.** Datasets with `external_processing_allowed: false`
  still cannot reach any provider.
- [ ] **Same input for every model.** Payload rendering and the question are
  unchanged, or their version is bumped and the change is explained.
- [ ] **Provider docs checked.** Adapter changes were verified against the
  provider's current official API documentation (link it below).
- [ ] **Dependencies.** Any new dependency is MIT, BSD or Apache-2.0.

## Test plan

<!-- How was this validated? Bulleted, factual. -->

- [ ] `make check` passes
- [ ] Manual scenario:
