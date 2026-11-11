# Preparing a source-only public release

The current checkout is not the complete publication surface. Earlier private
Git history contains benchmark text and screenshots even though they were removed
from HEAD. Do not make the original repository public on the strength of a clean
working tree or passing tests.

## Prepare a review candidate without history

After reviewing your working changes, run these commands from the project root:

```bash
make check
make release-history
```

The history check is expected to fail while earlier result artifacts remain
reachable. It reports paths, not transaction contents. It does not remove history
or inspect GitHub-only references, releases, caches, Actions artifacts or settings.

For a source-only release, prepare a **new folder outside this repository**:

```bash
make public-source PUBLIC_SOURCE_OUT=../fraud-decision-lab-public-source
```

The command refuses an existing destination or a destination inside the project.
It copies an explicit allowlist from the current working files known to Git,
including new unignored source files, so review pending changes before use.
It excludes `.git`, historical screenshots, private notes, external datasets,
processed records, results, cached responses, credentials and local environment
state. The only data file allowed is the original synthetic demo. Symlinks are
refused. A local Gitleaks installation is required; the script never installs it
or calls provider/network services.

`SOURCE_SNAPSHOT.json` records included paths and their SHA-256 hashes, the source
commit, whether working changes were present, and the snapshot time. This is a
review candidate, not an approval or a release. It contains no Git history and
does not modify the original repository. A failed scan leaves the candidate for
private inspection; do not publish it.

Review the candidate contents, scan it independently, and run its offline setup
and demo in a separate environment. Only after the release gates below are
resolved should a human choose a publication destination and initialize any new
public history. Never copy the original `.git` directory into the candidate.

## Mandatory release gates

- Review the exact files and snapshot hashes that will be distributed. Include
  source, configs, ordinary tests, license notices and original synthetic data
  only for the source-only release.
- Keep IEEE-CIS raw and derived artifacts out of distribution until publication
  rights are independently established. This includes labels, predictions,
  caches, aggregates, plots, trained models and report sections.
- Keep ULB-derived artifacts excluded until applicable attribution, redistribution
  and any derived-database obligations have been reviewed. A config permitting
  external processing does not approve publication.
- Review raw provider answers and trace metadata separately before sharing them.
  Do not assume that omission of request headers makes a trace public-safe.
- Verify the candidate has no secrets or private files. Review all public refs
  and GitHub-only releases, artifacts and settings at the actual destination.
- Run the local checks. Separately perform a current dependency advisory review;
  an unavailable offline scan is not a passing security result.
- Confirm private vulnerability reporting is enabled, owner/reviewer access is
  appropriate, and website/README links resolve after publication.
- Publish no quantitative historical claim without permitted prediction evidence,
  complete enough provenance, and clear limitations. Source-only distribution
  intentionally does not establish public verification of private experiments.

## Evidence and reproducibility

Saved predictions permit recalculating supported ranking metrics without paid
API calls. They do not establish execution authenticity or measured charges.
Report generation can fix plots and disclose incomplete historical evidence;
it must not fabricate missing calls, usage, timestamps or configurations.

Historical load tests without per-call records remain aggregate-only evidence.
Changing a prompt, feature rendering, training input or model setting requires
new affected predictions before presenting a new measurement. Fixing report
rendering, disclosure or aggregation of already saved attempts does not.

The supported project environment is Python 3.12 on macOS or Linux. Frozen Python
dependencies do not freeze the operating system, CPU or vendor-side model aliases.
Docker tags and apt package sources are mutable; record reviewed image digests
and distribution notices when releasing an image. No digest or vulnerability
assessment is inferred from the Dockerfile alone.

## Dependency checks

`make licenses` screens installed Python metadata for selected license markers.
It is not a legal opinion or a complete license notice inventory. Permissive
licenses and appropriately distributed MPL components can coexist with MIT code;
retain their applicable notices when redistributing dependencies or images.

`make vulnerabilities` is optional and uses `pip-audit` only if it is already
installed in the project environment. Network access is blocked; only available
cached advisory responses can be used. Missing tools, expired/missing cache data
or unsupported advisory behavior produce **NOT VERIFIED** and a nonzero exit.
No tool, package or advisory feed is downloaded. A cached result must still be
reviewed for freshness before a human treats it as a current release assessment.
