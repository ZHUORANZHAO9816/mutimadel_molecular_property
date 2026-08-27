# Public release audit

Audit date: 2026-08-27. This report covers the local publication candidates;
the repository still has an unborn `main` branch and no published release.

## Local candidate audit

`python scripts/audit_release.py` passed with 476 publication candidates, 474
UTF-8 text files, 116 result trace records, and 16 valid README-local links.
No candidate exceeded 50 MiB or matched the prohibited checkpoint/cache/log
classes. The scan found no private-key header, AWS access-key pattern, or
personal macOS/Linux absolute path. The ignored local full datasets, run
directories, joblib files, and model weights are not publication candidates.

Each copied result `metrics.json` has sibling `config.yaml` and
`environment.json`. `scripts/summarize_results.py` replaces the machine's
repository prefix with `${PROJECT_ROOT}` while copying these records, preserving
traceability without publishing a workstation path. The sole README image,
`docs/assets/architecture.svg`, is an original project-authored vector diagram.

## Git history and links

`git rev-list --all --count` reported zero commits reachable from normal branch
or tag history. Consequently there is no public Git history to scan yet; the
release audit must be rerun after staging/committing and before pushing.

README local links were checked automatically. External links were checked on
2026-08-27: the DOI/article, PubMed record, Tencent GROVER repository, and
corrected original repository resolved. The article's printed repository name
uses `multimodal`; the accessible repository is
`https://github.com/zzr624663649/mutimadel_molecular_property`. The latter has
five commits but no visible license file, so link correction does not remove the
license gate.

## License and external release gates

The bundled GROVER lineage includes its MIT attribution and the new engineering
work has an MIT license. The supplied original GTpro-derived files have no
verifiable license, so public redistribution remains blocked pending author or
rights-holder confirmation; see `LICENSE_SCOPE.md`.

Remote CI, Git tags/releases, and formal checkpoint hosting cannot be verified
in this local unborn repository. Draft release notes are available under
`docs/releases/`; none claims that a release already exists.
