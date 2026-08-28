# Public release audit

Audit date: 2026-08-28. This report covers the local publication candidates and
the merged original repository history. No versioned GitHub Release has been
published.

## Local candidate audit

`python scripts/audit_release.py` passed with 494 publication candidates, 488
UTF-8 text files, 116 result trace records, and 14 valid README-local links.
No candidate exceeded 50 MiB or matched the prohibited checkpoint/cache/log
classes. The scan found no private-key header, AWS access-key pattern, or
personal macOS/Linux absolute path. The versioned paper datasets carry a source
manifest and separate attribution statement. Run directories, joblib files,
generated shards, and model weights are not publication candidates.

Each copied result `metrics.json` has sibling `config.yaml` and
`environment.json`. `scripts/summarize_results.py` replaces the machine's
repository prefix with `${PROJECT_ROOT}` while copying these records, preserving
traceability without publishing a workstation path. README uses the paper's
high-resolution Figures 1, 2, 3, and 6 under `docs/assets/`, with DOI, figure
numbers, publisher copyright, and author-request attribution. The supplementary
`docs/assets/architecture.svg` is an original project-authored vector diagram.

## Git history and links

`git rev-list --all --count` reported eight commits after the optimized history
was joined to the original repository with a non-destructive merge. The release
audit must still be rerun before each tagged release.

README local links were checked automatically. External links were checked on
2026-08-27: the DOI/article, PubMed record, Tencent GROVER repository, and
corrected original repository resolved. The article's printed repository name
uses `multimodal`; the accessible repository is
`https://github.com/ZHUORANZHAO9816/mutimadel_molecular_property`. The latter has
five commits but no visible license file, so link correction does not remove the
license gate.

## License and external release gates

The bundled GROVER lineage includes its MIT attribution and the new engineering
work has an MIT license. The supplied original GTpro-derived files have no
verifiable license, so public redistribution remains blocked pending author or
rights-holder confirmation; see `LICENSE_SCOPE.md`.

Remote CI, Git tags/releases, and formal checkpoint hosting have not yet been
verified. Draft release notes are available under `docs/releases/`; none claims
that a versioned release already exists.
