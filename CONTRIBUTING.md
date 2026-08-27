# Contributing

Use a focused branch and keep unrelated user work intact. Install
`.[train,test]`, run `pytest -q` and `python examples/ci_smoke_test.py`, and add
tests for behavior changes.

Data and results must remain auditable: do not commit full raw datasets or model
weights; record source/checksum/config/seed; keep test selection isolated; and
generate tables through `scripts/summarize_results.py`. Never describe fixture
or smoke output as a pretrained model, formal benchmark, or paper reproduction.

Changes to bundled/upstream-derived code require provenance and license review.
By contributing repository-authored code, you agree to license that contribution
under the root MIT license. This does not relicense pre-existing GTpro-derived
files described in `LICENSE_SCOPE.md`.

Pull requests should state the goal, files changed, verification commands,
result provenance when relevant, and remaining limitations.
