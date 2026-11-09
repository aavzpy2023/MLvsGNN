# Changelog

What each archived release of this repository contains, so that a reader can tell which state of the code produced
which state of the manuscript. Every release is archived on Zenodo with its own DOI; the concept DOI
[10.5281/zenodo.24083750](https://doi.org/10.5281/zenodo.24083750) always resolves to the latest.

## Versions and what they mean

| Version | State of the work |
| --- | --- |
| `v0.9.x` | milestones before submission: the analysis is complete in parts, the manuscript is not final |
| `v1.0.0` | the version submitted to the journal, and the DOI the manuscript cites |
| `v1.1`, `v1.2`, … | the revisions (R1, R2), one release each, so a reviewer can see exactly what changed |
| `v2.0` | the published version |

A release is created from a tag through the GitHub *Releases* page, which is what triggers the Zenodo archive; the
metadata of the archive comes from [`.zenodo.json`](.zenodo.json) and the citation from
[`CITATION.cff`](CITATION.cff).

## Unreleased — on the way to v1.0.0

- The generalization cohort reported as a section of its own: twelve MoleculeACE targets, with the asymmetry of the
  claim stated in Methods (the leakage half is audited on one study; the model-family half is what generalizes).
- Step 22, which aggregates the twelve targets into the tables and macros of that section, and steps 20, 21 and 22
  registered in `run_pipeline.py` so that a reviewer reaches them from the single entry point.
- The stronger neural baselines (pretrained CheMeleon, five-seed ensemble) on every target of the cohort, so that the
  comparison is against the models that reached parity on TgDHFR and not only against the untuned graph network.
- The title and the closing claim of the abstract, which change together with the scope of the paper.
- The four measured durations still written by hand, replaced by macros derived from the master logs (tracked as
  `KNOWN GAP` in `tests/test_no_new_literals.py`).
- A sensitivity table for the equivalence margin (±0.025, ±0.05, ±0.10).
- The journal, and the data-availability statement with the version DOI of the submission release.

## v0.9.0 — 2026-11-09

[DOI 10.5281/zenodo.24083751](https://doi.org/10.5281/zenodo.24083751) · 243 commits since 2025-12-12

First archived snapshot, published to verify the archiving circuit before submission.

**The analysis.** The audit of the published TgDHFR pipeline, with each leak removed one at a time and the score of
every rung of the ladder. The leakage-free benchmark of four molecular representations under 5×5 nested
cross-validation, with Nadeau–Bengio corrected paired tests, Holm adjustment and equivalence tests. Neural baselines
on the same folds: the default and tuned D-MPNN, a five-seed ensemble, a descriptor-augmented network and a graph
network initialized from a pretrained foundation model. Scaffold-grouped and outlier-sensitivity re-runs, and
y-randomization in four forms. Virtual screening of an FDA-approved library with its applicability domain, conformal
prediction intervals, docking and ADMET profiling. The published 80/20 split of twelve MoleculeACE targets, including
the activity-cliff subsets.

**The manuscript machinery.** Every number in the text is a macro the pipeline writes: the settings through step 00.1,
the measurements through the step that measures them. `MACROS.md` indexes all of them with their current value,
whether the text cites them and the command that regenerates each family. The supplement's table and figure numbers
are resolved with `\ref` across documents instead of typed in. 408 tests refuse a hand-written number, a macro that
disagrees with its code, an incoherent set of results, a compound named in the prose that is no longer in the data,
and a title that differs between the manuscript and the archive metadata.

**Not final at this point.** The per-target nested cross-validation of the generalization cohort was still running,
and the section that reports it was not written; results, discussion and conclusions change with it.
