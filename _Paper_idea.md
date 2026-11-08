# Paper outline (v2, Q1 upgrade) — leakage-free audit of a TgDHFR DNN pipeline, and classical ML vs. graph networks

Working title: *When Simple Is Enough: Leakage-Free Benchmarking of Classical Machine Learning vs. Neural Models* (the audit axis may justify a new title; decide in the rewrite).

Supersedes the section map in `PAPER_OUTLINE.md` (v1, benchmark only). Built from the project memory as of 2026-10-21. Numbers are those in `results/` after the two-stage `train_internal` fix; the manuscript must take them from `latex/*.tex`, never retype them.

**Central question.** Karimova & Senanayake 2025 (AI Chemistry 3:100084) report R² ≈ 0.82–0.86 for a TgDHFR DNN ensemble with feature selection and Gaussian-noise augmentation. How much of that survives a leakage-free evaluation, and once it does, do neural models (DNN, D-MPNN, CheMeleon) buy anything over classical ensembles on fingerprints?

**Thesis (changed from v1).** Not "classical wins" but: (1) the published gain is mostly leakage; (2) leak-free, strong neural models are statistically equivalent to Morgan ensembles (TOST) — at a fraction of the cost for the classical side.

**Status:** ✔ done and checked · 🔄 run in progress / numbers provisional · ✎ text must be rewritten · ☐ not started · ⏸ postponed.

---

## Abstract  ✎

Rewrite last. Must state: the leaks found in the original code, the size of the inflation, that the gains are not reproduced leak-free, and that strong neural baselines are equivalent (not better) than classical ones. Keep what was **not** shown: selectivity against human DHFR, validated leads, generality beyond one target.

## 1. Introduction  ✎

- **1.1 Low-data regimes** — small sets overfit; simple models often hold up.
- **1.2 Engineered features vs. end-to-end learning** — fingerprints/descriptors vs. GNNs and pretrained models; the original TgDHFR study (noise augmentation + DNN ensemble) as the case under test.
- **1.3 Data leakage** — preprocessing, feature selection and early stopping must stay inside the fitting loop; motivates nested CV and the audit.
- **1.4 Objectives and contributions** — new list:
  1. Audit of the original pipeline: three concrete leaks, reproduced and quantified.
  2. A leakage ladder that removes the leaks one at a time on a common test set.
  3. Leak-free nested-CV benchmark with Nadeau–Bengio corrected tests and TOST equivalence.
  4. Strong neural baselines (DNN, MLP-ECFP, D-MPNN ensemble, D-MPNN+RDKit, CheMeleon) against classical ensembles.
  5. Robustness: scaffold-grouped splits, extreme potencies, activity cliffs.
  6. Screening, docking and ADMET as hypothesis generation, not validated leads.

## 2. Materials and Methods

- **2.1 Dataset and curation ✔** — 859 compounds; Tukey exclusion of two extremes; duplicates by canonical SMILES.
- **2.2 Representations ✔** — Morgan, RDKit 2D, combinations with 3D (ETKDGv3 + MMFF).
- **2.3 Nested cross-validation ✔** — repeated 5×5; inner search over stacking ensembles; preprocessing inside folds; Nadeau–Bengio + Holm. Add: ± 3 train-range clipping safeguard (step 01) after out-of-range descriptors exploded through Yeo–Johnson in the DNN. ✎
- **2.4 Audit of the original pipeline ☐ (new)** — what the code does (local notebooks, TF/Keras + feature-engine + PaDEL, no seeds for augmentation noise), the three leaks:
  - permutation importance computed on the 15 % test split, then used to select features scored on that split;
  - `EarlyStopping(restore_best_weights)` monitors the same set R² is reported on;
  - augmentation notebook concatenates preprocessed train+test and re-splits (preprocessing fitted on rows that end up in the hold-out).
  Plus the verbatim re-run protocol (step 15a notebooks, 15b ten seeds, TF 2.17.1 environment).
- **2.5 Leakage ladder ☐ (new)** — rungs R0 (as published) → R1 (no re-split) → R2 (internal early stopping) → R3 (out-of-fold importance) → R4 (no duplicates) → clean baseline; common test set of 130 molecules; note that only 20 of the 131 R0 scoring molecules come from the original test set. Two-stage `train_internal`: internal early stopping only picks epochs/LR schedule; the final net is retrained on 100 % of training rows (the 85 % training had confounded R1→R2).
- **2.6 Classical benchmark counterpart of the leak ✔/✎** — step 04: Bleaky vs. clean B on a classical model (selecting bits on the test set alone reproduces ~0.83 with RF/XGB/SVM), with out-of-fold importance. Mention `DEPLOYED_ARCHITECTURE`.
- **2.7 Neural models ☐/✔** — DNN in nested CV on the same outer folds (step 17); MLP-ECFP, D-MPNN ensemble of 5, D-MPNN+RDKit, CheMeleon (pretrained, ~4.6 h/fold) in step 19; default and tuned D-MPNN (steps 09, 12) kept. Folds exported once (step 14) so every model sees identical splits.
- **2.8 Statistics ✎** — corrected paired tests, TOST with a stated margin; **D2 ☐**: Bayesian ROPE analysis as a complement (pending).
- **2.9 Screening and ligand efficiency ✔** — FDA library, domain filters, Isolation Forest; now with the 765-bit mask.
- **2.10 Docking ✔** and **2.11 ADMET ✔** — unchanged.
- **2.12 Environment and reproducibility ✔** — two conda environments (main, `karimova_repro` with TF), lock files, seeds, run times, tmux/`run_pipeline` stages.

## 3. Results

- **3.1 Reproducing the published numbers 🔄** — verbatim notebooks: 0.753 / 0.824 / 0.858 vs. printed 0.746 / 0.816 / 0.858; ten fresh seeds: 0.763 ± 0.010 (baseline), ~0.81 (selection), ~0.857 (augmentation). The published numbers are reproducible *with* the leaks.
- **3.2 Leakage ladder 🔄** — R0 0.853 → R1 0.825 → R2 0.800 → R3 0.752 → R4 0.740; clean baseline 0.752. Selection and augmentation add nothing once leak-free (clean selection variants 0.71–0.73 < baseline). Each rung's Δ with CI in `18_leakage_ladder_deltas.csv`. Figure 16.1 ☐.
- **3.3 Classical counterpart: selecting on the test set ✔** — Bleaky 0.830 vs. B 0.741 (+0.096, CI [+0.055, +0.147]): a classical model reaches the published figure by leaking.
- **3.4 Classical ensembles across feature spaces ✔** — fingerprints carry the signal; original selection/augmentation gains replayed on a classical learner.
- **3.5 Classical ensembles vs. neural models 🔄 (central benchmark, narrative changed)**
  - Random folds, step 19: CheMeleon 0.762 and D-MPNN ens5 0.741 are TOST-equivalent to Morgan (R² and MAE); MLP-ECFP 0.709 and D-MPNN+RDKit 0.701 are lower.
  - Step 17 DNN (leak-free): baseline 0.673, full pipeline 0.715 vs. Morgan ~0.76 (provisional until paired tests are regenerated).
  - Default and tuned D-MPNN kept. Costs: classical minutes vs. D-MPNN ens5 ~35 min/fold vs. CheMeleon ~4.6 h/fold.
  - Message: equivalent at a fraction of the cost. Figure 17.1 ☐.
- **3.6 Robustness: scaffold-grouped splits 🔄** — now four classical representations plus neural models (current: Morgan 0.76 → 0.58, GNN 0.72 → 0.53; drop ≈ 0.18–0.24). Waiting for SC17 (DNN) and SC19 (D-MPNN ens5, CheMeleon), then step 13 + 11. **D1 ☐**: add activity-cliff analysis.
- **3.7 Sensitivity to extreme potency values ✔** — unchanged.
- **3.8 Virtual screening ✎** — after the new mask: domain 1927 compounds, same 4 consensus hits, top-10 ranks 8–10 changed (terconazole, itraconazole…), Spearman −0.42 vs. previous ranking. Text naming bendamustine / erdafitinib / brigatinib and "1255 bits" must be rewritten by hand.
- **3.9 ML–docking consensus ✎** and **3.10 ADMET profiling ✎** — regenerate with the new candidate list.
- **3.11 Second target ⏸** — human DHFR or a MoleculeACE set to show the audit conclusions are not specific to TgDHFR. Postponed by the user; Caco-2 as a second dataset was judged low value.

## 4. Discussion

- **4.1 What the audit means ✎** — the apparent benefit of selection and augmentation is an evaluation artifact; how each leak inflates R²; recommendations for reporting (nested CV, common test sets, equivalence tests).
- **4.2 Inductive bias in small data ✎** — equivalence, not superiority, of classical and neural; caveats: modest network search, no learning curve, one target.
- **4.3 Dimensionality, 3D overhead and cost ✔/✎** — measured cost per model family, now including pretrained and ensemble networks.
- **4.4 Scope and limitations ✎** — scaffold-split drop, activity cliffs, unstable inner search, ADMET model limits, no selectivity against human DHFR, y-randomization scope (below), single target.
- **4.5 Conclusions and future work ✎** — (i) published gains are not reproducible leak-free, (ii) neural ≈ classical at far higher cost, (iii) screening gives hypotheses only.

## Rewrite checklist (phase 5)

- Y-randomization (step 03) runs RF+XGB+SVM on all 2048 Morgan bits, while the screening model (step 05) uses the selected mask: state that it validates the architecture, not the final selected model.
- Replace "classical wins" framing with "equivalent at a fraction of the cost".
- Update every 1255-bit / old-candidate mention to the 765-bit mask and new ranking.
- Re-run `11_check_manuscript` after the scaffold and step 17/19 outputs are final.
- Open items from the brief not recorded in detail in memory: E1/E2 (confirm with the user).

## Pipeline map (what produces what)

| Paper part | Steps |
| :--- | :--- |
| Data (2.1, 3.7) | `00`, `10` |
| Classical benchmark and statistics (2.3, 3.3, 3.4) | `01`, `02`, `03`, `04` |
| Fold export (shared splits) | `14` |
| Audit: verbatim and 10-seed reproduction (2.4, 3.1) | `15` (TF env) |
| Leakage ladder (2.5, 3.2) | `16` (TF env) |
| DNN in nested CV (2.7, 3.5) | `17` (TF env), `SC17` on scaffold |
| Audit tables / LaTeX variables | `18` → `karimova_audit_variables.tex` |
| D-MPNN default and tuned (2.7, 3.5) | `09`, `12` |
| Strong neural baselines (2.7, 3.5) | `19` → `gnn_strong_variables.tex`, `SC19` on scaffold |
| Screening, docking, ADMET (2.9–2.11, 3.8–3.10) | `05`–`08` |
| Robustness (3.6 scaffold, 3.7 outliers) | `SC01`, `SC09`, `SC17`, `SC19`, `13`; `S01`, `S02`, `10` |
| Figures, tables and manuscript check | `*.1` scripts (new: 16.1, 17.1), `11` |

## Next actions

1. Wait for tmux `ladder` (16 → 18) and `pipeline` (17 → SC17 → SC19 → 13 → 11); review the outputs.
2. Figures 16.1 and 17.1; extend step 13 to four representations plus activity cliffs (D1).
3. D2 Bayesian ROPE.
4. Rewrite per the checklist; nothing from the Q1 work is committed yet (results, LaTeX variables and figures pending).
