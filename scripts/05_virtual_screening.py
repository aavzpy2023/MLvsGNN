import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import warnings
import time
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

# ── Configuration and Logging Integration ────────────────────────────────────
from paths_config import *
from logger_utils import setup_logger, explain
from qsartools.applicability_domain import (contains_only_elements, fit_similarity_domain, in_similarity_domain,
                                            nearest_similarity, similarity_to_domain)
from qsartools.featurization import morgan_fingerprints
from qsartools.latex import format_macros
from qsartools.ligand_efficiency import ligand_efficiency_metrics

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

LOG_FILE = os.path.join(LOGS_DIR, "05_virtual_screening.log")
OUTPUT_CSV = FDA_CANDIDATES_CSV


def find_control(df_res, drug):
    """Exact (case-insensitive) name match first, so that 'Trimethoprim' is not resolved to a combination product."""
    exact = df_res[df_res['Name'].str.lower() == drug.lower()]
    return exact if not exact.empty else df_res[df_res['Name'].str.contains(drug, case=False, na=False)]


# =========================================================
# MAIN PIPELINE
# =========================================================
def run():
    # 1. Initialize the agnostic logger
    setup_logger(LOG_FILE)

    start_pipeline = time.time()
    print("=" * 100)
    print("VIRTUAL SCREENING PIPELINE — TgDHFR Inhibitor Search")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    explain("What this step does", """
        Uses the trained model from step 04 to predict how strongly each FDA-approved drug would inhibit the parasite enzyme
        TgDHFR (Toxoplasma gondii dihydrofolate reductase), without doing any lab work (virtual screening).

        1) Load the model. 2) Measure how similar each drug is to the training molecules (Applicability Domain: the model is
        only trustworthy for molecules similar to those it was trained on); drugs far from them are still scored, but flagged.
        3) Drop drugs that are very large (MW above the limit) or contain atoms outside the allowed set (H, C, N, O, F, P, S,
        Cl, Br, I). 4) Predict pIC50. 5) Report and export the ranking with each drug's similarity to the training set.

        pIC50 = -log10(IC50 in molar). Higher means more potent: 6 is IC50 = 1 uM, 7 is 100 nM, 8 is 10 nM.
        Candidates are later confirmed by docking (step 07) and filtered for drug-likeness/safety (step 08).
    """)

    # ── ML Architecture Loading ──────────────────────────────────────────────
    print(f"\n[1/5] ML ARCHITECTURE LOADING")
    print(f"  [-] Model path: {MODEL_FILE}")
    if not os.path.exists(MODEL_FILE):
        print(f"  [ERROR] {MODEL_FILE} not found. Interrupting execution.")
        sys.exit(1)

    t0 = time.time()
    model = joblib.load(MODEL_FILE)
    print(f"  [+] Ensembled model loaded successfully ({time.time() - t0:.2f}s)")

    mask = None
    if os.path.exists(MASK_FILE):
        mask = np.load(MASK_FILE)
        n_selected = mask.sum()
        total_feats = len(mask)
        print(f"  [-] Mask path: {MASK_FILE}")
        print(
            f"  [+] Mask applied: {n_selected} / {total_feats} features ({(n_selected / total_feats) * 100:.2f}% retention)")
    else:
        print("  [!] No mask detected. Using full 2D Morgan topology (2048 dimensions).")

    # ── Applicability Domain Configuration ────────────────────────────────────
    print(f"\n[2/5] APPLICABILITY DOMAIN CONFIGURATION (SIMILARITY TO THE TRAINING SET)")
    print(f"  [-] Training source: {TRAIN_FILE}")
    t0 = time.time()

    df_train = pd.read_csv(TRAIN_FILE).dropna(subset=['Smiles', 'pIC50 Value'])
    train_mols = [Chem.MolFromSmiles(s) for s in df_train['Smiles']]
    X_train_full = morgan_fingerprints(train_mols).astype(np.float32)

    # Similarity is computed on the full 2048-bit fingerprints (Sheridan et al.: all the descriptors)
    domain = fit_similarity_domain(X_train_full, k=AD_K, z=AD_Z)
    train_similarity = nearest_similarity(X_train_full, X_train_full, AD_K, exclude_self=True)

    print(f"  [+] Training fingerprints generated: {X_train_full.shape}")
    print(f"  [+] Similarity domain fitted (k={AD_K}, z={AD_Z}) in {time.time() - t0:.2f}s: minimum mean similarity to the "
          f"{AD_K} nearest training molecules = {domain['min_similarity']:.3f} "
          f"(training molecules, one against the others: median {np.median(train_similarity):.3f})")
    explain("Applicability Domain", f"""
        Prediction error grows as a molecule becomes less similar to the training set (Sheridan et al., 2004). Each FDA drug
        is given its mean Tanimoto similarity to its {AD_K} most similar training molecules. A drug is flagged as inside the
        domain when that value is at least {domain['min_similarity']:.3f}, which is 1 - (mean + {AD_Z} standard deviations) of the
        same distance measured for the {len(df_train)} training molecules against each other (Sahigara et al., 2012). The
        multiplier is arbitrary: the literature gives no rule for it. No drug is excluded on this flag; it is reported with
        every candidate, because predictions outside the domain are extrapolations.
    """)

    # ── FDA Library Ingestion and Structural Filtering ────────────────────────
    print(f"\n[3/5] FDA LIBRARY PROCESSING")
    print(f"  [-] Screening source: {FDA_FILE}")
    if not os.path.exists(FDA_FILE):
        print(f"  [ERROR] {FDA_FILE} not found.")
        sys.exit(1)

    df_fda = pd.read_csv(FDA_FILE)
    df_fda.columns = df_fda.columns.str.strip()
    initial_fda_count = len(df_fda)
    print(f"  [+] Raw compounds imported: {initial_fda_count}")

    t0 = time.time()
    candidates = []
    for _, row in df_fda.iterrows():
        smi = row.get('isosmiles') or row.get('canonicalsmiles')
        if pd.isna(smi): continue

        mol = Chem.MolFromSmiles(str(smi))
        if mol is None: continue

        mw = row.get('mw')
        mw = Descriptors.MolWt(mol) if (pd.isna(mw) or mw == '') else float(mw)

        if mw > MAX_MW or not contains_only_elements(mol, ALLOWED_ATOMS):
            continue

        fp = morgan_fingerprints([mol])[0].astype(np.float32)
        candidates.append({
            'CID': row.get('cid'), 'Name': row.get('cmpdname'),
            'SMILES': str(smi), 'FP': fp,
        })

    structural_retention = (len(candidates) / initial_fda_count) * 100
    print(
        f"  [+] Physical/Structural Filtering (MW < {MAX_MW}, Allowed Atoms): {len(candidates)} candidates retained ({structural_retention:.1f}%)")

    # ── In-Domain Filtering ───────────────────────────────────────────────────
    X_fda_full = np.array([c['FP'] for c in candidates])
    fda_similarity = similarity_to_domain(domain, X_fda_full)
    inside = in_similarity_domain(domain, X_fda_full)
    final = candidates
    X_final = X_fda_full[:, mask] if mask is not None else X_fda_full

    print(f"  [+] Similarity to the training set of the {len(candidates)} candidates: median {np.median(fda_similarity):.3f} "
          f"(5th-95th percentile {np.percentile(fda_similarity, 5):.3f}-{np.percentile(fda_similarity, 95):.3f})")
    print(f"  [+] Inside the applicability domain: {int(inside.sum())} compounds ({100 * inside.mean():.1f}%); "
          f"all are scored, none is excluded")
    print(f"  [+] Total filtering time: {time.time() - t0:.2f}s")
    explain("Reading the filter counts", f"""
        {initial_fda_count} drugs in the library -> {len(candidates)} after size/atom filters, all scored -> {int(inside.sum())}
        of them inside the model's domain. Drugs lost at the first stage were never scored. A low share inside the domain
        means the FDA library is chemically far from the training set: the predictions are then extrapolations and are
        less reliable, however high they rank.
    """)

    # ── ML Inference ──────────────────────────────────────────────────────────
    print(f"\n[4/5] ML INFERENCE - BIOACTIVITY PREDICTION (pIC50)")
    t0 = time.time()
    preds = model.predict(X_final)
    if preds.ndim > 1:
        preds = preds.flatten()

    print(f"  [+] Inferences completed in {time.time() - t0:.3f}s")
    print(f"  [-] Prediction Statistics -> Mean: {preds.mean():.4f} | Max: {preds.max():.4f} | Min: {preds.min():.4f}")

    # Result dataframe construction
    rows = []
    for i, c in enumerate(final):
        metrics = ligand_efficiency_metrics(preds[i], c['SMILES'], decimals=2)
        rows.append({
            'CID': c['CID'], 'Name': c['Name'], 'SMILES': c['SMILES'],
            'pIC50_pred': round(float(preds[i]), 4),
            **metrics,
            'AD_similarity': round(float(fda_similarity[i]), 4),
            'In_AD': bool(inside[i]),
        })
    df_res = pd.DataFrame(rows).sort_values('pIC50_pred', ascending=False).reset_index(drop=True)

    # ── Reports and Export ────────────────────────────────────────────────────
    print(f"\n[5/5] PERFORMANCE REPORTS AND EXPORT")
    explain("Report A: sanity check against known drugs", """
        Pyrimethamine and trimethoprim are classic DHFR inhibitors with known experimental pIC50. If the model predicts them
        close to the lab value it is behaving sensibly. Status: absolute error < 0.5 = EXCELLENT, < 1.0 = ACCEPTABLE, else ALERT.
        Caution: if these drugs (or close analogues) are in the training set, a good match is partly memorisation and not a
        true test of generalisation.
    """)
    print("\n" + "-" * 80)
    print("REPORT A: Internal Validation vs Known TgDHFR Controls")
    print("-" * 80)
    print(f"{'Drug':<18} | {'pIC50 Exp.':>12} | {'pIC50 Pred.':>12} | {'Abs Error':>10} | {'Status'}")
    print("-" * 80)
    for drug, real in KNOWN_VALUES.items():
        match = find_control(df_res, drug)
        if not match.empty:
            pred = match.iloc[0]['pIC50_pred']
            err = abs(real - pred)
            status = "EXCELLENT" if err < 0.5 else "ACCEPTABLE" if err < 1.0 else "ALERT"
            print(f"{drug:<18} | {real:>12.2f} | {pred:>12.4f} | {err:>10.4f} | {status}")
        else:
            print(f"{drug:<18} | [Excluded in previous filters]")

    explain("Report B: ranking and efficiency metrics", """
        Besides predicted potency (pIC50), the table gives metrics that judge whether the potency is 'earned' efficiently:
        LE (ligand efficiency) = potency per heavy atom, >= 0.3 is generally considered good; BEI = potency per kDa of mass;
        LLE = pIC50 minus logP, >= 5 suggests potency is not just coming from greasiness (lipophilicity); SEI = potency per
        polar surface area. A high pIC50 with poor LE/LLE usually means a big or greasy molecule that is a weaker lead.
    """)
    print("\n" + "-" * 100)
    print("REPORT B: Top 10 FDA Candidates (Sorted by Predicted Affinity)")
    print("-" * 100)
    print(f"{'Rank':<5} {'Compound Name':<30} {'pIC50':>7} {'LE':>6} {'BEI':>7} {'LLE':>6} {'SEI':>6}")
    print("-" * 100)
    for rank, (_, r) in enumerate(df_res.head(10).iterrows(), 1):
        name = str(r['Name'])[:28]
        print(f"{rank:<5} {name:<30} {r['pIC50_pred']:>7.4f} {r['LE']:>6.2f} "
              f"{r['BEI']:>7.2f} {r['LLE']:>6.2f} {r['SEI']:>6.2f}")

    # Safe export
    paper_top_mask = df_res['Name'].str.contains('|'.join(PAPER_TOP), case=False, na=False)

    top10 = df_res.head(10).copy()
    top10['Type'] = '[TOP_ML]'

    refs = df_res[paper_top_mask].copy()
    refs['Type'] = '[REFERENCE]'

    combined = pd.concat([top10, refs]).drop_duplicates('CID').sort_values('pIC50_pred', ascending=False)

    # Write to agnostic CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    os.makedirs(os.path.dirname(LATEX_SCREENING), exist_ok=True)
    values = {"ScreenNLibrary": initial_fda_count, "ScreenNStructural": len(candidates), "ScreenNDomain": int(inside.sum()),
              "ScreenNCandidates": len(combined), "ScreenNTopML": int((combined['Type'] == '[TOP_ML]').sum()),
              "ScreenNReference": int((combined['Type'] == '[REFERENCE]').sum()),
              "ScreenNTrainMols": len(df_train), "ScreenMaskBits": int(mask.sum()) if mask is not None else 2048,
              "ScreenMaxMW": f"{MAX_MW:.0f}", "ScreenADNeighbors": AD_K, "ScreenADZ": f"{AD_Z}",
              "ScreenADMinSimilarity": f"{domain['min_similarity']:.3f}",
              "ScreenADTrainMedian": f"{np.median(train_similarity):.2f}",
              "ScreenADFdaMedian": f"{np.median(fda_similarity):.2f}",
              "ScreenADFdaLow": f"{np.percentile(fda_similarity, 5):.2f}",
              "ScreenADFdaHigh": f"{np.percentile(fda_similarity, 95):.2f}",
              "ScreenADFdaMax": f"{fda_similarity.max():.2f}",
              "ScreenNCandidatesInDomain": int(combined['In_AD'].sum()),
              "ScreenADCandidatesMedian": f"{combined['AD_similarity'].median():.2f}",
              "ScreenTrainMaxMW": f"{max(Descriptors.MolWt(m) for m in train_mols):.0f}",
              "ScreenTrainNOutsideElements": sum(1 for m in train_mols if not contains_only_elements(m, ALLOWED_ATOMS))}
    for cand in ("Methotrexate", "Trimetrexate", "Pralatrexate", "Triamterene", "Pyrimethamine"):
        row = find_control(df_res, cand)
        if not row.empty:
            r0 = row.iloc[0]
            values[f"Cand{cand}Pred"] = f"{r0['pIC50_pred']:.2f}"
            values[f"Cand{cand}LE"] = f"{r0['LE']:.2f}"
            values[f"Cand{cand}BEI"] = f"{r0['BEI']:.1f}"
    antifolate_le = [find_control(df_res, n).iloc[0]['LE'] for n in ("Methotrexate", "Trimetrexate", "Pralatrexate")
                     if not find_control(df_res, n).empty]
    values["AntifolateLEMin"], values["AntifolateLEMax"] = f"{min(antifolate_le):.2f}", f"{max(antifolate_le):.2f}"
    for drug, real in KNOWN_VALUES.items():
        match = find_control(df_res, drug)
        if not match.empty:
            pred = match.iloc[0]['pIC50_pred']
            values[f"Control{drug}Exp"] = f"{real:.2f}"
            values[f"Control{drug}Pred"] = f"{pred:.2f}"
            values[f"Control{drug}Err"] = f"{abs(real - pred):.2f}"
    os.makedirs(os.path.dirname(LATEX_SCREENING), exist_ok=True)
    with open(LATEX_SCREENING, "w", encoding="utf-8") as f:
        f.write(format_macros(values, header="Auto-generated by 05_virtual_screening.py"))

    print("\n" + "=" * 100)
    print(f"[SUCCESS] {len(combined)} candidates safely exported to:")
    print(f"        -> {OUTPUT_CSV}")
    print(f"        Logs saved in: {LOG_FILE}")
    print(f"        Total execution time: {time.time() - start_pipeline:.2f}s")
    print("=" * 100)


if __name__ == "__main__":
    run()