"""Prepare the MoleculeACE benchmark targets for the generalization section.

MoleculeACE (van Tilborg, Alenicheva and Grisoni, J. Chem. Inf. Model. 2022, 62, 5938-5951) distributes 30 curated
ChEMBL bioactivity sets, each for one protein target, with a predefined 80/20 split and a flag marking the compounds
that belong to an activity cliff. This step converts the subset of them that falls in the low-data regime of this
study into the four-column format the pipeline reads, so that every benchmark step can be run on them unchanged
through MLGNN_TRAIN_FILE.

The archive is downloaded once into its own directory and only its CSV files are read; nothing from it is executed.
The MoleculeACE repository is MIT-licensed and its data derive from ChEMBL, so both are cited in the manuscript.
"""
import os
import sys
import io
import tarfile
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths_config import *  # noqa: F401,F403
from logger_utils import setup_logger, explain

SOURCE_URL = "https://github.com/molML/MoleculeACE/archive/refs/heads/main.tar.gz"
SOURCE_DIR = os.path.join(DATA_DIR, "moleculeace", "_source")
OUT_DIR = os.path.join(DATA_DIR, "moleculeace")
INDEX_CSV = os.path.join(OUT_DIR, "index.csv")
# The generalization cohort: every target whose size falls in the regime of the main benchmark.
MAX_COMPOUNDS = int(os.environ.get("MLGNN_MACE_MAX_N", "1000"))


def download_source():
    """Fetch the archive once and extract only the benchmark CSV files into their own directory."""
    marker = os.path.join(SOURCE_DIR, "benchmark_data")
    if os.path.isdir(marker) and os.listdir(marker):
        print(f"[CACHE] Source already present in {marker}", flush=True)
        return marker
    os.makedirs(SOURCE_DIR, exist_ok=True)
    print(f"[DOWNLOAD] {SOURCE_URL}", flush=True)
    with urllib.request.urlopen(SOURCE_URL, timeout=300) as response:
        payload = response.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        wanted = [m for m in archive.getmembers()
                  if m.isfile() and "/Data/benchmark_data/" in m.name and m.name.endswith(".csv")]
        os.makedirs(marker, exist_ok=True)
        for member in wanted:
            target = os.path.join(marker, os.path.relpath(member.name, member.name.split("benchmark_data/")[0]
                                                          + "benchmark_data"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with archive.extractfile(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    print(f"[DOWNLOAD] {len(wanted)} files extracted to {marker}", flush=True)
    return marker


def main():
    setup_logger(os.path.join(LOGS_DIR, "20_moleculeace_prepare.log"))
    print("=" * 105)
    print("20_moleculeace_prepare.py — MoleculeACE targets for the generalization benchmark")
    print("=" * 105)
    explain("What this step does", f"""
        Converts the MoleculeACE bioactivity sets with at most {MAX_COMPOUNDS} compounds into the format the pipeline
        reads (cid, Smiles, Standard Value, pIC50 Value), one CSV per target. Their potency column is already the
        negative logarithm of a molar concentration (pKi or pEC50), the same scale as the pIC50 of the main benchmark,
        so no transformation is applied and no value is recomputed.

        The predefined 80/20 split and the activity-cliff flag of each compound are written next to each data set, so
        that the benchmark can be scored both on our own nested cross-validation and on the partition the original
        authors published, and so that the error on cliff compounds can be reported separately.
    """)
    source = download_source()
    metadata_path = os.path.join(source, "metadata", "datasets.csv")
    if not os.path.exists(metadata_path):
        print(f"[ERROR] Metadata not found at {metadata_path}")
        sys.exit(1)
    metadata = pd.read_csv(metadata_path).set_index("Dataset")

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for name in sorted(os.listdir(source)):
        if not name.endswith(".csv"):
            continue
        dataset = name[:-4]
        source_df = pd.read_csv(os.path.join(source, name))
        if len(source_df) > MAX_COMPOUNDS:
            continue
        if dataset not in metadata.index:
            print(f"  [WARNING] {dataset} has no metadata entry; skipped.")
            continue
        meta = metadata.loc[dataset]

        converted = pd.DataFrame({
            "cid": range(len(source_df)),
            "Smiles": source_df["smiles"],
            "Standard Value": source_df["exp_mean [nM]"],
            "pIC50 Value": source_df["y [pEC50/pKi]"],
        })
        if converted["Smiles"].duplicated().any():
            n_duplicated = int(converted["Smiles"].duplicated().sum())
            print(f"  [WARNING] {dataset}: {n_duplicated} duplicated SMILES in the source file.")
        converted.to_csv(os.path.join(OUT_DIR, f"{dataset}.csv"), index=False)

        # The published partition and the cliff flags travel beside the data, not inside it, so that the pipeline
        # sees exactly the four columns it expects.
        source_df[["smiles", "split", "cliff_mol"]].rename(columns={"smiles": "Smiles"}).to_csv(
            os.path.join(OUT_DIR, f"{dataset}_split.csv"), index=False)

        rows.append(dict(dataset=dataset, chembl_id=meta["ChEMBL ID"], target=meta["Target name"],
                         abbreviation=meta["Abbriviation"], endpoint=meta["Type"],
                         receptor_class=meta["Receptor Class"], n=len(converted),
                         cliff_pct=float(meta["Percentage cliffs"]),
                         pic50_min=round(float(converted["pIC50 Value"].min()), 2),
                         pic50_max=round(float(converted["pIC50 Value"].max()), 2)))
        print(f"  {dataset:<18} {meta['Abbriviation']:<8} n={len(converted):<5} "
              f"{meta['Type']:<5} {meta['Receptor Class']:<12} cliffs={meta['Percentage cliffs']}%", flush=True)

    if not rows:
        print(f"[ERROR] No data set with at most {MAX_COMPOUNDS} compounds was found.")
        sys.exit(1)
    index = pd.DataFrame(rows).sort_values("n").reset_index(drop=True)
    index.to_csv(INDEX_CSV, index=False)
    print(f"\n[EXPORT] {len(index)} targets ({index['n'].sum()} compounds) written to {OUT_DIR}")
    print(f"[EXPORT] Index saved to {INDEX_CSV}")


if __name__ == "__main__":
    main()
