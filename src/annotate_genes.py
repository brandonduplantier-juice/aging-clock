"""
Map the clock's selected CpG sites to genes using the Illumina 450K manifest.

The trained elastic net keeps a sparse set of CpG sites (cg IDs). On their own
those IDs say nothing biological. This script attaches the gene each site sits in
or near, so the enrichment step and the dashboard can speak in gene names.

Manifest source:
  The official Illumina 450K manifest is hosted on GEO as a gzipped CSV
  supplementary file on platform GPL13534 (about 50 MB, far smaller than the full
  platform SOFT file). This script downloads it once, caches a slim two-column
  cg -> gene map under data/raw, and reuses that on later runs. If the download
  fails it falls back to GEOparse. You can also point it at a manifest you already
  have with the AGING_CLOCK_MANIFEST environment variable.

Reads:
  data/methylation_top.pkl   to recover the CpG (probe) order
  results/clock.joblib       the fitted model (nonzero coefficients = selected sites)

Writes:
  results/gene_annotation.csv   one row per selected CpG: cg, coef, impact, genes, primary_gene
"""

import gzip
import io
import os
import re
import sys
import urllib.request

import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RESULTS_DIR = os.path.join(ROOT, "results")

GENE_COL_CANDIDATES = ["UCSC_RefGene_Name", "UCSC_REFGENE_NAME", "Gene_Symbol", "gene", "GENE"]
SUPPL_DIR = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13534/suppl/"
MANIFEST_CANDIDATES = [
    "GPL13534_HumanMethylation450_15017482_v.1.2.csv.gz",
    "GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz",
]
CACHE = os.path.join(RAW_DIR, "gene_map_450k.csv")
GZ_PATH = os.path.join(RAW_DIR, "manifest_450k.csv.gz")


def selected_sites():
    """Return a DataFrame of the CpGs the model actually uses (nonzero coef)."""
    x = pd.read_pickle(os.path.join(DATA_DIR, "methylation_top.pkl"))
    probes = list(x.columns)
    bundle = joblib.load(os.path.join(RESULTS_DIR, "clock.joblib"))
    coef = np.asarray(bundle["model"].coef_, dtype=float).ravel()
    scale = np.asarray(bundle["scaler"].scale_, dtype=float).ravel()
    rows = []
    for i, c in enumerate(coef):
        if c != 0.0:
            sc = scale[i] if scale[i] != 0 else 1.0
            rows.append({"cg": str(probes[i]), "coef": float(c), "impact": float(c / sc)})
    df = pd.DataFrame(rows)
    return df.reindex(df["impact"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def _discover_manifest_url():
    """List the GEO suppl directory and return the newest manifest csv.gz URL."""
    try:
        with urllib.request.urlopen(SUPPL_DIR, timeout=90) as r:
            html = r.read().decode("latin-1", "ignore")
        names = re.findall(r"GPL13534_HumanMethylation450[^\"'>]*?\.csv\.gz", html)
        if names:
            return SUPPL_DIR + sorted(set(names))[-1]
    except Exception as e:
        print("[annotate] could not list suppl directory: {}".format(repr(e)[:120]))
    return None


def _geoparse_fallback():
    print("[annotate] falling back to GEOparse platform download (this is large and slow)")
    import GEOparse
    gpl = GEOparse.get_GEO(geo="GPL13534", destdir=RAW_DIR, silent=True)
    table = gpl.table
    if "ID" in table.columns:
        table = table.set_index("ID")
    gene_col = next((c for c in GENE_COL_CANDIDATES if c in table.columns), None)
    if gene_col is None:
        print("[annotate] columns:", list(table.columns))
        sys.exit("No gene column found in GPL13534 table.")
    return table[gene_col]


def _parse_manifest(path):
    """Parse the Illumina manifest CSV (with its preamble and [Controls] block)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="latin-1", errors="ignore") as fh:
        lines = fh.readlines()
    hdr = next((i for i, l in enumerate(lines) if l.startswith("IlmnID,")), None)
    if hdr is None:
        sys.exit("Could not find the IlmnID header in the manifest at {}.".format(path))
    ctrl = next((i for i in range(hdr + 1, len(lines)) if lines[i].startswith("[Controls]")), len(lines))
    df = pd.read_csv(io.StringIO("".join(lines[hdr:ctrl])), low_memory=False)
    gene_col = next((c for c in GENE_COL_CANDIDATES if c in df.columns), None)
    if gene_col is None:
        print("[annotate] columns found:", list(df.columns))
        sys.exit("No gene column found in the manifest. Add it to GENE_COL_CANDIDATES.")
    idcol = "IlmnID" if "IlmnID" in df.columns else df.columns[0]
    return df.set_index(idcol)[gene_col]


def load_manifest():
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(CACHE):
        print("[annotate] using cached gene map ({})".format(CACHE))
        return pd.read_csv(CACHE, index_col=0).iloc[:, 0]

    env = os.environ.get("AGING_CLOCK_MANIFEST")
    src = None
    if env and os.path.exists(env):
        print("[annotate] using manifest from AGING_CLOCK_MANIFEST: {}".format(env))
        src = env
    elif os.path.exists(GZ_PATH):
        src = GZ_PATH
    else:
        urls = []
        disc = _discover_manifest_url()
        if disc:
            urls.append(disc)
        urls += [SUPPL_DIR + c for c in MANIFEST_CANDIDATES]
        for u in urls:
            try:
                print("[annotate] downloading manifest: {}".format(u))
                urllib.request.urlretrieve(u, GZ_PATH)
                src = GZ_PATH
                break
            except Exception as e:
                print("[annotate]   failed: {}".format(repr(e)[:120]))
        if src is None:
            return _geoparse_fallback()

    gmap = _parse_manifest(src)
    gmap.to_frame("gene").to_csv(CACHE)
    print("[annotate] parsed manifest: {} probes, cached slim map".format(len(gmap)))
    return gmap


def clean_genes(raw):
    """Turn 'ELOVL2;ELOVL2;FAM150B' into a de-duplicated list, order kept."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    parts = [p.strip() for p in str(raw).split(";") if p.strip()]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main():
    sites = selected_sites()
    print("[annotate] selected CpGs: {}".format(len(sites)))
    gene_map = load_manifest()
    index_set = set(gene_map.index)

    genes_col, primary_col = [], []
    for cg in sites["cg"]:
        genes = clean_genes(gene_map.get(cg)) if cg in index_set else []
        genes_col.append(";".join(genes))
        primary_col.append(genes[0] if genes else "")

    sites["genes"] = genes_col
    sites["primary_gene"] = primary_col

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sites.to_csv(os.path.join(RESULTS_DIR, "gene_annotation.csv"), index=False)

    n_with_gene = int((sites["primary_gene"] != "").sum())
    print("[annotate] {} of {} sites mapped to a gene".format(n_with_gene, len(sites)))
    top = sites[sites["primary_gene"] != ""].head(10)
    if len(top):
        print("[annotate] top sites by impact and their genes:")
        for _, r in top.iterrows():
            print("    {}  {:+.2f} yr  {}".format(r["cg"], r["impact"], r["primary_gene"]))
    if "ELOVL2" in set(sites["primary_gene"]):
        print("[annotate] note: ELOVL2 (the canonical age-methylation gene) is in the signature.")
    print("[annotate] wrote results/gene_annotation.csv")


if __name__ == "__main__":
    main()
