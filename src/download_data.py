"""
Build the GSE40279 methylation matrix by streaming the already-downloaded GEO
family SOFT file, instead of using GEOparse.pivot_samples.

Why this rewrite: pivot_samples assembles an object-dtype array of about 310
million cells (roughly 2.3 GB) and crashes with a MemoryError. Streaming the SOFT
file line by line and storing each sample's values as float32 keeps peak memory
to a little over the size of the final matrix, and it reuses the file already on
disk, so there is no second large download.

Reads:
  data/raw/GSE40279_family.soft.gz   (already downloaded by the earlier run)

Outputs:
  data/methylation_top.pkl  samples x N_PROBES beta values (float32)
  data/meta.csv             sample, age, sex

This is slower than a C-based parser because it walks every line in Python, so
expect a few minutes. If it is ever too slow, the alternative is to download the
smaller series-matrix file and read it with pandas, but that means another
download. Memory safety and reusing the existing file win here.
"""

import gzip
import os
import re
import sys

import numpy as np
import pandas as pd

# Number of CpG probes to keep. We rank probes by how much they vary across
# people and keep the most variable ones. Lower is faster; higher can be slightly
# more accurate.
N_PROBES = 20000

# Paths, resolved relative to this file so the script runs from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
SOFT_PATH = os.path.join(DATA_DIR, "raw", "GSE40279_family.soft.gz")

# The SOFT file stores each person's metadata as free text like "age (y): 67".
# We search that text for an age rather than assuming a fixed label, so if the
# label ever changes only these patterns need updating.
AGE_PATTERNS = [
    re.compile(r"age\s*\(y\)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\bage\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
]
SEX_PATTERN = re.compile(r"\b(gender|sex)\b\s*[:=]\s*([a-z]+)", re.IGNORECASE)


def parse_age(text):
    """Return the first age found in a characteristics string, or None."""
    for pat in AGE_PATTERNS:
        m = pat.search(text)
        if m:
            return float(m.group(1))
    return None


def parse_sex(text):
    """Return the sex found in a characteristics string, or None."""
    m = SEX_PATTERN.search(text)
    if m:
        return m.group(2).strip().lower()
    return None


def main():
    if not os.path.exists(SOFT_PATH):
        sys.exit(f"SOFT file not found at {SOFT_PATH}. The earlier run should "
                 f"have downloaded it; if not, rerun the original download once.")

    print(f"[download] streaming {os.path.basename(SOFT_PATH)} (no re-download, takes a few minutes)")

    # Accumulators filled as we walk the file.
    master_probes = None   # the probe (CpG) ID order, taken from the first sample
    sample_names = []      # GSM id per sample, in file order
    ages = {}              # GSM id -> age
    sexes = {}             # GSM id -> sex
    columns = []           # one float32 array of beta values per sample

    # Parser state. The SOFT file is one long text stream, so we track where we
    # are: which sample we are on, and whether we are inside its data table.
    cur = None             # current GSM id
    in_table = False       # are we between !sample_table_begin and !sample_table_end
    header_seen = False    # have we skipped the table's column-header line yet
    probe_buf = []         # probe ids collected for the current sample
    value_buf = []         # beta values (as strings) for the current sample

    def flush_sample():
        """Finalize the current sample: convert its values to float32 and store."""
        nonlocal master_probes
        if cur is None or not value_buf:
            return
        # Convert the whole column in one vectorized call (fast); anything that
        # is not a number becomes NaN instead of crashing.
        vals = pd.to_numeric(value_buf, errors="coerce").astype(np.float32)
        if master_probes is None:
            master_probes = list(probe_buf)  # first sample defines the probe order
        elif len(probe_buf) != len(master_probes):
            # Every sample on one array shares the same probes in the same order.
            # If a count differs, stop rather than silently misalign the data.
            sys.exit(f"sample {cur} has {len(probe_buf)} probes, expected "
                     f"{len(master_probes)}; probe sets differ, cannot align.")
        sample_names.append(cur)
        columns.append(np.asarray(vals))

    # Walk the gzipped file one line at a time. This is the memory-safe core: we
    # never hold the whole decompressed file in memory at once.
    with gzip.open(SOFT_PATH, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("^SAMPLE"):
                # A new sample block starts here. Reset the per-sample state.
                cur = line.split("=", 1)[1].strip()
                in_table = False
                header_seen = False
                probe_buf = []
                value_buf = []
                if len(sample_names) and len(sample_names) % 100 == 0:
                    print(f"[download]   parsed {len(sample_names)} samples so far")
                continue
            if line.startswith("!Sample_characteristics") and cur is not None:
                # A metadata line for the current sample. Pull age and sex if present.
                content = line.split("=", 1)[1].strip() if "=" in line else ""
                if cur not in ages:
                    a = parse_age(content)
                    if a is not None:
                        ages[cur] = a
                if cur not in sexes:
                    s = parse_sex(content)
                    if s is not None:
                        sexes[cur] = s
                continue
            if line.startswith("!sample_table_begin"):
                in_table = True       # the beta-value rows start after this marker
                header_seen = False
                probe_buf = []
                value_buf = []
                continue
            if line.startswith("!sample_table_end"):
                in_table = False
                flush_sample()        # this sample's table is complete
                continue
            if in_table:
                if not header_seen:
                    header_seen = True   # first line in the table is the column header, skip it
                    continue
                # A data row: probe id in column 0, beta value in column 1.
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                probe_buf.append(parts[0])
                value_buf.append(parts[1])

    if not columns:
        sys.exit("No sample tables parsed. The SOFT file may be malformed.")

    print(f"[download] parsed {len(columns)} samples, {len(master_probes)} probes")

    # Assemble the full matrix as float32 (rows = samples, columns = probes). We
    # copy each sample's column into a preallocated array and free the source as
    # we go, which keeps peak memory near the size of one matrix rather than two.
    mat = np.empty((len(columns), len(master_probes)), dtype=np.float32)
    for j, c in enumerate(columns):
        mat[j, :] = c
        columns[j] = None
    del columns

    # Drop any probe missing in some sample, then keep only the most-variable
    # probes. Ranking by variance is unsupervised: it never looks at age, so it
    # cannot leak the label into the test set.
    valid = ~np.isnan(mat).any(axis=0)
    mat = mat[:, valid]
    probes = [p for p, keep in zip(master_probes, valid) if keep]
    print(f"[download] {mat.shape[0]} samples, {mat.shape[1]} complete probes")

    variances = mat.var(axis=0)
    top_idx = np.argsort(variances)[::-1][:N_PROBES]  # indices of the highest-variance probes
    top_idx.sort()                                    # restore original column order
    mat = mat[:, top_idx]
    probes = [probes[i] for i in top_idx]
    print(f"[download] reduced to top {mat.shape[1]} variable probes")

    # Build the metadata table aligned to the samples we parsed.
    meta = pd.DataFrame({
        "sample": sample_names,
        "age": [ages.get(s) for s in sample_names],
        "sex": [sexes.get(s, "unknown") for s in sample_names],
    }).set_index("sample")

    # If we could not read any ages, stop loudly rather than train on nothing.
    missing = int(meta["age"].isna().sum())
    if missing == len(meta):
        sys.exit("Could not parse any ages from the SOFT characteristics. "
                 "Adjust AGE_PATTERNS and rerun.")
    if missing:
        print(f"[download] WARNING: {missing} samples had no parseable age; dropping them")

    # Final matrix as a labeled DataFrame, restricted to samples that have an age.
    betas = pd.DataFrame(mat, index=sample_names, columns=probes)
    keep = meta.dropna(subset=["age"]).index
    betas = betas.loc[keep]
    meta = meta.loc[keep]

    os.makedirs(DATA_DIR, exist_ok=True)
    betas.to_pickle(os.path.join(DATA_DIR, "methylation_top.pkl"))
    meta.to_csv(os.path.join(DATA_DIR, "meta.csv"))
    print(f"[download] wrote data/methylation_top.pkl ({betas.shape[0]} x {betas.shape[1]}) and data/meta.csv")


if __name__ == "__main__":
    main()
