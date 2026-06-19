"""
Run the four portfolio-strengthening analyses, then build REPORT.md.

Order:
  1. annotate_genes.py        map CpGs to genes (needs the 450K manifest)
  2. enrichment.py            gene-set enrichment (needs internet, Enrichr)
  3. benchmark_hannum.py      head-to-head vs the published Hannum clock
  4. external_validation.py   apply the clock to a second cohort
  5. make_report.py           assemble REPORT.md from whatever exists

Each step is independent: if one fails (for example enrichment with no internet),
the rest still run and the report fills in only what was computed. Run the core
pipeline (run_all.py) first so the trained model and metrics exist.
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")

STEPS = [
    "annotate_genes.py",
    "enrichment.py",
    "benchmark_hannum.py",
    "external_validation.py",
    "make_report.py",
]


def main():
    only = sys.argv[1:]  # optionally pass specific step names to run a subset
    for step in STEPS:
        if only and step not in only:
            continue
        print("\n=== {} ===".format(step))
        try:
            runpy.run_path(os.path.join(SRC, step), run_name="__main__")
        except SystemExit as e:
            print("[run_extras] {} stopped: {}".format(step, e))
        except Exception as e:
            print("[run_extras] {} failed: {}".format(step, repr(e)[:200]))
    print("\n[run_extras] done. See results/ and REPORT.md")


if __name__ == "__main__":
    main()
