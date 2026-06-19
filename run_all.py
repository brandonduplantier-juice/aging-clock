"""
Run the full aging-clock pipeline: download, train, plot.

Skips the download if data/methylation_top.pkl already exists, so reruns are fast.
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
DATA = os.path.join(HERE, "data", "methylation_top.pkl")


def run(stage):
    path = os.path.join(SRC, stage)
    print(f"\n=== {stage} ===")
    runpy.run_path(path, run_name="__main__")


def main():
    if os.path.exists(DATA) and "--force-download" not in sys.argv:
        print("[run_all] data/methylation_top.pkl exists, skipping download "
              "(pass --force-download to refetch)")
    else:
        run("download_data.py")
    run("train_clock.py")
    run("plots.py")
    print("\n[run_all] done. See results/ for metrics.json and the plots.")


if __name__ == "__main__":
    main()
