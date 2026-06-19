"""
Pipeline runner for the aging clock. This is the single entry point.

It runs the three stages in order:
  1. download_data.py  fetch and prepare the methylation matrix
  2. train_clock.py    train the model and write metrics
  3. plots.py          draw the result figures

Run it with:  python run_all.py
The download step is skipped automatically if the prepared data already exists,
so reruns (for example after changing the model) are fast.
"""

import os
import runpy
import sys

# Resolve paths relative to this file so the runner works no matter where it is
# called from (a scheduled task, a different drive, and so on).
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
DATA = os.path.join(HERE, "data", "methylation_top.pkl")  # the prepared matrix


def run(stage):
    """Execute one stage script as if it were run directly with python."""
    path = os.path.join(SRC, stage)
    print(f"\n=== {stage} ===")
    # run_name="__main__" makes each stage's own `if __name__ == "__main__"`
    # block fire, so the scripts also work when run on their own.
    runpy.run_path(path, run_name="__main__")


def main():
    # Skip the slow download and parse step if the prepared data is already on
    # disk. Pass --force-download on the command line to refetch from scratch.
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
