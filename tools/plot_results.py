"""Create a static Matplotlib summary of saved episode aggregates."""
import argparse
import csv
from pathlib import Path
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", default=Path("results/episodes.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/episodes.png"))
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/neuroworm-matplotlib")))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with args.csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("CSV has no episode rows")
    x = np.arange(1, len(rows) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for axis, key, label in zip(axes, ("survival_duration_s", "food_eaten", "average_firing_rate_hz"),
                               ("Duration (s)", "Food eaten", "Mean firing rate (Hz/neuron)")):
        axis.bar(x, [float(row[key]) for row in rows], color="#2b9d89")
        axis.set(xlabel="Recorded episode", ylabel=label)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("NeuroWorm episode aggregates (includes session-end partial episodes)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
