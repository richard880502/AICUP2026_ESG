"""Plot promise vs non-promise similarity distribution for baseline and exp2d."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    with open(RESULTS_DIR / "promise_sim.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    colors = {"E": "#2ecc71", "S": "#3498db", "G": "#e67e22"}

    for ax, entry in zip(axes, data):
        sims = np.array(entry["sims"])
        esg_types = entry["esg_types"]

        for t in ["E", "S", "G"]:
            idx = [i for i, e in enumerate(esg_types) if e == t]
            ax.hist(
                sims[idx],
                bins=40,
                alpha=0.6,
                color=colors[t],
                label=f"{t} (n={len(idx)}, μ={np.mean(sims[idx]):.3f})",
                density=True,
            )

        ax.axvline(entry["sim_mean"], color="black", linestyle="--", linewidth=1.5,
                   label=f"overall μ={entry['sim_mean']:.3f}")
        ax.set_title(entry["model"], fontsize=14)
        ax.set_xlabel("Cosine Similarity (promise vs non-promise)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.set_xlim(-0.2, 1.05)

    fig.suptitle("Promise vs Non-Promise Similarity Distribution", fontsize=15, y=1.02)
    plt.tight_layout()

    out = RESULTS_DIR / "promise_sim_dist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
