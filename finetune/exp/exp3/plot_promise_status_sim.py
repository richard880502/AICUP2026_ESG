"""Plot promise_status Yes/No pairwise similarity distributions."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    with open(RESULTS_DIR / "promise_status_sim.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    colors = {"yes_vs_yes": "#3498db", "no_vs_no": "#e74c3c", "yes_vs_no": "#95a5a6", "any_vs_any": "#2ecc71"}
    labels = {"yes_vs_yes": "Yes vs Yes", "no_vs_no": "No vs No", "yes_vs_no": "Yes vs No", "any_vs_any": "Any vs Any"}

    for ax, entry in zip(axes, data):
        for key in ["yes_vs_yes", "no_vs_no", "yes_vs_no", "any_vs_any"]:
            sims = np.array(entry[key]["sims"])
            mean = entry[key]["mean"]
            ax.hist(
                sims,
                bins=60,
                alpha=0.5,
                color=colors[key],
                label=f"{labels[key]} (μ={mean:.3f})",
                density=True,
            )
            ax.axvline(mean, color=colors[key], linestyle="--", linewidth=1.2)

        ax.set_title(entry["model"], fontsize=14)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle("Promise Status (Yes/No) Pairwise Similarity Distribution", fontsize=14, y=1.01)
    plt.tight_layout()

    out = RESULTS_DIR / "promise_status_sim_dist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
