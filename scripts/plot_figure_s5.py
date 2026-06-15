"""Figure S5: NK strategy look-up tables at higher DE iteration counts.

Same construction as Figure 5A (optimal base chance / splitting vs rho_NK, mean +/- std of the
per-(N,K) optimum within each rho bin), repeated for M = 25 (= Fig 5A), 50 and 100 generations.
Reads the nk_strategy_grid* raw products produced by generate_figure5_raw_data.py.

Run from the repository root::

    python scripts/plot_figure_s5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from slide.utils import get_figures_dir, get_raw_data_dir, load_pickle

# M (DE generations) -> raw-product key. M=25 is the Figure 5A grid.
PANELS = [(25, "nk_strategy_grid"), (50, "nk_strategy_grid_M50"), (100, "nk_strategy_grid_M100")]
SAVE_TYPES = ("pdf", "png", "eps")


def lookup_from_grid(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a (pairs, split, base, reps) grid to per-rho-bin optimal base/split (mean +/- std)."""
    data = np.asarray(payload["data"], dtype=float)
    params = payload["params"]
    base_chances = np.asarray(params["base_chances"], dtype=float)
    splits = np.asarray(params["splits"], dtype=int)
    nk_pairs = np.asarray(params["nk_pairs"], dtype=int)
    rho = (nk_pairs[:, 1] + 1) / nk_pairs[:, 0]

    opt_split, opt_base = [], []
    for item in data:
        space = item.mean(axis=-1)  # (split, base), averaged over reps
        row, column = np.unravel_index(np.nanargmax(space), space.shape)
        opt_split.append(int(splits[row]))
        opt_base.append(float(base_chances[column]))
    opt_split = np.asarray(opt_split, dtype=float)
    opt_base = np.asarray(opt_base, dtype=float)

    rounded = np.round(rho, 1)
    grouped = np.unique(rounded)
    bc_mean, bc_std, sp_mean, sp_std = [], [], [], []
    for value in grouped:
        mask = rounded == value
        bc_mean.append(opt_base[mask].mean())
        bc_std.append(opt_base[mask].std())
        sp_mean.append(opt_split[mask].mean())
        sp_std.append(opt_split[mask].std())
    return grouped, np.asarray(bc_mean), np.asarray(bc_std), np.asarray(sp_mean), np.asarray(sp_std)


def main() -> None:
    """Build and save the three-panel Figure S5 look-up tables."""
    raw_dir = get_raw_data_dir()
    figures_dir = get_figures_dir()
    color_base, color_split = "tab:orange", "tab:blue"

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), dpi=300)
    for ax, (m, key), letter in zip(axes, PANELS, "ABC"):
        payload = load_pickle(raw_dir / f"{key}_raw_data.pkl")
        rho, bc_m, bc_s, sp_m, sp_s = lookup_from_grid(payload)
        ax2 = ax.twinx()
        ax.errorbar(rho, bc_m, yerr=bc_s, fmt="o--", capsize=3, color=color_base, label="Optimal base chance")
        ax2.errorbar(rho, sp_m, yerr=sp_s, fmt="o--", capsize=3, color=color_split, label="Optimal splitting")
        ax.set_xlabel(r"$\rho_{NK}$", fontsize=9)
        ax.set_xlim(0.0, 1.05)
        ax.set_title(f"{letter}: M = {m} generations", fontsize=10)
        ax.tick_params(axis="y", labelcolor=color_base)
        ax2.tick_params(axis="y", labelcolor=color_split)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        if ax is axes[0]:
            ax.set_ylabel(r"Predicted base chance $b$", color=color_base, fontsize=9)
        if ax is axes[-1]:
            ax2.set_ylabel("Predicted splitting", color=color_split, fontsize=9)
        # shared y-limits so panels are comparable
        ax.set_ylim(-0.02, 0.21)
        ax2.set_ylim(0, 26)

    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color=color_base, marker="o", ls="--", label="Optimal base chance"),
        mlines.Line2D([], [], color=color_split, marker="o", ls="--", label="Optimal splitting"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("SLIDE NK strategy look-up vs DE horizon", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    for ext in SAVE_TYPES:
        out = figures_dir / ext / f"figure_S5.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
