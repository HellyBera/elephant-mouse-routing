
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Locate the project root regardless of where
# the script is executed.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game_theory import (
    calculate_game,
    find_nash_equilibria,
    find_social_optimum
)

DATA = ROOT / "results" / "data"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False
})

COLORS = {
    "selfish": "#C65D57",
    "team": "#3688B8",
    "tolled": "#38A17A"
}


def save_figure(name):
    plt.tight_layout()

    for extension in ("png", "pdf"):
        plt.savefig(
            FIGURES / f"{name}.{extension}",
            bbox_inches="tight"
        )

    plt.close()
    print(f"Saved figure: {name}")


def load_csv(name, required_columns):
    path = DATA / name

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding experiment."
        )

    df = pd.read_csv(path)

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}"
        )

    return df


# ----------------------------------------------------
# FIGURE 1: Nash equilibrium vs social optimum
# ----------------------------------------------------

def plot_stage_game():
    rows = []

    for beta in (0.10, 0.25):
        game = calculate_game(beta)
        equilibria = find_nash_equilibria(game)
        _, optimum = find_social_optimum(game)

        equilibrium_costs = [
            game[a]["social_cost"] for a in equilibria
        ]

        rows.append({
            "background_load": beta,
            "number_of_ne": len(equilibria),
            "social_optimum": optimum,
            "best_ne": min(equilibrium_costs),
            "worst_ne": max(equilibrium_costs),
            "poa": max(equilibrium_costs) / optimum,
            "pos": min(equilibrium_costs) / optimum
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "stage_game_summary.csv", index=False)

    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(7, 4.5))

    for offset, column, label in [
        (-width, "social_optimum", "Social optimum"),
        (0, "best_ne", "Best Nash equilibrium"),
        (width, "worst_ne", "Worst Nash equilibrium")
    ]:
        plt.bar(
            x + offset,
            df[column],
            width,
            label=label
        )

    plt.xticks(x, ["10% load", "25% load"])
    plt.ylabel("Social cost (lower is better)")
    plt.title("Stage-game equilibrium inefficiency")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.2)

    save_figure("fig1_stage_game")


# ----------------------------------------------------
# FIGURE 2: Multi-agent learning curves
# ----------------------------------------------------

def plot_learning_curves():
    df = load_csv(
        "cooperation_training.csv",
        ["mode", "seed", "episode", "mean_social_cost"]
    )

    plt.figure(figsize=(7, 4.5))

    for mode in ("selfish", "team", "tolled"):
        subset = df[df["mode"] == mode].copy()

        if subset.empty:
            raise ValueError(f"No training data for {mode}")

        smoothed = []

        for seed, group in subset.groupby("seed"):
            group = group.sort_values("episode").copy()

            group["smoothed_cost"] = (
                group["mean_social_cost"]
                .rolling(window=100, min_periods=1)
                .mean()
            )

            smoothed.append(
                group[["episode", "smoothed_cost"]]
            )

        combined = pd.concat(smoothed)

        curve = (
            combined.groupby("episode")["smoothed_cost"]
            .agg(["mean", "std"])
            .reset_index()
        )

        x = curve["episode"].to_numpy()
        y = curve["mean"].to_numpy()
        sd = curve["std"].fillna(0).to_numpy()

        plt.plot(
            x, y,
            label=mode.capitalize(),
            color=COLORS[mode]
        )

        plt.fill_between(
            x, y - sd, y + sd,
            color=COLORS[mode],
            alpha=0.12
        )

    plt.xlabel("Training episode")
    plt.ylabel("Mean social cost")
    plt.title("Multi-agent learning curves")
    plt.legend()
    plt.grid(alpha=0.2)

    save_figure("fig2_learning_curves")


# ----------------------------------------------------
# FIGURES 3, 4, 5: Cooperation comparison
# ----------------------------------------------------

def plot_cooperation_comparison():
    df = load_csv(
        "cooperation_evaluation.csv",
        [
            "mode", "seed", "mean_social_cost",
            "mean_efficiency_ratio", "switch_rate",
            "mean_stage_regret",
            "mean_tolled_stage_regret"
        ]
    )

    metrics = [
        "mean_social_cost",
        "mean_efficiency_ratio",
        "switch_rate",
        "mean_stage_regret",
        "mean_tolled_stage_regret"
    ]

    summary = df.groupby("mode")[metrics].agg(
        ["mean", "std"]
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary.to_csv(TABLES / "cooperation_summary.csv")

    modes = ["selfish", "team", "tolled"]

    figures_to_make = [
        (
            "mean_social_cost",
            "Mean network social cost",
            "fig3_social_cost"
        ),
        (
            "mean_efficiency_ratio",
            "Social efficiency ratio",
            "fig4_efficiency"
        ),
        (
            "switch_rate",
            "Route-switching rate",
            "fig5_switching"
        )
    ]

    for metric, title, filename in figures_to_make:
        means = [
            summary.loc[m, f"{metric}_mean"]
            for m in modes
        ]

        stds = [
            summary.loc[m, f"{metric}_std"]
            for m in modes
        ]

        plt.figure(figsize=(6.5, 4.5))

        bars = plt.bar(
            [m.capitalize() for m in modes],
            means,
            yerr=stds,
            capsize=6,
            color=[COLORS[m] for m in modes]
        )

        for bar, value in zip(bars, means):
            plt.annotate(
                f"{value:.4f}",
                (
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height()
                ),
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                fontsize=9
            )

        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.title(title)
        plt.grid(axis="y", alpha=0.2)

        # Give error bars and annotations room.
        plt.ylim(0, max(
            m + s for m, s in zip(means, stds)
        ) * 1.3)

        save_figure(filename)


# ----------------------------------------------------
# FIGURE 6: Punishment incentive thresholds
# ----------------------------------------------------

def plot_punishment():
    df = load_csv(
        "minmax_punishment_results.csv",
        [
            "beta", "cooperation", "agent",
            "overall_gamma_required",
            "incentive_feasible"
        ]
    )

    rows = []

    for beta, group in df.groupby("beta"):
        # All four agents appear for every cooperation
        # profile; retain unique overall thresholds.
        thresholds = (
            group["overall_gamma_required"]
            .dropna()
            .unique()
        )

        if len(thresholds) == 0:
            threshold = np.nan
        else:
            threshold = float(np.min(thresholds))

        rows.append({
            "background_load": beta,
            "pure_minmax_gamma": threshold
        })

    summary = pd.DataFrame(rows).sort_values(
        "background_load"
    )

    summary.to_csv(
        TABLES / "punishment_summary.csv",
        index=False
    )

    plt.figure(figsize=(6.5, 4.5))

    x = np.arange(len(summary))
    thresholds = summary["pure_minmax_gamma"].to_numpy()

    plt.bar(
        x, thresholds,
        color="#3688B8",
        width=0.5,
        label="Pure minmax incentive threshold"
    )

    plt.axhline(
        0.9,
        color="#C65D57",
        linestyle="--",
        label="Model discount factor (0.9)"
    )

    plt.xticks(
        x,
        [
            f"{beta:.0%} load"
            for beta in summary["background_load"]
        ]
    )

    plt.ylim(0, 1.05)
    plt.ylabel("Discount factor")
    plt.title("Conditional punishment incentive thresholds")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.2)

    save_figure("fig6_punishment")


def main():
    plot_stage_game()
    plot_learning_curves()
    plot_cooperation_comparison()
    plot_punishment()

    print("\nAll six figures generated.")
    print("Figures folder:", FIGURES)
    print("Tables folder:", TABLES)


if __name__ == "__main__":
    main()
