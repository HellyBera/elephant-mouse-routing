
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_iteration import PolicyIterationSolver
from monte_carlo import MonteCarloAgent
from q_learning import QLearningAgent

DATA = ROOT / "results" / "data"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

for folder in (DATA, FIGURES, TABLES):
    folder.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2026]
EPISODES = 3000
STEPS = 200

AGENT_NAMES = [
    "Elephant 1",
    "Elephant 2",
    "Mouse 3",
    "Mouse 4"
]


def evaluate_policy(solver, optimal, learned_policy):
    """
    Compare a learned policy against the exact MDP
    using the uniform initial-state distribution.
    """

    learned_values, _ = solver.evaluate_policy(
        learned_policy
    )

    optimal_values = optimal["values"]
    optimal_policy = optimal["policy"]

    optimal_mean = float(np.mean(optimal_values))
    learned_mean = float(np.mean(learned_values))

    agreement = np.mean([
        learned_policy[s] == optimal_policy[s]
        for s in range(solver.n_states)
    ])

    # Check value quality independently of action-label ties
    value_gap = optimal_mean - learned_mean

    return {
        "optimal_value": optimal_mean,
        "learned_value": learned_mean,
        "value_gap": value_gap,
        "policy_agreement": float(agreement)
    }


def train_one(algorithm, agent_id, seed):
    """Train Monte Carlo or Q-learning from scratch."""

    if algorithm == "Monte Carlo":
        learner = MonteCarloAgent(
            agent_id=agent_id,
            seed=seed + agent_id
        )

        history = learner.train(
            episodes=EPISODES,
            steps_per_episode=STEPS,
            seed=seed + 100 + agent_id
        )

    elif algorithm == "Q-learning":
        learner = QLearningAgent(
            agent_id=agent_id,
            seed=seed + agent_id
        )

        rewards, td_errors = learner.train(
            episodes=EPISODES,
            steps_per_episode=STEPS,
            seed=seed + 100 + agent_id
        )

        history = rewards

    else:
        raise ValueError("Unknown algorithm")

    return learner, history


def run_experiments():
    """Train and evaluate both RL algorithms."""

    all_results = []
    all_history = []

    # Solve exact benchmark once per agent.
    benchmarks = {}

    for agent_id in range(4):
        solver = PolicyIterationSolver(agent_id)
        optimal = solver.solve()
        benchmarks[agent_id] = (solver, optimal)

    for algorithm in ("Monte Carlo", "Q-learning"):

        for agent_id in range(4):

            solver, optimal = benchmarks[agent_id]

            for seed in SEEDS:

                print("\n" + "=" * 55)
                print(
                    f"{algorithm} | {AGENT_NAMES[agent_id]} "
                    f"| Seed {seed}"
                )
                print("=" * 55)

                learner, history = train_one(
                    algorithm, agent_id, seed
                )

                learned_policy = learner.extract_policy(
                    solver.states
                )

                metrics = evaluate_policy(
                    solver, optimal, learned_policy
                )

                all_results.append({
                    "algorithm": algorithm,
                    "agent": AGENT_NAMES[agent_id],
                    "agent_id": agent_id,
                    "seed": seed,
                    **metrics
                })

                for ep, reward in enumerate(history, 1):
                    all_history.append({
                        "algorithm": algorithm,
                        "agent": AGENT_NAMES[agent_id],
                        "seed": seed,
                        "episode": ep,
                        "mean_training_reward": reward
                    })

                print(
                    f"Optimal value: "
                    f"{metrics['optimal_value']:.6f}"
                )

                print(
                    f"Learned value: "
                    f"{metrics['learned_value']:.6f}"
                )

                print(
                    f"Value gap: "
                    f"{metrics['value_gap']:.6f}"
                )

                print(
                    f"Policy agreement: "
                    f"{metrics['policy_agreement'] * 100:.2f}%"
                )

    results = pd.DataFrame(all_results)
    histories = pd.DataFrame(all_history)

    results.to_csv(
        DATA / "rl_comparison.csv",
        index=False
    )

    histories.to_csv(
        DATA / "rl_training_history.csv",
        index=False
    )

    return results, histories


def create_summary(results):
    """Calculate mean and sample standard deviation."""

    summary = (
        results.groupby(["agent", "algorithm"])
        .agg(
            mean_value_gap=("value_gap", "mean"),
            std_value_gap=("value_gap", "std"),
            mean_agreement=("policy_agreement", "mean"),
            std_agreement=("policy_agreement", "std"),
            mean_learned_value=("learned_value", "mean"),
            optimal_value=("optimal_value", "first"),
            num_seeds=("seed", "nunique")
        )
        .reset_index()
    )

    summary.to_csv(
        TABLES / "rl_comparison_summary.csv",
        index=False
    )

    print("\nFINAL RL COMPARISON")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    return summary


def plot_value_gap(summary):
    """Plot the final policy value gaps."""

    agents = AGENT_NAMES
    x = np.arange(len(agents))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for offset, algorithm in [
        (-width / 2, "Monte Carlo"),
        (width / 2, "Q-learning")
    ]:
        rows = (
            summary[summary["algorithm"] == algorithm]
            .set_index("agent")
            .loc[agents]
        )

        ax.bar(
            x + offset,
            rows["mean_value_gap"].to_numpy(),
            width=width,
            yerr=rows["std_value_gap"].to_numpy(),
            capsize=4,
            label=algorithm
        )

    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Value gap from optimal (lower is better)")
    ax.set_title("Monte Carlo vs Q-learning")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"fig7_rl_value_gap.{ext}",
            dpi=300,
            bbox_inches="tight"
        )

    plt.close(fig)


def plot_policy_agreement(summary):
    """Plot agreement with the selected optimal policy."""

    agents = AGENT_NAMES
    x = np.arange(len(agents))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for offset, algorithm in [
        (-width / 2, "Monte Carlo"),
        (width / 2, "Q-learning")
    ]:
        rows = (
            summary[summary["algorithm"] == algorithm]
            .set_index("agent")
            .loc[agents]
        )

        ax.bar(
            x + offset,
            100 * rows["mean_agreement"].to_numpy(),
            width=width,
            yerr=100 * rows["std_agreement"].to_numpy(),
            capsize=4,
            label=algorithm
        )

    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Policy agreement (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Agreement with policy iteration")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"fig8_policy_agreement.{ext}",
            dpi=300,
            bbox_inches="tight"
        )

    plt.close(fig)


def main():
    results, histories = run_experiments()
    summary = create_summary(results)

    plot_value_gap(summary)
    plot_policy_agreement(summary)

    print("\nExperiment completed.")
    print("Raw results:", DATA / "rl_comparison.csv")
    print("Summary:", TABLES / "rl_comparison_summary.csv")
    print("Figures:", FIGURES)


if __name__ == "__main__":
    main()
