
import csv
from pathlib import Path

from game_theory import (
    calculate_game,
    find_nash_equilibria,
    find_social_optimum,
    format_profile
)

TOL = 1e-10
RESULTS_DIR = Path("results/data")


def best_deviation_cost(game, cooperation, agent):
    """Find lowest one-shot cost from unilateral deviation."""

    costs = []

    for route in range(3):
        changed = list(cooperation)
        changed[agent] = route

        costs.append(
            game[tuple(changed)]["agent_costs"][agent]
        )

    return min(costs)


def incentive_threshold(
    cooperation_cost,
    deviation_cost,
    punishment_cost
):
    """
    Return minimum discount factor to deter deviation.

    None means no gamma < 1 can sustain cooperation
    under this punishment.
    """

    # No profitable deviation
    if deviation_cost >= cooperation_cost - TOL:
        return 0.0

    denominator = punishment_cost - deviation_cost

    # Punishment is insufficient
    if denominator <= TOL:
        return None

    threshold = (
        (cooperation_cost - deviation_cost)
        / denominator
    )

    if threshold >= 1.0 - TOL:
        return None

    return max(0.0, threshold)


def analyze_punishment(beta):
    """Evaluate all social optima against all pure Nash punishments."""

    game = calculate_game(beta)
    equilibria = find_nash_equilibria(game)
    optimal_profiles, optimum_cost = find_social_optimum(game)

    results = []

    for cooperation in optimal_profiles:

        cooperation_costs = game[cooperation]["agent_costs"]

        for punishment in equilibria:

            punishment_costs = game[punishment]["agent_costs"]
            thresholds = []

            for agent in range(4):
                deviation_cost = best_deviation_cost(
                    game, cooperation, agent
                )

                threshold = incentive_threshold(
                    cooperation_costs[agent],
                    deviation_cost,
                    punishment_costs[agent]
                )

                thresholds.append(threshold)

            feasible = all(
                threshold is not None
                for threshold in thresholds
            )

            overall_threshold = (
                max(thresholds) if feasible else None
            )

            results.append({
                "beta": beta,
                "cooperation": cooperation,
                "punishment": punishment,
                "optimum_cost": optimum_cost,
                "thresholds": thresholds,
                "gamma_required": overall_threshold,
                "feasible": feasible,
                "sustainable_at_0_9": (
                    feasible and overall_threshold <= 0.9
                )
            })

    return results


def save_results(rows, filename):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    path = RESULTS_DIR / filename

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", path)


def main():
    saved_rows = []

    for beta in (0.10, 0.25):

        results = analyze_punishment(beta)

        print("\n" + "=" * 65)
        print(f"BACKGROUND LOAD: {beta:.0%}")
        print("=" * 65)

        feasible = [
            row for row in results if row["feasible"]
        ]

        print("Combinations evaluated:", len(results))
        print("Feasible punishment plans:", len(feasible))

        if feasible:
            best = min(
                feasible,
                key=lambda row: row["gamma_required"]
            )

            print("\nBest feasible Nash-reversion plan:")
            print(
                "Cooperation:",
                format_profile(best["cooperation"])
            )
            print(
                "Punishment:",
                format_profile(best["punishment"])
            )
            print(
                "Agent thresholds:",
                best["thresholds"]
            )
            print(
                "Required gamma:",
                round(best["gamma_required"], 6)
            )
            print(
                "Sustainable at gamma=0.9:",
                best["sustainable_at_0_9"]
            )
        else:
            print(
                "No tested pure Nash-reversion plan "
                "sustains the social optimum."
            )

        for row in results:
            saved_rows.append({
                "beta": row["beta"],
                "cooperation": str(row["cooperation"]),
                "punishment": str(row["punishment"]),
                "gamma_required": row["gamma_required"],
                "feasible": row["feasible"],
                "sustainable_at_0_9": (
                    row["sustainable_at_0_9"]
                ),
                **{
                    f"agent_{i+1}_threshold": row["thresholds"][i]
                    for i in range(4)
                }
            })

    save_results(
        saved_rows,
        "repeated_game_thresholds.csv"
    )


if __name__ == "__main__":
    main()
