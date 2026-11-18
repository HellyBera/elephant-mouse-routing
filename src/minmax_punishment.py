
import csv
from itertools import product
from pathlib import Path

from game_theory import (
    calculate_game,
    find_nash_equilibria,
    find_social_optimum,
    format_profile
)

from repeated_game import (
    best_deviation_cost,
    incentive_threshold
)

NUM_AGENTS = 4
NUM_ROUTES = 3
TOL = 1e-10
GAMMA = 0.9
RESULTS_DIR = Path("results/data")


def make_profile(agent, own_action, opponent_actions):
    """
    Construct full joint actions from one player's
    action and the other three players' actions.
    """

    profile = [None] * NUM_AGENTS
    others = [i for i in range(NUM_AGENTS) if i != agent]

    profile[agent] = own_action

    for j, route in zip(others, opponent_actions):
        profile[j] = route

    return tuple(profile)


def calculate_pure_minmax(game, agent):
    """
    Compute:
       max_{a_-i} min_{a_i} c_i(a_i, a_-i)

    Returns the punishment cost and opponent profiles
    attaining the maximum.
    """

    if agent not in range(NUM_AGENTS):
        raise ValueError("Invalid agent ID.")

    candidate_results = []

    for opponent_actions in product(
        range(NUM_ROUTES), repeat=NUM_AGENTS - 1
    ):

        own_costs = []

        for own_action in range(NUM_ROUTES):
            profile = make_profile(
                agent, own_action, opponent_actions
            )

            cost = game[profile]["agent_costs"][agent]
            own_costs.append(cost)

        best_cost = min(own_costs)

        best_responses = [
            action
            for action, cost in enumerate(own_costs)
            if abs(cost - best_cost) <= TOL
        ]

        candidate_results.append({
            "opponent_actions": opponent_actions,
            "guaranteed_cost": best_cost,
            "best_responses": best_responses
        })

    minmax_value = max(
        row["guaranteed_cost"]
        for row in candidate_results
    )

    worst_opponent_profiles = [
        row
        for row in candidate_results
        if abs(
            row["guaranteed_cost"] - minmax_value
        ) <= TOL
    ]

    return {
        "agent": agent,
        "minmax_cost": minmax_value,
        "punishment_profiles": worst_opponent_profiles,
        "all_candidates": candidate_results
    }


def analyze_minmax(beta):
    """
    Calculate pure-action minmax punishment benchmarks
    for each socially optimal cooperation profile.
    """

    game = calculate_game(beta)

    optimal_profiles, optimal_social_cost = (
        find_social_optimum(game)
    )

    equilibria = find_nash_equilibria(game)

    minmax_results = [
        calculate_pure_minmax(game, i)
        for i in range(NUM_AGENTS)
    ]

    results = []

    for cooperation in optimal_profiles:

        cooperative_costs = game[cooperation]["agent_costs"]

        agent_results = []

        for i in range(NUM_AGENTS):

            cooperation_cost = cooperative_costs[i]

            deviation_cost = best_deviation_cost(
                game, cooperation, i
            )

            punishment_cost = (
                minmax_results[i]["minmax_cost"]
            )

            gamma_threshold = incentive_threshold(
                cooperation_cost,
                deviation_cost,
                punishment_cost
            )

            # Largest cost this agent experiences among
            # the pure Nash equilibria.
            nash_punishment_cost = max(
                game[ne]["agent_costs"][i]
                for ne in equilibria
            )

            agent_results.append({
                "agent": i,
                "cooperation_cost": cooperation_cost,
                "deviation_cost": deviation_cost,
                "minmax_cost": punishment_cost,
                "nash_punishment_cost": nash_punishment_cost,
                "gamma_threshold": gamma_threshold,
                "punishment_profiles": (
                    minmax_results[i]["punishment_profiles"]
                )
            })

        feasible = all(
            row["gamma_threshold"] is not None
            for row in agent_results
        )

        gamma_required = (
            max(row["gamma_threshold"] for row in agent_results)
            if feasible else None
        )

        results.append({
            "beta": beta,
            "cooperation": cooperation,
            "optimal_social_cost": optimal_social_cost,
            "agents": agent_results,
            "feasible": feasible,
            "gamma_required": gamma_required,
            "sustainable_incentive_at_0_9": (
                feasible and gamma_required <= GAMMA + TOL
            )
        })

    return results


def save_results(results, filename):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for result in results:
        for agent_data in result["agents"]:
            rows.append({
                "beta": result["beta"],
                "cooperation": str(result["cooperation"]),
                "agent": agent_data["agent"] + 1,
                "cooperation_cost": agent_data["cooperation_cost"],
                "deviation_cost": agent_data["deviation_cost"],
                "nash_punishment_cost": (
                    agent_data["nash_punishment_cost"]
                ),
                "pure_minmax_cost": agent_data["minmax_cost"],
                "gamma_threshold": agent_data["gamma_threshold"],
                "overall_gamma_required": (
                    result["gamma_required"]
                ),
                "incentive_feasible": result["feasible"],
                "incentive_at_gamma_0_9": (
                    result["sustainable_incentive_at_0_9"]
                )
            })

    path = RESULTS_DIR / filename

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", path)


def main():
    all_results = []

    for beta in (0.10, 0.25):
        results = analyze_minmax(beta)
        all_results.extend(results)

        print("\n" + "=" * 65)
        print(f"PURE MINMAX ANALYSIS | LOAD = {beta:.0%}")
        print("=" * 65)

        for result in results:
            print(
                "\nCooperation:",
                format_profile(result["cooperation"])
            )

            for data in result["agents"]:
                print(
                    f" Agent {data['agent'] + 1}: "
                    f"Coop={data['cooperation_cost']:.6f}, "
                    f"Deviate={data['deviation_cost']:.6f}, "
                    f"Pure minmax={data['minmax_cost']:.6f}, "
                    f"Gamma*={data['gamma_threshold']}"
                )

                example = data["punishment_profiles"][0]

                print(
                    "   Punishing opponent actions:",
                    example["opponent_actions"],
                    "| Agent best responses:",
                    example["best_responses"]
                )

            print(
                "All incentive constraints feasible:",
                result["feasible"]
            )
            print(
                "Required gamma:",
                result["gamma_required"]
            )
            print(
                "Incentive condition at gamma=0.9:",
                result["sustainable_incentive_at_0_9"]
            )

    save_results(
        all_results,
        "minmax_punishment_results.csv"
    )


if __name__ == "__main__":
    main()
