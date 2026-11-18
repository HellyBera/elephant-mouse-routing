
from itertools import product

import numpy as np
import pygambit as gbt

from game_theory import calculate_game

ACTIONS = list(product(range(3), repeat=4))


def create_gambit_game(beta):
    """Convert the routing cost game into Gambit payoffs."""

    game_data = calculate_game(beta)

    # Gambit maximizes payoffs, so payoff = -cost.
    payoffs = [
        np.zeros((3, 3, 3, 3))
        for _ in range(4)
    ]

    for actions, result in game_data.items():
        for agent in range(4):
            payoffs[agent][actions] = (
                -result["agent_costs"][agent]
            )

    game = gbt.Game.from_arrays(
        *payoffs,
        title=f"Routing Game beta={beta}"
    )

    return game, game_data



def extract_probabilities(profile):
    """Extract route probabilities for all four agents."""

    probabilities = []

    for player, mixed in profile.mixed_strategies():
        strategy_probs = np.array(
            [float(mixed[s]) for s in player.strategies],
            dtype=float
        )

        probabilities.append(strategy_probs)

    return probabilities



def expected_cost(agent, route, probabilities, game_data):
    """Cost of choosing one route against mixed opponents."""

    total = 0.0

    for actions in ACTIONS:
        if actions[agent] != route:
            continue

        probability = 1.0

        for other in range(4):
            if other != agent:
                probability *= probabilities[other][actions[other]]

        total += (
            probability
            * game_data[actions]["agent_costs"][agent]
        )

    return total


def maximum_regret(probabilities, game_data):
    """Verify whether a mixed profile is approximately NE."""

    regrets = []

    for agent in range(4):
        costs = np.array([
            expected_cost(agent, route, probabilities, game_data)
            for route in range(3)
        ])

        actual = np.dot(probabilities[agent], costs)
        best = np.min(costs)

        regrets.append(max(0.0, actual - best))

    return max(regrets)


def analyze_mixed_game(beta, max_solutions=20):
    gambit_game, game_data = create_gambit_game(beta)

    result = gbt.nash.enumpoly_solve(
        gambit_game,
        use_strategic=True,
        stop_after=max_solutions
    )

    print(f"\nBackground load: {beta:.0%}")
    print(f"Solutions returned: {len(result.equilibria)}")

    mixed_count = 0

    for index, equilibrium in enumerate(result.equilibria, 1):
        probabilities = extract_probabilities(equilibrium)

        regret = maximum_regret(probabilities, game_data)

        is_genuinely_mixed = any(
            sum(p > 1e-6 for p in strategy) > 1
            for strategy in probabilities
        )

        if is_genuinely_mixed:
            mixed_count += 1

        print(f"\nEquilibrium {index}")

        for agent, strategy in enumerate(probabilities, 1):
            print(
                f" Agent {agent}: "
                f"{np.round(strategy, 6).tolist()}"
            )

        print(f" Maximum regret: {regret:.10f}")
        print(f" Genuinely mixed: {is_genuinely_mixed}")

        if regret > 1e-6:
            print(" WARNING: Numerical equilibrium check failed.")

    print(f"\nGenuinely mixed solutions found: {mixed_count}")

    if len(result.equilibria) >= max_solutions:
        print("Search stopped at the requested solution limit.")

    return result


if __name__ == "__main__":
    analyze_mixed_game(0.10)
    analyze_mixed_game(0.25)
