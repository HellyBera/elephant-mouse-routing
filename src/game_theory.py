
from itertools import product
from network import simulate_network

NUM_AGENTS = 4
NUM_ROUTES = 3
TOL = 1e-10


def generate_profiles():
    """Generate all 3^4 = 81 routing combinations."""
    return list(product(range(NUM_ROUTES), repeat=NUM_AGENTS))


def calculate_game(beta):
    """Calculate costs for every routing combination."""
    game = {}

    for actions in generate_profiles():
        result = simulate_network(actions, beta)

        game[actions] = {
            "agent_costs": result.agent_costs,
            "social_cost": result.social_cost
        }

    return game


def is_nash_equilibrium(actions, game):
    """Check if any agent can improve by changing route."""

    current_costs = game[actions]["agent_costs"]

    for agent in range(NUM_AGENTS):

        current_route = actions[agent]
        current_cost = current_costs[agent]

        for new_route in range(NUM_ROUTES):

            if new_route == current_route:
                continue

            # Copy the profile and change one agent's route
            new_actions = list(actions)
            new_actions[agent] = new_route
            new_actions = tuple(new_actions)

            new_cost = game[new_actions]["agent_costs"][agent]

            # A strict improvement means it is not a NE
            if new_cost < current_cost - TOL:
                return False

    return True


def find_nash_equilibria(game):
    """Find all pure Nash equilibria."""
    return [
        actions
        for actions in game
        if is_nash_equilibrium(actions, game)
    ]


def find_social_optimum(game):
    """Find all routing profiles with minimum social cost."""

    min_cost = min(
        result["social_cost"]
        for result in game.values()
    )

    optimal_profiles = [
        actions
        for actions, result in game.items()
        if abs(result["social_cost"] - min_cost) <= TOL
    ]

    return optimal_profiles, min_cost


def calculate_poa_pos(game, equilibria, optimal_cost):
    """Calculate Price of Anarchy and Price of Stability."""

    if not equilibria:
        return None, None

    if optimal_cost <= 0:
        raise ValueError("Optimal cost must be positive.")

    equilibrium_costs = [
        game[a]["social_cost"] for a in equilibria
    ]

    poa = max(equilibrium_costs) / optimal_cost
    pos = min(equilibrium_costs) / optimal_cost

    return poa, pos


def format_profile(actions):
    """Convert (0, 1, 0, 2) to (R1, R2, R1, R3)."""
    return "(" + ", ".join(f"R{a + 1}" for a in actions) + ")"


def run_analysis(beta):
    """Run complete one-shot game analysis."""

    game = calculate_game(beta)

    equilibria = find_nash_equilibria(game)
    optimal_profiles, optimal_cost = find_social_optimum(game)

    poa, pos = calculate_poa_pos(
        game, equilibria, optimal_cost
    )

    print("\n" + "=" * 55)
    print(f"BACKGROUND LOAD = {beta * 100:.0f}%")
    print("=" * 55)

    print(f"\nTotal profiles: {len(game)}")

    print(f"\nPure Nash equilibria: {len(equilibria)}")

    for actions in equilibria:
        cost = game[actions]["social_cost"]
        print(f"{format_profile(actions)}  Cost = {cost:.6f}")

    print("\nSocially optimal profiles:")

    for actions in optimal_profiles:
        print(f"{format_profile(actions)}  Cost = {optimal_cost:.6f}")

    print(f"\nPrice of Anarchy: {poa:.6f}")
    print(f"Price of Stability: {pos:.6f}")

    return {
        "game": game,
        "equilibria": equilibria,
        "optimal_profiles": optimal_profiles,
        "optimal_cost": optimal_cost,
        "poa": poa,
        "pos": pos
    }


if __name__ == "__main__":

    # Low background load
    run_analysis(0.10)

    # High background load
    run_analysis(0.25)
