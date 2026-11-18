
from itertools import product
from math import prod

from game_theory import calculate_game, format_profile

ROUTES = range(3)
TOL = 1e-10

PAIR_ACTIONS = list(product(ROUTES, repeat=2))


def combine_actions(leaders, leader_actions, follower_actions):
    """Build the complete four-agent action profile."""

    followers = tuple(i for i in range(4) if i not in leaders)
    actions = [None] * 4

    for agent, route in zip(leaders, leader_actions):
        actions[agent] = route

    for agent, route in zip(followers, follower_actions):
        actions[agent] = route

    return tuple(actions)


def pair_deviations(actions):
    """All deviations by one of the two agents."""

    for position in range(2):
        for route in ROUTES:
            if route == actions[position]:
                continue

            alternative = list(actions)
            alternative[position] = route

            yield position, tuple(alternative)


def follower_equilibria(game, leaders, leader_actions):
    """Pure NE of the followers after an observed leader choice."""

    followers = tuple(i for i in range(4) if i not in leaders)
    equilibria = []

    for follower_actions in PAIR_ACTIONS:
        profile = combine_actions(
            leaders, leader_actions, follower_actions
        )

        current_costs = game[profile]["agent_costs"]
        is_equilibrium = True

        for position, alternative in pair_deviations(follower_actions):
            agent = followers[position]

            deviated_profile = combine_actions(
                leaders, leader_actions, alternative
            )

            new_cost = game[deviated_profile]["agent_costs"][agent]

            if new_cost < current_costs[agent] - TOL:
                is_equilibrium = False
                break

        if is_equilibrium:
            equilibria.append(follower_actions)

    return equilibria


def analyze_sequential_game(beta, leaders):
    """Enumerate pure-strategy SPNE using backward induction."""

    game = calculate_game(beta)

    # Stage 2: solve every possible follower subgame.
    responses = {
        a: follower_equilibria(game, leaders, a)
        for a in PAIR_ACTIONS
    }

    if any(len(v) == 0 for v in responses.values()):
        raise ValueError(
            "A follower subgame has no pure equilibrium. "
            "Mixed-strategy continuation is required."
        )

    # A continuation plan specifies an equilibrium follower
    # response for EVERY possible leader action.
    possible_responses = [
        responses[a] for a in PAIR_ACTIONS
    ]

    spne_plans = 0
    unique_outcomes = set()

    for selection in product(*possible_responses):
        response_plan = dict(zip(PAIR_ACTIONS, selection))

        # Stage 1: find pure NE of the induced leader game.
        for leader_actions in PAIR_ACTIONS:

            follower_actions = response_plan[leader_actions]

            profile = combine_actions(
                leaders, leader_actions, follower_actions
            )

            current_costs = game[profile]["agent_costs"]
            is_leader_ne = True

            for position, deviation in pair_deviations(leader_actions):
                agent = leaders[position]

                # Followers respond to the deviated leader choice.
                new_follower_actions = response_plan[deviation]

                new_profile = combine_actions(
                    leaders, deviation, new_follower_actions
                )

                new_cost = game[new_profile]["agent_costs"][agent]

                if new_cost < current_costs[agent] - TOL:
                    is_leader_ne = False
                    break

            if is_leader_ne:
                spne_plans += 1
                unique_outcomes.add(profile)

    outcomes = sorted(unique_outcomes)

    return {
        "beta": beta,
        "leaders": leaders,
        "responses": responses,
        "continuation_plans": prod(
            len(v) for v in possible_responses
        ),
        "pure_spne_plans": spne_plans,
        "outcomes": outcomes,
        "game": game
    }


def print_results(result):
    leaders = result["leaders"]

    label = (
        "ELEPHANTS FIRST"
        if leaders == (0, 1)
        else "MICE FIRST"
    )

    print("\n" + "=" * 60)
    print(label)
    print(f"Background load: {result['beta']:.0%}")
    print("=" * 60)

    print("\nFollower equilibria by first-stage choice:")

    for leader_actions, equilibria in result["responses"].items():
        print(f"  {leader_actions}: {equilibria}")

    print("\nContinuation plans:",
          result["continuation_plans"])

    print("Pure SPNE strategy plans:",
          result["pure_spne_plans"])

    print("\nDistinct equilibrium outcomes:")

    for profile in result["outcomes"]:
        cost = result["game"][profile]["social_cost"]
        print(f"  {format_profile(profile)}: {cost:.6f}")


if __name__ == "__main__":

    for beta in (0.10, 0.25):
        for leaders in ((0, 1), (2, 3)):
            result = analyze_sequential_game(beta, leaders)
            print_results(result)
