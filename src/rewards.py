
from network import (
    simulate_network,
    ROUTE_CAPACITY,
    AGENT_DEMAND,
    AGENT_ALPHA,
    calculate_latency,
    calculate_throughput,
    L_MAX
)
from environment import SWITCHING_COST


NUM_AGENTS = 4
NUM_ROUTES = 3

REWARD_MODES = ("selfish", "team", "tolled")


def validate_actions(actions):
    """Check that all four routing decisions are valid."""

    actions = tuple(actions)

    if len(actions) != NUM_AGENTS:
        raise ValueError("Exactly four actions required.")

    if any(a not in range(NUM_ROUTES) for a in actions):
        raise ValueError("Routes must be 0, 1, or 2.")

    return actions


def costs_without_agent(actions, beta, removed_agent):
    """
    Calculate the costs of the remaining agents when
    one agent is removed from the network.

    Other agents keep their current routes.
    Background traffic remains unchanged.

    Returns:
        Dictionary mapping remaining agent IDs to costs.
    """

    actions = validate_actions(actions)

    if removed_agent not in range(NUM_AGENTS):
        raise ValueError("Invalid agent ID.")

    if not 0 <= beta <= 1:
        raise ValueError("Invalid background load.")

    # Start with background traffic only.
    loads = [
        beta * capacity
        for capacity in ROUTE_CAPACITY
    ]

    # Add demands of everyone except removed_agent.
    for i, route in enumerate(actions):

        if i == removed_agent:
            continue

        loads[route] += AGENT_DEMAND[i]

    delays = calculate_latency(loads)
    throughput = calculate_throughput(loads)

    latency = [
        min(1.0, d / L_MAX)
        for d in delays
    ]

    costs = {}

    for i, route in enumerate(actions):

        if i == removed_agent:
            continue

        alpha = AGENT_ALPHA[i]

        costs[i] = (
            alpha * (1 - throughput[route])
            + (1 - alpha) * latency[route]
        )

    return costs


def calculate_externality_tolls(actions, beta):
    """
    Calculate the congestion externality imposed by
    each agent on the other three agents.
    """

    actions = validate_actions(actions)

    result = simulate_network(actions, beta)
    actual_costs = result.agent_costs

    tolls = []

    for i in range(NUM_AGENTS):

        counterfactual = costs_without_agent(
            actions, beta, removed_agent=i
        )

        externality = sum(
            actual_costs[j] - counterfactual[j]
            for j in range(NUM_AGENTS)
            if j != i
        )

        tolls.append(externality)

    return tuple(tolls)


def calculate_rewards(
    actions,
    beta,
    previous_actions,
    mode="selfish"
):
    """
    Return four learning rewards.

    selfish: individual network reward
    team:    same total reward for all agents
    tolled:  individual reward minus congestion externality
    """

    actions = validate_actions(actions)
    previous_actions = validate_actions(previous_actions)

    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode: {mode}")

    result = simulate_network(actions, beta)

    switching = tuple(
        int(actions[i] != previous_actions[i])
        for i in range(NUM_AGENTS)
    )

    selfish_rewards = tuple(
        -result.agent_costs[i]
        - SWITCHING_COST * switching[i]
        for i in range(NUM_AGENTS)
    )

    if mode == "selfish":
        rewards = selfish_rewards
        tolls = (0.0,) * NUM_AGENTS

    elif mode == "team":
        total_team_reward = sum(selfish_rewards)

        rewards = (total_team_reward,) * NUM_AGENTS
        tolls = (0.0,) * NUM_AGENTS

    else:
        tolls = calculate_externality_tolls(
            actions, beta
        )

        rewards = tuple(
            selfish_rewards[i] - tolls[i]
            for i in range(NUM_AGENTS)
        )

    info = {
        "mode": mode,
        "social_cost": result.social_cost,
        "network_costs": result.agent_costs,
        "switching": switching,
        "tolls": tolls,
        "selfish_rewards": selfish_rewards
    }

    return rewards, info


if __name__ == "__main__":

    actions = (0, 1, 0, 2)
    previous_actions = (0, 1, 0, 2)
    beta = 0.10

    print("Routing profile:", actions)
    print("Background load:", beta)

    for mode in REWARD_MODES:

        rewards, info = calculate_rewards(
            actions,
            beta,
            previous_actions,
            mode=mode
        )

        print("\nMode:", mode)
        print(
            "Rewards:",
            tuple(round(r, 6) for r in rewards)
        )

        print(
            "Tolls:",
            tuple(round(t, 6) for t in info["tolls"])
        )

        print(
            "Social cost:",
            round(info["social_cost"], 6)
        )
