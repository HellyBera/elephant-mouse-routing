
from dataclasses import dataclass
from typing import Tuple


# Parameters from the project proposal
ROUTE_CAPACITY = (50.0, 100.0, 40.0)  # Mbps
ROUTE_DELAY = (2.0, 10.0, 5.0)        # ms

AGENT_DEMAND = (20.0, 20.0, 6.0, 6.0)  # Mbps
AGENT_ALPHA = (0.8, 0.8, 0.1, 0.1)

D_MAX = 100.0
L_MAX = 100.0


@dataclass
class NetworkResult:
    loads: Tuple[float, ...]
    delays: Tuple[float, ...]
    throughput: Tuple[float, ...]
    latency: Tuple[float, ...]
    agent_costs: Tuple[float, ...]
    social_cost: float


def calculate_route_load(actions, beta):
    """Calculate traffic load on each route."""
    loads = [beta * c for c in ROUTE_CAPACITY]

    for agent, route in enumerate(actions):
        loads[route] += AGENT_DEMAND[agent]

    return tuple(loads)


def calculate_latency(loads):
    """Calculate route delays in milliseconds."""
    delays = []

    for r in range(3):
        x = loads[r]
        capacity = ROUTE_CAPACITY[r]

        if x >= capacity:
            delay = D_MAX
        else:
            delay = min(
                D_MAX,
                ROUTE_DELAY[r] / (1 - x / capacity)
            )

        delays.append(delay)

    return tuple(delays)


def calculate_throughput(loads):
    """Calculate normalized route throughput."""
    throughput = []

    for r in range(3):
        x = loads[r]
        capacity = ROUTE_CAPACITY[r]

        if x == 0:
            t = 1.0
        else:
            t = min(1.0, capacity / x)

        throughput.append(t)

    return tuple(throughput)


def calculate_agent_costs(actions, throughput, latency):
    """Calculate each agent's QoS cost."""
    costs = []

    for i, route in enumerate(actions):
        alpha = AGENT_ALPHA[i]

        cost = (
            alpha * (1 - throughput[route])
            + (1 - alpha) * latency[route]
        )

        costs.append(cost)

    return tuple(costs)


def simulate_network(actions, beta):
    """Run the complete network cost calculation."""

    if len(actions) != 4:
        raise ValueError("Exactly four actions are required.")

    if any(a not in (0, 1, 2) for a in actions):
        raise ValueError("Routes must be 0, 1, or 2.")

    if not 0 <= beta <= 1:
        raise ValueError("Background load must be between 0 and 1.")

    loads = calculate_route_load(actions, beta)
    delays = calculate_latency(loads)
    throughput = calculate_throughput(loads)

    latency = tuple(
        min(1.0, d / L_MAX) for d in delays
    )

    costs = calculate_agent_costs(
        actions, throughput, latency
    )

    return NetworkResult(
        loads=loads,
        delays=delays,
        throughput=throughput,
        latency=latency,
        agent_costs=costs,
        social_cost=sum(costs)
    )


if __name__ == "__main__":

    # 0 = R1, 1 = R2, 2 = R3
    actions = (0, 1, 0, 2)
    beta = 0.10

    result = simulate_network(actions, beta)

    print("Route loads:", result.loads)
    print("Route delays:", result.delays)
    print("Normalized throughput:", result.throughput)
    print("Normalized latency:", result.latency)
    print("Agent costs:", result.agent_costs)
    print("Total social cost:", result.social_cost)
