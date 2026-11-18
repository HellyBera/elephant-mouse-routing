
import random
from itertools import product

from network import simulate_network


# Background states:
# 0 = low load (10%)
# 1 = high load (25%)

BETA_VALUES = (0.10, 0.25)

TRANSITION_MATRIX = (
    (0.98, 0.02),
    (0.04, 0.96)
)

SWITCHING_COST = 0.02
GAMMA = 0.90

NUM_AGENTS = 4
NUM_ROUTES = 3


class RoutingEnvironment:
    """Repeated four-agent Markov routing game."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

        self.action_space = tuple(range(NUM_ROUTES))

        self.states = [
            (beta_state, actions)
            for beta_state in range(2)
            for actions in product(range(NUM_ROUTES), repeat=4)
        ]

        self.state = None

    def reset(self, seed=None):
        """Randomly initialize the Markov-game state."""

        if seed is not None:
            self.rng.seed(seed)

        self.state = self.rng.choice(self.states)

        return self.state

    def _sample_next_beta(self, beta_state):
        """Sample background load using the Markov chain."""

        probability_high = TRANSITION_MATRIX[beta_state][1]

        if self.rng.random() < probability_high:
            return 1

        return 0

    def step(self, joint_actions):
        """Execute one simultaneous routing decision."""

        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        joint_actions = tuple(joint_actions)

        if len(joint_actions) != NUM_AGENTS:
            raise ValueError("Exactly four actions required.")

        if any(a not in self.action_space for a in joint_actions):
            raise ValueError("Invalid routing action.")

        beta_state, previous_actions = self.state
        beta = BETA_VALUES[beta_state]

        # Network cost from Step 1
        result = simulate_network(joint_actions, beta)

        # Calculate switching penalties
        switching = tuple(
            1 if joint_actions[i] != previous_actions[i] else 0
            for i in range(NUM_AGENTS)
        )

        # Reward = negative QoS cost - switching penalty
        rewards = tuple(
            -result.agent_costs[i]
            - SWITCHING_COST * switching[i]
            for i in range(NUM_AGENTS)
        )

        next_beta_state = self._sample_next_beta(beta_state)

        # Current actions become the next state's previous actions
        next_state = (next_beta_state, joint_actions)

        info = {
            "beta": beta,
            "network_costs": result.agent_costs,
            "social_cost": result.social_cost,
            "switching": switching,
            "total_reward": sum(rewards)
        }

        self.state = next_state

        return next_state, rewards, info


if __name__ == "__main__":

    env = RoutingEnvironment(seed=42)

    state = env.reset()
    print("Initial state:", state)

    print("\nRunning 10 random steps:\n")

    for t in range(10):
        actions = tuple(
            env.rng.choice(env.action_space)
            for _ in range(NUM_AGENTS)
        )

        next_state, rewards, info = env.step(actions)

        print(f"Step {t + 1}")
        print(" Actions:", actions)
        print(" Background load:", info["beta"])
        print(" Rewards:", tuple(round(r, 5) for r in rewards))
        print(" Social cost:", round(info["social_cost"], 5))
        print(" Next state:", next_state)
        print()
