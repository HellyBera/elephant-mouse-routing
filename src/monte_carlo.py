
import random
from collections import defaultdict

from environment import RoutingEnvironment, GAMMA
from policy_iteration import (
    PolicyIterationSolver,
    FIXED_ROUTES
)

NUM_ACTIONS = 3


class MonteCarloAgent:

    def __init__(
        self,
        agent_id,
        seed=42,
        epsilon=1.0,
        min_epsilon=0.05,
        epsilon_decay=0.997
    ):
        if agent_id not in range(4):
            raise ValueError("Invalid agent ID")

        self.agent_id = agent_id
        self.rng = random.Random(seed)

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        # Q-values and first-visit counts
        self.Q = defaultdict(
            lambda: [0.0] * NUM_ACTIONS
        )
        self.N = defaultdict(
            lambda: [0] * NUM_ACTIONS
        )

    def greedy_action(self, state):
        """Choose an action with the highest Q-value."""

        values = self.Q[state]
        best = max(values)

        # Deterministic tie-breaking for reproducibility
        return next(
            a for a, value in enumerate(values)
            if value == best
        )

    def choose_action(self, state):
        """Epsilon-greedy exploration."""

        if self.rng.random() < self.epsilon:
            return self.rng.randrange(NUM_ACTIONS)

        return self.greedy_action(state)

    def update_episode(self, episode):
        """
        First-visit Monte Carlo update.

        episode entries: (state, action, reward)
        """

        returns = [0.0] * len(episode)
        G = 0.0

        # Calculate discounted return from every time step
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + GAMMA * G
            returns[t] = G

        # Only the FIRST occurrence within the episode
        seen = set()

        for t, (state, action, reward) in enumerate(episode):
            key = (state, action)

            if key in seen:
                continue

            seen.add(key)

            self.N[state][action] += 1

            n = self.N[state][action]

            self.Q[state][action] += (
                returns[t] - self.Q[state][action]
            ) / n

    def train(
        self,
        episodes=3000,
        steps_per_episode=200,
        seed=42
    ):
        """Train against three fixed-route opponents."""

        env = RoutingEnvironment(seed=seed)
        episode_returns = []

        for ep in range(episodes):

            state = env.reset()
            trajectory = []
            total_reward = 0.0

            for t in range(steps_per_episode):

                action = self.choose_action(state)

                joint_actions = list(FIXED_ROUTES)
                joint_actions[self.agent_id] = action

                next_state, rewards, info = env.step(
                    tuple(joint_actions)
                )

                reward = rewards[self.agent_id]

                trajectory.append(
                    (state, action, reward)
                )

                total_reward += reward
                state = next_state

            self.update_episode(trajectory)

            episode_returns.append(
                total_reward / steps_per_episode
            )

            self.epsilon = max(
                self.min_epsilon,
                self.epsilon * self.epsilon_decay
            )

            if (ep + 1) % 500 == 0:
                print(
                    f"Agent {self.agent_id + 1} | "
                    f"Episode {ep + 1}/{episodes} | "
                    f"Epsilon {self.epsilon:.4f} | "
                    f"Mean reward {episode_returns[-1]:.5f}"
                )

        return episode_returns

    def extract_policy(self, states):
        """Return a deterministic greedy policy."""

        return [
            self.greedy_action(state)
            for state in states
        ]


def evaluate_against_optimum(agent_id, agent):
    """Compare learned policy with exact policy iteration."""

    solver = PolicyIterationSolver(agent_id)

    optimal = solver.solve()
    learned_policy = agent.extract_policy(solver.states)

    # Exact evaluation of the learned deterministic policy
    learned_values, _ = solver.evaluate_policy(
        learned_policy
    )

    optimal_values = optimal["values"]

    # Uniform initial-state distribution
    mean_optimal = sum(optimal_values) / len(optimal_values)
    mean_learned = sum(learned_values) / len(learned_values)

    # Agreement with one selected optimal policy.
    # Ties can make this smaller even for an optimal learner.
    agreement = sum(
        a == b for a, b in zip(
            learned_policy,
            optimal["policy"]
        )
    ) / len(learned_policy)

    value_gap = mean_optimal - mean_learned

    return {
        "mean_optimal_value": mean_optimal,
        "mean_learned_value": mean_learned,
        "value_gap": value_gap,
        "policy_agreement": agreement
    }


if __name__ == "__main__":

    for agent_id in range(4):

        print("\n" + "=" * 55)
        print(f"MONTE CARLO TRAINING: AGENT {agent_id + 1}")
        print("=" * 55)

        agent = MonteCarloAgent(
            agent_id=agent_id,
            seed=42 + agent_id
        )

        agent.train(
            episodes=3000,
            steps_per_episode=200,
            seed=100 + agent_id
        )

        metrics = evaluate_against_optimum(
            agent_id, agent
        )

        print("\nEvaluation:")
        print(
            "Optimal mean value:",
            round(metrics["mean_optimal_value"], 6)
        )
        print(
            "Learned mean value:",
            round(metrics["mean_learned_value"], 6)
        )
        print(
            "Value gap:",
            round(metrics["value_gap"], 6)
        )
        print(
            "Policy agreement:",
            f"{100 * metrics['policy_agreement']:.2f}%"
        )
