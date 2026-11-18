
import random
from collections import defaultdict

from environment import RoutingEnvironment, GAMMA
from policy_iteration import (
    PolicyIterationSolver,
    FIXED_ROUTES
)

NUM_ACTIONS = 3


class QLearningAgent:

    def __init__(
        self,
        agent_id,
        seed=42,
        learning_rate=0.1,
        epsilon=1.0,
        min_epsilon=0.05,
        epsilon_decay=0.997
    ):
        if agent_id not in range(4):
            raise ValueError("Invalid agent ID.")

        self.agent_id = agent_id
        self.rng = random.Random(seed)

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        # Q(s,a), initialized to zero
        self.Q = defaultdict(
            lambda: [0.0] * NUM_ACTIONS
        )

    def greedy_action(self, state):
        """Select the action with the highest Q-value."""

        values = self.Q[state]
        best = max(values)

        # Deterministic tie-breaking
        return next(
            a for a, value in enumerate(values)
            if value == best
        )

    def choose_action(self, state):
        """Epsilon-greedy action selection."""

        if self.rng.random() < self.epsilon:
            return self.rng.randrange(NUM_ACTIONS)

        return self.greedy_action(state)

    def update(self, state, action, reward, next_state):
        """Perform one Q-learning update."""

        current_q = self.Q[state][action]

        # Continuing discounted MDP:
        # every transition has a next-state value.
        best_next_q = max(self.Q[next_state])

        td_target = reward + GAMMA * best_next_q
        td_error = td_target - current_q

        self.Q[state][action] += (
            self.learning_rate * td_error
        )

        return td_error

    def train(
        self,
        episodes=3000,
        steps_per_episode=200,
        seed=42
    ):
        """Train one agent against three fixed opponents."""

        env = RoutingEnvironment(seed=seed)

        episode_rewards = []
        td_errors = []

        for ep in range(episodes):

            state = env.reset()
            total_reward = 0.0
            total_td_error = 0.0

            for t in range(steps_per_episode):

                action = self.choose_action(state)

                # Other three agents use fixed routes.
                joint_actions = list(FIXED_ROUTES)
                joint_actions[self.agent_id] = action

                next_state, rewards, info = env.step(
                    tuple(joint_actions)
                )

                reward = rewards[self.agent_id]

                td_error = self.update(
                    state,
                    action,
                    reward,
                    next_state
                )

                total_reward += reward
                total_td_error += abs(td_error)

                state = next_state

            episode_rewards.append(
                total_reward / steps_per_episode
            )

            td_errors.append(
                total_td_error / steps_per_episode
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
                    f"Mean reward "
                    f"{episode_rewards[-1]:.5f} | "
                    f"TD error {td_errors[-1]:.6f}"
                )

        return episode_rewards, td_errors

    def extract_policy(self, states):
        """Extract greedy policy for all states."""

        return [
            self.greedy_action(state)
            for state in states
        ]


def evaluate_against_optimum(agent_id, agent):
    """Evaluate learned Q-learning policy exactly."""

    solver = PolicyIterationSolver(agent_id)

    optimal = solver.solve()
    learned_policy = agent.extract_policy(
        solver.states
    )

    # Evaluate learned policy using the exact MDP.
    learned_values, _ = solver.evaluate_policy(
        learned_policy
    )

    optimal_values = optimal["values"]

    mean_optimal = (
        sum(optimal_values) / len(optimal_values)
    )

    mean_learned = (
        sum(learned_values) / len(learned_values)
    )

    value_gap = mean_optimal - mean_learned

    agreement = sum(
        a == b
        for a, b in zip(
            learned_policy, optimal["policy"]
        )
    ) / len(learned_policy)

    return {
        "mean_optimal_value": mean_optimal,
        "mean_learned_value": mean_learned,
        "value_gap": value_gap,
        "policy_agreement": agreement
    }


if __name__ == "__main__":

    for agent_id in range(4):

        print("\n" + "=" * 55)
        print(f"Q-LEARNING TRAINING: AGENT {agent_id + 1}")
        print("=" * 55)

        agent = QLearningAgent(
            agent_id=agent_id,
            seed=42 + agent_id
        )

        rewards, td_errors = agent.train(
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
            f"{metrics['policy_agreement'] * 100:.2f}%"
        )
