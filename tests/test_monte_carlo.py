
import sys
from pathlib import Path
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from monte_carlo import (
    MonteCarloAgent,
    evaluate_against_optimum
)
from environment import GAMMA


class TestMonteCarlo(unittest.TestCase):

    def test_initial_q_values(self):
        agent = MonteCarloAgent(agent_id=0)

        state = (0, (0, 1, 0, 2))

        self.assertEqual(
            agent.Q[state], [0.0, 0.0, 0.0]
        )

    def test_greedy_action(self):
        agent = MonteCarloAgent(agent_id=0)
        state = (0, (0, 1, 0, 2))

        agent.Q[state] = [1.0, 3.0, 2.0]

        self.assertEqual(
            agent.greedy_action(state), 1
        )

    def test_discounted_return(self):
        agent = MonteCarloAgent(agent_id=0)

        s1 = (0, (0, 0, 0, 0))
        s2 = (0, (1, 0, 0, 0))

        episode = [
            (s1, 0, -1.0),
            (s2, 1, -2.0)
        ]

        agent.update_episode(episode)

        self.assertAlmostEqual(
            agent.Q[s1][0],
            -1.0 + GAMMA * (-2.0)
        )

        self.assertAlmostEqual(
            agent.Q[s2][1], -2.0
        )

    def test_first_visit_only(self):
        agent = MonteCarloAgent(agent_id=0)
        s = (0, (0, 0, 0, 0))

        episode = [
            (s, 0, -1.0),
            (s, 0, -2.0)
        ]

        agent.update_episode(episode)

        self.assertEqual(agent.N[s][0], 1)
        self.assertAlmostEqual(
            agent.Q[s][0],
            -1.0 - 2.0 * GAMMA
        )

    def test_training(self):
        agent = MonteCarloAgent(
            agent_id=0, seed=42
        )

        rewards = agent.train(
            episodes=10,
            steps_per_episode=20,
            seed=42
        )

        self.assertEqual(len(rewards), 10)

        self.assertTrue(
            sum(sum(n) for n in agent.N.values()) > 0
        )

        self.assertLess(agent.epsilon, 1.0)

    def test_evaluation(self):
        agent = MonteCarloAgent(
            agent_id=0, seed=42
        )

        agent.train(
            episodes=10,
            steps_per_episode=20,
            seed=42
        )

        metrics = evaluate_against_optimum(
            0, agent
        )

        self.assertGreaterEqual(
            metrics["value_gap"], -1e-7
        )

        self.assertGreaterEqual(
            metrics["policy_agreement"], 0.0
        )

        self.assertLessEqual(
            metrics["policy_agreement"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
