
import sys
from pathlib import Path
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from q_learning import (
    QLearningAgent,
    evaluate_against_optimum
)
from environment import GAMMA


class TestQLearning(unittest.TestCase):

    def test_initial_q_values(self):
        agent = QLearningAgent(agent_id=0)
        state = (0, (0, 1, 0, 2))

        self.assertEqual(
            agent.Q[state], [0.0, 0.0, 0.0]
        )

    def test_greedy_action(self):
        agent = QLearningAgent(agent_id=0)
        state = (0, (0, 1, 0, 2))

        agent.Q[state] = [1.0, 3.0, 2.0]

        self.assertEqual(
            agent.greedy_action(state), 1
        )

    def test_q_learning_update(self):
        agent = QLearningAgent(
            agent_id=0,
            learning_rate=0.1
        )

        state = (0, (0, 0, 0, 0))
        next_state = (0, (1, 0, 0, 0))

        agent.Q[state] = [0.0, 0.0, 0.0]
        agent.Q[next_state] = [1.0, 2.0, 3.0]

        agent.update(
            state,
            action=0,
            reward=-1.0,
            next_state=next_state
        )

        expected = (
            0.1 * (-1.0 + GAMMA * 3.0)
        )

        self.assertAlmostEqual(
            agent.Q[state][0], expected
        )

    def test_epsilon_greedy(self):
        agent = QLearningAgent(
            agent_id=0, epsilon=0.0
        )

        state = (0, (0, 0, 0, 0))
        agent.Q[state] = [1.0, 5.0, 2.0]

        actions = [
            agent.choose_action(state)
            for _ in range(100)
        ]

        self.assertTrue(
            all(action == 1 for action in actions)
        )

    def test_training(self):
        agent = QLearningAgent(
            agent_id=0, seed=42
        )

        rewards, td_errors = agent.train(
            episodes=10,
            steps_per_episode=20,
            seed=42
        )

        self.assertEqual(len(rewards), 10)
        self.assertEqual(len(td_errors), 10)
        self.assertLess(agent.epsilon, 1.0)

        self.assertTrue(
            len(agent.Q) > 0
        )

    def test_evaluation(self):
        agent = QLearningAgent(
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

    def test_reproducibility(self):
        agent1 = QLearningAgent(
            agent_id=0, seed=42
        )

        agent2 = QLearningAgent(
            agent_id=0, seed=42
        )

        result1 = agent1.train(
            episodes=10,
            steps_per_episode=20,
            seed=100
        )

        result2 = agent2.train(
            episodes=10,
            steps_per_episode=20,
            seed=100
        )

        self.assertEqual(result1, result2)

        self.assertEqual(
            dict(agent1.Q),
            dict(agent2.Q)
        )


if __name__ == "__main__":
    unittest.main()
