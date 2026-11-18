
import sys
from pathlib import Path
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from multi_agent import (
    create_agents,
    train_multi_agent,
    evaluate_agents,
    stage_game_regret
)
from game_theory import calculate_game, find_nash_equilibria


class TestMultiAgent(unittest.TestCase):

    def test_four_agents_created(self):
        agents = create_agents(seed=42)

        self.assertEqual(len(agents), 4)
        self.assertEqual(
            [a.agent_id for a in agents],
            [0, 1, 2, 3]
        )

    def test_training(self):
        agents, history = train_multi_agent(
            seed=42,
            episodes=5,
            steps_per_episode=20
        )

        self.assertEqual(len(history), 5)

        for agent in agents:
            self.assertGreater(len(agent.Q), 0)

    def test_zero_regret_at_nash(self):
        for beta in (0.10, 0.25):

            game = calculate_game(beta)
            equilibria = find_nash_equilibria(game)

            for actions in equilibria:
                regret = stage_game_regret(
                    actions, beta, game
                )

                self.assertAlmostEqual(regret, 0.0)

    def test_nonnegative_regret(self):
        game = calculate_game(0.10)

        for actions in game:
            regret = stage_game_regret(
                actions, 0.10, game
            )

            self.assertGreaterEqual(regret, 0.0)

    def test_evaluation(self):
        agents, _ = train_multi_agent(
            seed=42,
            episodes=5,
            steps_per_episode=20
        )

        metrics = evaluate_agents(
            agents,
            seed=123,
            episodes=3,
            steps_per_episode=20
        )

        self.assertGreaterEqual(
            metrics["mean_efficiency_ratio"],
            1.0 - 1e-9
        )

        self.assertGreaterEqual(
            metrics["switch_rate"], 0.0
        )

        self.assertLessEqual(
            metrics["switch_rate"], 1.0
        )

    def test_reproducibility(self):
        agents1, history1 = train_multi_agent(
            seed=42,
            episodes=5,
            steps_per_episode=20
        )

        agents2, history2 = train_multi_agent(
            seed=42,
            episodes=5,
            steps_per_episode=20
        )

        self.assertEqual(history1, history2)

        for a1, a2 in zip(agents1, agents2):
            self.assertEqual(
                dict(a1.Q),
                dict(a2.Q)
            )


if __name__ == "__main__":
    unittest.main()
