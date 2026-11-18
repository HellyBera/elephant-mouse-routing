
import sys
from pathlib import Path
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from cooperation import (
    build_toll_cache,
    learning_rewards,
    train_cooperation,
    evaluate_cooperation,
    tolled_stage_regret
)

from multi_agent import (
    train_multi_agent,
    build_reference_games
)


class TestCooperation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cache = build_toll_cache()
        cls.reference = build_reference_games()

    def test_toll_cache_size(self):
        self.assertEqual(len(self.cache), 162)

    def test_selfish_reward(self):
        original = (-0.1, -0.2, -0.3, -0.4)
        tolls = (0.01, 0.02, 0.03, 0.04)

        result = learning_rewards(
            "selfish", original, tolls
        )

        self.assertEqual(result, original)

    def test_team_reward(self):
        original = (-0.1, -0.2, -0.3, -0.4)

        result = learning_rewards(
            "team", original, (0.0,) * 4
        )

        for reward in result:
            self.assertAlmostEqual(reward, -1.0)

    def test_tolled_reward(self):
        original = (-0.1, -0.2, -0.3, -0.4)
        tolls = (0.01, 0.02, 0.03, 0.04)

        result = learning_rewards(
            "tolled", original, tolls
        )

        for i in range(4):
            self.assertAlmostEqual(
                result[i],
                original[i] - tolls[i]
            )

    def test_training_all_modes(self):
        for mode in ("selfish", "team", "tolled"):

            agents, history = train_cooperation(
                mode=mode,
                seed=42,
                episodes=5,
                steps_per_episode=20,
                toll_cache=self.cache
            )

            self.assertEqual(len(agents), 4)
            self.assertEqual(len(history), 5)

            for agent in agents:
                self.assertGreater(len(agent.Q), 0)

    def test_selfish_matches_previous_training(self):
        # The cooperation implementation must reproduce
        # the original selfish multi-agent experiment.

        old_agents, old_history = train_multi_agent(
            seed=42,
            episodes=5,
            steps_per_episode=20
        )

        new_agents, new_history = train_cooperation(
            mode="selfish",
            seed=42,
            episodes=5,
            steps_per_episode=20
        )

        for old, new in zip(old_agents, new_agents):
            self.assertEqual(
                dict(old.Q),
                dict(new.Q)
            )

        for old, new in zip(old_history, new_history):
            self.assertAlmostEqual(
                old["mean_social_cost"],
                new["mean_social_cost"]
            )

            self.assertAlmostEqual(
                old["switch_rate"],
                new["switch_rate"]
            )

    def test_tolled_regret_at_social_optimum(self):
        for beta_state in (0, 1):

            data = self.reference[beta_state]
            optimum = data["optimum_cost"]

            for actions, result in data["game"].items():

                if abs(result["social_cost"] - optimum) < 1e-10:

                    regret = tolled_stage_regret(
                        actions,
                        beta_state,
                        self.reference,
                        self.cache
                    )

                    self.assertAlmostEqual(
                        regret, 0.0, places=9
                    )

    def test_evaluation(self):
        agents, _ = train_cooperation(
            mode="team",
            seed=42,
            episodes=5,
            steps_per_episode=20
        )

        metrics = evaluate_cooperation(
            agents,
            seed=123,
            episodes=3,
            steps_per_episode=20,
            reference=self.reference,
            toll_cache=self.cache
        )

        self.assertGreaterEqual(
            metrics["mean_efficiency_ratio"],
            1.0 - 1e-9
        )

        self.assertGreaterEqual(
            metrics["mean_tolled_stage_regret"],
            0.0
        )

        self.assertTrue(
            0.0 <= metrics["switch_rate"] <= 1.0
        )


if __name__ == "__main__":
    unittest.main()
