
import sys
from pathlib import Path
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from repeated_game import (
    best_deviation_cost,
    incentive_threshold,
    analyze_punishment
)

from game_theory import (
    calculate_game,
    find_nash_equilibria,
    find_social_optimum
)


class TestRepeatedGame(unittest.TestCase):

    def test_no_profitable_deviation(self):
        self.assertEqual(
            incentive_threshold(
                cooperation_cost=0.1,
                deviation_cost=0.12,
                punishment_cost=0.2
            ),
            0.0
        )

    def test_threshold_formula(self):
        threshold = incentive_threshold(
            cooperation_cost=0.20,
            deviation_cost=0.10,
            punishment_cost=0.30
        )

        self.assertAlmostEqual(threshold, 0.5)

    def test_insufficient_punishment(self):
        threshold = incentive_threshold(
            cooperation_cost=0.20,
            deviation_cost=0.10,
            punishment_cost=0.15
        )

        self.assertIsNone(threshold)

    def test_best_deviation(self):
        game = calculate_game(0.10)
        optima, _ = find_social_optimum(game)

        for agent in range(4):
            deviation_cost = best_deviation_cost(
                game, optima[0], agent
            )

            self.assertLessEqual(
                deviation_cost,
                game[optima[0]]["agent_costs"][agent]
                + 1e-10
            )

    def test_combinations(self):
        for beta in (0.10, 0.25):
            game = calculate_game(beta)

            equilibria = find_nash_equilibria(game)
            optima, _ = find_social_optimum(game)

            results = analyze_punishment(beta)

            self.assertEqual(
                len(results),
                len(equilibria) * len(optima)
            )

    def test_feasible_thresholds(self):
        for beta in (0.10, 0.25):
            results = analyze_punishment(beta)

            for result in results:
                if result["feasible"]:
                    gamma = result["gamma_required"]

                    self.assertGreaterEqual(gamma, 0.0)
                    self.assertLess(gamma, 1.0)


if __name__ == "__main__":
    unittest.main()
