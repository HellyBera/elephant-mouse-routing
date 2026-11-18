
import sys
from pathlib import Path
from itertools import product
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from minmax_punishment import (
    make_profile,
    calculate_pure_minmax,
    analyze_minmax
)

from game_theory import calculate_game


class TestMinmaxPunishment(unittest.TestCase):

    def test_make_profile(self):
        profile = make_profile(
            agent=0,
            own_action=1,
            opponent_actions=(0, 2, 0)
        )

        self.assertEqual(profile, (1, 0, 2, 0))

    def test_all_opponent_combinations(self):
        game = calculate_game(0.10)

        result = calculate_pure_minmax(game, 0)

        self.assertEqual(
            len(result["all_candidates"]), 27
        )

    def test_minmax_is_max_of_best_responses(self):
        for beta in (0.10, 0.25):
            game = calculate_game(beta)

            for agent in range(4):
                result = calculate_pure_minmax(
                    game, agent
                )

                maximum = max(
                    row["guaranteed_cost"]
                    for row in result["all_candidates"]
                )

                self.assertAlmostEqual(
                    result["minmax_cost"],
                    maximum
                )

    def test_each_candidate_is_best_response_cost(self):
        game = calculate_game(0.10)

        for agent in range(4):
            result = calculate_pure_minmax(
                game, agent
            )

            for candidate in result["all_candidates"]:
                costs = []

                for action in range(3):
                    profile = make_profile(
                        agent,
                        action,
                        candidate["opponent_actions"]
                    )

                    costs.append(
                        game[profile]["agent_costs"][agent]
                    )

                self.assertAlmostEqual(
                    candidate["guaranteed_cost"],
                    min(costs)
                )

    def test_minmax_at_least_nash_cost(self):
        """
        Every pure Nash equilibrium gives an agent
        a best response to its opponents' pure routes.

        The pure minmax value must be at least as large.
        """

        from game_theory import find_nash_equilibria

        for beta in (0.10, 0.25):
            game = calculate_game(beta)
            equilibria = find_nash_equilibria(game)

            for agent in range(4):
                minmax = calculate_pure_minmax(
                    game, agent
                )["minmax_cost"]

                for ne in equilibria:
                    self.assertGreaterEqual(
                        minmax + 1e-10,
                        game[ne]["agent_costs"][agent]
                    )

    def test_analysis_output(self):
        for beta in (0.10, 0.25):
            results = analyze_minmax(beta)

            self.assertEqual(len(results), 2)

            for result in results:
                self.assertEqual(
                    len(result["agents"]), 4
                )

                if result["feasible"]:
                    self.assertGreaterEqual(
                        result["gamma_required"], 0
                    )

                    self.assertLess(
                        result["gamma_required"], 1
                    )


if __name__ == "__main__":
    unittest.main()
