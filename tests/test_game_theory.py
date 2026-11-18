
import sys
from pathlib import Path
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from game_theory import (
    generate_profiles,
    calculate_game,
    find_nash_equilibria,
    find_social_optimum,
    calculate_poa_pos
)


class TestGameTheory(unittest.TestCase):

    def test_profile_count(self):
        profiles = generate_profiles()
        self.assertEqual(len(profiles), 81)
        self.assertEqual(len(set(profiles)), 81)

    def test_low_load_ne(self):
        game = calculate_game(0.10)
        equilibria = find_nash_equilibria(game)
        self.assertEqual(len(equilibria), 6)

    def test_high_load_ne(self):
        game = calculate_game(0.25)
        equilibria = find_nash_equilibria(game)
        self.assertEqual(len(equilibria), 8)

    def test_low_load_optimum(self):
        game = calculate_game(0.10)
        profiles, cost = find_social_optimum(game)

        self.assertAlmostEqual(cost, 0.1081168831)
        self.assertEqual(
            set(profiles),
            {(1, 2, 0, 0), (2, 1, 0, 0)}
        )

    def test_poa_pos(self):
        game = calculate_game(0.10)
        equilibria = find_nash_equilibria(game)
        _, optimal_cost = find_social_optimum(game)

        poa, pos = calculate_poa_pos(
            game, equilibria, optimal_cost
        )

        self.assertAlmostEqual(poa, 1.65419265)
        self.assertAlmostEqual(pos, 1.35470207)

    def test_social_optimum_not_ne(self):
        for beta in (0.10, 0.25):
            game = calculate_game(beta)
            equilibria = set(find_nash_equilibria(game))
            optimal_profiles, _ = find_social_optimum(game)

            self.assertTrue(
                all(p not in equilibria for p in optimal_profiles)
            )


if __name__ == "__main__":
    unittest.main()
