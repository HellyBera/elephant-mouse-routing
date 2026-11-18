
import sys
from pathlib import Path
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from sequential_game import (
    analyze_sequential_game,
    combine_actions,
    follower_equilibria
)
from game_theory import calculate_game


class TestSequentialGame(unittest.TestCase):

    def test_combine_actions(self):
        profile = combine_actions(
            (0, 1), (0, 2), (1, 0)
        )
        self.assertEqual(profile, (0, 2, 1, 0))

    def test_follower_ne(self):
        game = calculate_game(0.10)
        equilibria = follower_equilibria(
            game, (0, 1), (0, 1)
        )
        self.assertEqual(
            set(equilibria), {(0, 2), (2, 0)}
        )

    def test_elephants_first_low(self):
        result = analyze_sequential_game(0.10, (0, 1))
        self.assertEqual(result["pure_spne_plans"], 8)
        self.assertEqual(
            set(result["outcomes"]),
            {(0, 2, 0, 0), (2, 0, 0, 0)}
        )

    def test_mice_first_low(self):
        result = analyze_sequential_game(0.10, (2, 3))
        self.assertEqual(result["pure_spne_plans"], 1024)
        self.assertEqual(len(result["outcomes"]), 4)

    def test_elephants_first_high(self):
        result = analyze_sequential_game(0.25, (0, 1))
        self.assertEqual(result["pure_spne_plans"], 32)
        self.assertEqual(len(result["outcomes"]), 4)

    def test_mice_first_high(self):
        result = analyze_sequential_game(0.25, (2, 3))
        self.assertEqual(result["pure_spne_plans"], 1024)
        self.assertEqual(len(result["outcomes"]), 4)


if __name__ == "__main__":
    unittest.main()
