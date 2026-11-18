
import sys
from pathlib import Path
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from network import simulate_network


class TestNetwork(unittest.TestCase):

    def test_route_loads(self):
        result = simulate_network((0, 1, 0, 2), 0.10)
        self.assertEqual(result.loads, (31.0, 30.0, 10.0))

    def test_delay(self):
        result = simulate_network((0, 1, 0, 2), 0.10)
        self.assertAlmostEqual(result.delays[0], 5.2631579)

    def test_social_cost(self):
        result = simulate_network((0, 1, 0, 2), 0.10)
        self.assertAlmostEqual(
            result.social_cost, 0.1464661654, places=6
        )

    def test_overloaded_route(self):
        result = simulate_network((0, 0, 0, 0), 0.25)
        self.assertEqual(result.delays[0], 100.0)
        self.assertLess(result.throughput[0], 1.0)

    def test_invalid_action(self):
        with self.assertRaises(ValueError):
            simulate_network((0, 1, 3, 2), 0.10)


if __name__ == "__main__":
    unittest.main()
