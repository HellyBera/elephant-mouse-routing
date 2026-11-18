
import sys
from pathlib import Path
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from environment import (
    RoutingEnvironment,
    BETA_VALUES,
    TRANSITION_MATRIX,
    SWITCHING_COST
)
from network import simulate_network


class TestEnvironment(unittest.TestCase):

    def test_state_space(self):
        env = RoutingEnvironment(seed=42)
        self.assertEqual(len(env.states), 162)
        self.assertEqual(len(set(env.states)), 162)

    def test_reset(self):
        env = RoutingEnvironment(seed=42)
        state = env.reset()

        self.assertIn(state, env.states)

    def test_no_switching_penalty(self):
        env = RoutingEnvironment(seed=42)
        actions = (0, 1, 0, 2)

        # Fix state for a deterministic test
        env.state = (0, actions)

        _, rewards, info = env.step(actions)

        expected = simulate_network(actions, 0.10)

        for i in range(4):
            self.assertAlmostEqual(
                rewards[i],
                -expected.agent_costs[i]
            )

        self.assertEqual(info["switching"], (0, 0, 0, 0))

    def test_switching_penalty(self):
        env = RoutingEnvironment(seed=42)

        previous = (0, 1, 0, 2)
        current = (1, 1, 0, 2)

        env.state = (0, previous)

        _, rewards, info = env.step(current)

        expected = simulate_network(current, 0.10)

        self.assertAlmostEqual(
            rewards[0],
            -expected.agent_costs[0] - SWITCHING_COST
        )

        for i in (1, 2, 3):
            self.assertAlmostEqual(
                rewards[i],
                -expected.agent_costs[i]
            )

        self.assertEqual(info["switching"], (1, 0, 0, 0))

    def test_previous_action_update(self):
        env = RoutingEnvironment(seed=42)
        env.state = (0, (0, 0, 0, 0))

        actions = (0, 1, 2, 0)
        next_state, _, _ = env.step(actions)

        self.assertEqual(next_state[1], actions)

    def test_transition_matrix(self):
        for row in TRANSITION_MATRIX:
            self.assertAlmostEqual(sum(row), 1.0)
            self.assertTrue(all(0 <= p <= 1 for p in row))

    def test_invalid_actions(self):
        env = RoutingEnvironment(seed=42)
        env.reset()

        with self.assertRaises(ValueError):
            env.step((0, 1, 2))

        with self.assertRaises(ValueError):
            env.step((0, 1, 2, 3))

    def test_reproducibility(self):
        env1 = RoutingEnvironment(seed=123)
        env2 = RoutingEnvironment(seed=123)

        self.assertEqual(env1.reset(), env2.reset())

        actions = (0, 1, 0, 2)

        for _ in range(100):
            self.assertEqual(
                env1.step(actions),
                env2.step(actions)
            )


if __name__ == "__main__":
    unittest.main()
