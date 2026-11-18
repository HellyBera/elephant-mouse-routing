
import sys
from pathlib import Path
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from policy_iteration import (
    PolicyIterationSolver,
    GAMMA,
    SWITCHING_COST
)
from network import simulate_network


class TestPolicyIteration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.solver = PolicyIterationSolver(agent_id=0)
        cls.result = cls.solver.solve()

    def test_state_count(self):
        self.assertEqual(self.solver.n_states, 162)
        self.assertEqual(len(self.solver.model), 486)

    def test_transition_probabilities(self):
        for reward, transitions in self.solver.model.values():
            self.assertAlmostEqual(
                sum(p for p, _ in transitions), 1.0
            )

    def test_immediate_reward(self):
        # Previous actions equal current joint actions:
        # there is no switching penalty.
        state = (0, (0, 1, 0, 2))
        s = self.solver.state_index[state]

        reward, _ = self.solver.model[(s, 0)]

        expected = simulate_network(
            (0, 1, 0, 2), 0.10
        )

        self.assertAlmostEqual(
            reward,
            -expected.agent_costs[0]
        )

    def test_switching_penalty(self):
        state = (0, (0, 1, 0, 2))
        s = self.solver.state_index[state]

        reward, _ = self.solver.model[(s, 1)]

        expected = simulate_network(
            (1, 1, 0, 2), 0.10
        )

        self.assertAlmostEqual(
            reward,
            -expected.agent_costs[0] - SWITCHING_COST
        )

    def test_policy_is_greedy(self):
        policy = self.result["policy"]
        values = self.result["values"]

        for s in range(self.solver.n_states):
            chosen_value = self.solver.q_value(
                s, policy[s], values
            )

            best_value = max(
                self.solver.q_value(s, a, values)
                for a in range(3)
            )

            self.assertAlmostEqual(
                chosen_value, best_value, places=7
            )

    def test_bellman_optimality(self):
        values = self.result["values"]

        residual = max(
            abs(
                values[s] -
                max(
                    self.solver.q_value(s, a, values)
                    for a in range(3)
                )
            )
            for s in range(self.solver.n_states)
        )

        self.assertLess(residual, 1e-8)

    def test_optimal_beats_fixed_route(self):
        # Compare optimal policy with always using R1.
        baseline_policy = [0] * self.solver.n_states

        baseline_values, _ = self.solver.evaluate_policy(
            baseline_policy
        )

        optimal_values = self.result["values"]

        for optimal, baseline in zip(
            optimal_values, baseline_values
        ):
            self.assertGreaterEqual(
                optimal + 1e-8, baseline
            )


if __name__ == "__main__":
    unittest.main()
