
import sys
from pathlib import Path
from itertools import product
import unittest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from rewards import (
    costs_without_agent,
    calculate_externality_tolls,
    calculate_rewards
)

from network import simulate_network
from environment import SWITCHING_COST


class TestRewards(unittest.TestCase):

    def setUp(self):
        self.actions = (0, 1, 0, 2)
        self.beta = 0.10

    def test_selfish_rewards(self):

        rewards, info = calculate_rewards(
            self.actions,
            self.beta,
            self.actions,
            mode="selfish"
        )

        result = simulate_network(
            self.actions, self.beta
        )

        for i in range(4):
            self.assertAlmostEqual(
                rewards[i], -result.agent_costs[i]
            )

    def test_team_rewards(self):

        rewards, info = calculate_rewards(
            self.actions,
            self.beta,
            self.actions,
            mode="team"
        )

        expected = -simulate_network(
            self.actions, self.beta
        ).social_cost

        self.assertEqual(len(rewards), 4)

        for reward in rewards:
            self.assertAlmostEqual(reward, expected)

    def test_externality_tolls(self):

        tolls = calculate_externality_tolls(
            self.actions, self.beta
        )

        self.assertEqual(len(tolls), 4)

        # Agents 1 and 3 share R1.
        # They impose congestion costs on each other.
        self.assertGreater(tolls[0], 0)
        self.assertGreater(tolls[2], 0)

        # Agents 2 and 4 have no other agents
        # sharing their routes in this profile.
        self.assertAlmostEqual(tolls[1], 0.0)
        self.assertAlmostEqual(tolls[3], 0.0)

    def test_counterfactual(self):

        costs = costs_without_agent(
            self.actions,
            self.beta,
            removed_agent=0
        )

        self.assertNotIn(0, costs)
        self.assertEqual(len(costs), 3)

        actual = simulate_network(
            self.actions, self.beta
        ).agent_costs

        # Removing Agent 1 reduces Mouse 3's
        # congestion-related cost on R1.
        self.assertLess(costs[2], actual[2])

    def test_tolled_reward(self):

        rewards, info = calculate_rewards(
            self.actions,
            self.beta,
            self.actions,
            mode="tolled"
        )

        selfish, _ = calculate_rewards(
            self.actions,
            self.beta,
            self.actions,
            mode="selfish"
        )

        for i in range(4):
            self.assertAlmostEqual(
                rewards[i],
                selfish[i] - info["tolls"][i]
            )

    def test_switching_penalty(self):

        previous = (1, 1, 0, 2)

        rewards, info = calculate_rewards(
            self.actions,
            self.beta,
            previous,
            mode="selfish"
        )

        costs = simulate_network(
            self.actions, self.beta
        ).agent_costs

        self.assertAlmostEqual(
            rewards[0],
            -costs[0] - SWITCHING_COST
        )

        for i in (1, 2, 3):
            self.assertAlmostEqual(
                rewards[i], -costs[i]
            )

    def test_potential_game_property(self):
        """
        Verify that a unilateral agent deviation changes
        its toll-adjusted cost by exactly the change
        in total social cost.

        Test all 81 profiles, 4 agents, 2 alternatives
        per agent, and both background levels.
        """

        for beta in (0.10, 0.25):

            for actions in product(range(3), repeat=4):

                base_cost = simulate_network(
                    actions, beta
                ).social_cost

                base_tolls = calculate_externality_tolls(
                    actions, beta
                )

                base_agent_costs = simulate_network(
                    actions, beta
                ).agent_costs

                for agent in range(4):

                    for new_route in range(3):

                        if new_route == actions[agent]:
                            continue

                        changed = list(actions)
                        changed[agent] = new_route
                        changed = tuple(changed)

                        new_result = simulate_network(
                            changed, beta
                        )

                        new_tolls = calculate_externality_tolls(
                            changed, beta
                        )

                        delta_private = (
                            new_result.agent_costs[agent]
                            + new_tolls[agent]
                            - base_agent_costs[agent]
                            - base_tolls[agent]
                        )

                        delta_social = (
                            new_result.social_cost
                            - base_cost
                        )

                        self.assertAlmostEqual(
                            delta_private,
                            delta_social,
                            places=9
                        )

    def test_invalid_mode(self):

        with self.assertRaises(ValueError):
            calculate_rewards(
                self.actions,
                self.beta,
                self.actions,
                mode="unknown"
            )


if __name__ == "__main__":
    unittest.main()
