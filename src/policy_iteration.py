
from itertools import product

from network import simulate_network
from environment import (
    BETA_VALUES,
    TRANSITION_MATRIX,
    SWITCHING_COST,
    GAMMA
)

NUM_AGENTS = 4
NUM_ROUTES = 3
TOL = 1e-10

# Baseline actions for the other three agents:
# Agent 1: R1, Agent 2: R2,
# Agent 3: R1, Agent 4: R3
FIXED_ROUTES = (0, 1, 0, 2)


class PolicyIterationSolver:

    def __init__(self, agent_id, fixed_routes=FIXED_ROUTES):

        if agent_id not in range(NUM_AGENTS):
            raise ValueError("Agent ID must be 0, 1, 2 or 3.")

        if len(fixed_routes) != NUM_AGENTS or any(
            r not in range(NUM_ROUTES) for r in fixed_routes
        ):
            raise ValueError("Invalid fixed routes.")

        self.agent_id = agent_id
        self.fixed_routes = tuple(fixed_routes)

        # All 162 possible Markov states
        self.states = [
            (beta_state, actions)
            for beta_state in range(2)
            for actions in product(range(3), repeat=4)
        ]

        self.state_index = {
            state: i for i, state in enumerate(self.states)
        }

        self.n_states = len(self.states)

        # Precompute deterministic action outcomes
        self.model = self.build_model()

    def build_model(self):
        """
        Build expected rewards and transition probabilities
        for the focal agent against fixed opponents.
        """

        model = {}

        for state_index, state in enumerate(self.states):

            beta_state, previous_actions = state
            beta = BETA_VALUES[beta_state]

            for action in range(NUM_ROUTES):

                joint_actions = list(self.fixed_routes)
                joint_actions[self.agent_id] = action
                joint_actions = tuple(joint_actions)

                result = simulate_network(joint_actions, beta)

                switching = int(
                    action != previous_actions[self.agent_id]
                )

                reward = (
                    -result.agent_costs[self.agent_id]
                    - SWITCHING_COST * switching
                )

                # The next previous-actions component
                # equals the current joint action.
                next_low = self.state_index[
                    (0, joint_actions)
                ]

                next_high = self.state_index[
                    (1, joint_actions)
                ]

                p_low, p_high = TRANSITION_MATRIX[beta_state]

                model[(state_index, action)] = (
                    reward,
                    (
                        (p_low, next_low),
                        (p_high, next_high)
                    )
                )

        return model

    def q_value(self, state_index, action, values):
        """Calculate a Bellman action value."""

        reward, transitions = self.model[
            (state_index, action)
        ]

        future_value = sum(
            probability * values[next_state]
            for probability, next_state in transitions
        )

        return reward + GAMMA * future_value

    def evaluate_policy(
        self, policy, tolerance=1e-11, max_sweeps=10000
    ):
        """Iterative policy evaluation."""

        values = [0.0] * self.n_states

        for sweep in range(max_sweeps):
            new_values = [
                self.q_value(s, policy[s], values)
                for s in range(self.n_states)
            ]

            delta = max(
                abs(new_values[s] - values[s])
                for s in range(self.n_states)
            )

            values = new_values

            if delta < tolerance:
                return values, sweep + 1

        raise RuntimeError("Policy evaluation did not converge.")

    def improve_policy(self, policy, values):
        """Greedy policy improvement with stable tie handling."""

        new_policy = policy.copy()
        stable = True

        for s in range(self.n_states):

            q_values = [
                self.q_value(s, a, values)
                for a in range(NUM_ROUTES)
            ]

            best_value = max(q_values)
            old_action = policy[s]

            # Retain current action if it is still optimal.
            if q_values[old_action] >= best_value - TOL:
                best_action = old_action
            else:
                best_action = max(
                    range(NUM_ROUTES),
                    key=lambda a: q_values[a]
                )

            new_policy[s] = best_action

            if best_action != old_action:
                stable = False

        return new_policy, stable

    def solve(self, max_iterations=100):
        """Run policy iteration until policy convergence."""

        policy = [0] * self.n_states
        history = []

        for iteration in range(max_iterations):

            values, sweeps = self.evaluate_policy(policy)

            new_policy, stable = self.improve_policy(
                policy, values
            )

            changed = sum(
                new_policy[s] != policy[s]
                for s in range(self.n_states)
            )

            history.append({
                "iteration": iteration + 1,
                "evaluation_sweeps": sweeps,
                "changed_states": changed
            })

            policy = new_policy

            if stable:
                return {
                    "policy": policy,
                    "values": values,
                    "iterations": iteration + 1,
                    "history": history
                }

        raise RuntimeError("Policy iteration did not converge.")


def print_results(solver, result):

    print("\n" + "=" * 55)
    print(f"AGENT {solver.agent_id + 1}: POLICY ITERATION")
    print("=" * 55)

    print("Number of states:", solver.n_states)
    print("Policy iterations:", result["iterations"])

    for entry in result["history"]:
        print(
            f"Iteration {entry['iteration']}: "
            f"Changed states = {entry['changed_states']}, "
            f"Evaluation sweeps = "
            f"{entry['evaluation_sweeps']}"
        )

    policy = result["policy"]

    counts = [
        policy.count(action)
        for action in range(NUM_ROUTES)
    ]

    print("\nOptimal policy action counts:")
    for route, count in enumerate(counts, 1):
        print(f"  R{route}: {count} states")

    # Show the optimal decision for a sample state
    sample_state = (0, (0, 1, 0, 2))
    index = solver.state_index[sample_state]

    print("\nSample state:", sample_state)
    print("Optimal route:", f"R{policy[index] + 1}")
    print("Optimal value:", round(result["values"][index], 6))


if __name__ == "__main__":

    for agent_id in range(4):

        solver = PolicyIterationSolver(agent_id)
        result = solver.solve()

        print_results(solver, result)
