
import csv
import random
from pathlib import Path
from statistics import mean, stdev

from environment import RoutingEnvironment, GAMMA
from q_learning import QLearningAgent
from game_theory import calculate_game, find_social_optimum


NUM_AGENTS = 4
NUM_ROUTES = 3

# Same training settings as Step 7
EPISODES = 3000
STEPS_PER_EPISODE = 200

# Independent seeds for repeated experiments
SEEDS = [42, 123, 456]

# Evaluation settings
EVAL_EPISODES = 20
EVAL_STEPS = 200

RESULTS_DIR = Path("results/data")


def create_agents(seed):
    """Create four independent Q-learning agents."""

    return [
        QLearningAgent(
            agent_id=i,
            seed=seed + i,
            learning_rate=0.1,
            epsilon=1.0,
            min_epsilon=0.05,
            epsilon_decay=0.997
        )
        for i in range(NUM_AGENTS)
    ]


def stage_game_regret(actions, beta, game):
    """
    Maximum immediate cost reduction available through
    one agent changing its route unilaterally.

    Stage-game metric; excludes switching penalties.
    """

    current_costs = game[actions]["agent_costs"]
    regrets = []

    for i in range(NUM_AGENTS):

        alternative_costs = []

        for route in range(NUM_ROUTES):
            new_actions = list(actions)
            new_actions[i] = route

            alternative_costs.append(
                game[tuple(new_actions)]["agent_costs"][i]
            )

        regret = (
            current_costs[i] - min(alternative_costs)
        )

        regrets.append(max(0.0, regret))

    return max(regrets)


def build_reference_games():
    """Precalculate stage-game benchmarks."""

    reference = {}

    for beta_state, beta in enumerate((0.10, 0.25)):
        game = calculate_game(beta)
        _, optimum_cost = find_social_optimum(game)

        reference[beta_state] = {
            "game": game,
            "optimum_cost": optimum_cost
        }

    return reference


def train_multi_agent(
    seed=42,
    episodes=EPISODES,
    steps_per_episode=STEPS_PER_EPISODE
):
    """Train four Q-learning agents simultaneously."""

    agents = create_agents(seed)
    env = RoutingEnvironment(seed=seed + 1000)

    history = []

    for ep in range(episodes):

        state = env.reset()
        total_social_cost = 0.0
        total_switches = 0

        for t in range(steps_per_episode):

            # All decisions use the SAME current state.
            actions = tuple(
                agent.choose_action(state)
                for agent in agents
            )

            # Execute the joint action once.
            next_state, rewards, info = env.step(actions)

            # Every agent learns from its own reward.
            for i, agent in enumerate(agents):
                agent.update(
                    state,
                    actions[i],
                    rewards[i],
                    next_state
                )

            total_social_cost += info["social_cost"]
            total_switches += sum(info["switching"])

            state = next_state

        # Exploration decreases once per episode.
        for agent in agents:
            agent.epsilon = max(
                agent.min_epsilon,
                agent.epsilon * agent.epsilon_decay
            )

        history.append({
            "seed": seed,
            "episode": ep + 1,
            "mean_social_cost": (
                total_social_cost / steps_per_episode
            ),
            "switch_rate": (
                total_switches
                / (steps_per_episode * NUM_AGENTS)
            ),
            "epsilon": agents[0].epsilon
        })

        if (ep + 1) % 500 == 0:
            print(
                f"Seed {seed} | Episode {ep + 1} | "
                f"Social cost "
                f"{history[-1]['mean_social_cost']:.5f} | "
                f"Switch rate "
                f"{history[-1]['switch_rate']:.3f}"
            )

    return agents, history


def evaluate_agents(
    agents,
    seed=999,
    episodes=EVAL_EPISODES,
    steps_per_episode=EVAL_STEPS
):
    """
    Evaluate learned greedy policies without exploration.

    All performance metrics use identical network costs,
    independent of the Q-learning training rewards.
    """

    env = RoutingEnvironment(seed=seed)
    reference = build_reference_games()

    total_social_cost = 0.0
    total_efficiency_ratio = 0.0
    total_regret = 0.0
    total_switches = 0
    total_steps = 0

    beta_costs = {0: [], 1: []}

    for ep in range(episodes):

        state = env.reset()

        for t in range(steps_per_episode):

            beta_state = state[0]

            # Greedy evaluation: epsilon is not used.
            actions = tuple(
                agent.greedy_action(state)
                for agent in agents
            )

            next_state, rewards, info = env.step(actions)

            game = reference[beta_state]["game"]
            optimum = reference[beta_state]["optimum_cost"]

            social_cost = info["social_cost"]
            regret = stage_game_regret(
                actions,
                beta_state,
                game
            )

            total_social_cost += social_cost
            total_efficiency_ratio += social_cost / optimum
            total_regret += regret
            total_switches += sum(info["switching"])
            total_steps += 1

            beta_costs[beta_state].append(social_cost)

            state = next_state

    return {
        "mean_social_cost": total_social_cost / total_steps,
        "mean_efficiency_ratio": (
            total_efficiency_ratio / total_steps
        ),
        "mean_stage_regret": total_regret / total_steps,
        "switch_rate": (
            total_switches / (total_steps * NUM_AGENTS)
        ),
        "low_load_cost": (
            mean(beta_costs[0]) if beta_costs[0] else None
        ),
        "high_load_cost": (
            mean(beta_costs[1]) if beta_costs[1] else None
        )
    }


def save_csv(filename, rows):
    """Save reproducible numerical results."""

    if not rows:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    path = RESULTS_DIR / filename

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def main():

    all_history = []
    all_results = []

    for seed in SEEDS:

        print("\n" + "=" * 55)
        print(f"MULTI-AGENT Q-LEARNING | SEED {seed}")
        print("=" * 55)

        agents, history = train_multi_agent(seed=seed)

        metrics = evaluate_agents(
            agents, seed=seed + 2000
        )

        print("\nEvaluation:", metrics)

        all_history.extend(history)
        all_results.append({
            "seed": seed,
            **metrics
        })

    save_csv("multi_agent_training.csv", all_history)
    save_csv("multi_agent_evaluation.csv", all_results)

    print("\n" + "=" * 55)
    print("SUMMARY ACROSS SEEDS")
    print("=" * 55)

    for metric in [
        "mean_social_cost",
        "mean_efficiency_ratio",
        "mean_stage_regret",
        "switch_rate"
    ]:
        values = [row[metric] for row in all_results]

        print(
            f"{metric}: "
            f"{mean(values):.6f} +/- "
            f"{stdev(values):.6f}"
        )


if __name__ == "__main__":
    main()
