
import csv
from pathlib import Path
from statistics import mean, stdev

from environment import RoutingEnvironment, BETA_VALUES
from q_learning import QLearningAgent
from multi_agent import (
    create_agents,
    build_reference_games,
    stage_game_regret
)
from rewards import (
    calculate_externality_tolls,
    REWARD_MODES
)

SEEDS = [42, 123, 456]
EPISODES = 3000
STEPS_PER_EPISODE = 200
EVAL_EPISODES = 20
EVAL_STEPS = 200

RESULTS_DIR = Path("results/data")


def build_toll_cache():
    """
    Precompute tolls for all 81 action profiles
    and both background-load levels.

    This avoids recalculating counterfactual
    network costs during every training step.
    """
    reference = build_reference_games()
    cache = {}

    for beta_state, data in reference.items():
        beta = BETA_VALUES[beta_state]

        for actions in data["game"]:
            cache[(beta_state, actions)] = (
                calculate_externality_tolls(actions, beta)
            )

    return cache


def learning_rewards(mode, selfish_rewards, tolls):
    """Transform environment rewards for training."""

    if mode == "selfish":
        return tuple(selfish_rewards)

    if mode == "team":
        team_reward = sum(selfish_rewards)
        return (team_reward,) * 4

    if mode == "tolled":
        return tuple(
            selfish_rewards[i] - tolls[i]
            for i in range(4)
        )

    raise ValueError(f"Unknown reward mode: {mode}")


def train_cooperation(
    mode,
    seed=42,
    episodes=EPISODES,
    steps_per_episode=STEPS_PER_EPISODE,
    toll_cache=None
):
    """Train four agents under one reward scheme."""

    if mode not in REWARD_MODES:
        raise ValueError("Invalid reward mode.")

    agents = create_agents(seed)
    env = RoutingEnvironment(seed=seed + 1000)

    if toll_cache is None:
        toll_cache = (
            build_toll_cache() if mode == "tolled" else {}
        )

    history = []

    for ep in range(episodes):
        state = env.reset()

        total_social_cost = 0.0
        total_switches = 0

        for t in range(steps_per_episode):
            beta_state = state[0]

            # All agents act from the same current state.
            actions = tuple(
                agent.choose_action(state)
                for agent in agents
            )

            next_state, selfish_rewards, info = env.step(
                actions
            )

            tolls = (
                toll_cache[(beta_state, actions)]
                if mode == "tolled"
                else (0.0,) * 4
            )

            rewards = learning_rewards(
                mode, selfish_rewards, tolls
            )

            # Every agent updates only its own Q-table.
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

        for agent in agents:
            agent.epsilon = max(
                agent.min_epsilon,
                agent.epsilon * agent.epsilon_decay
            )

        history.append({
            "mode": mode,
            "seed": seed,
            "episode": ep + 1,
            "mean_social_cost": (
                total_social_cost / steps_per_episode
            ),
            "switch_rate": (
                total_switches / (4 * steps_per_episode)
            ),
            "epsilon": agents[0].epsilon
        })

        if (ep + 1) % 500 == 0:
            print(
                f"{mode.upper()} | Seed {seed} | "
                f"Episode {ep + 1} | "
                f"Social cost "
                f"{history[-1]['mean_social_cost']:.5f}"
            )

    return agents, history


def tolled_stage_regret(actions, beta_state, reference, toll_cache):
    """
    Immediate deviation incentive under tolled costs.
    Excludes switching costs and future value.
    """

    game = reference[beta_state]["game"]
    current_costs = game[actions]["agent_costs"]
    current_tolls = toll_cache[(beta_state, actions)]

    regrets = []

    for i in range(4):
        current = current_costs[i] + current_tolls[i]
        best = current

        for route in range(3):
            alternative = list(actions)
            alternative[i] = route
            alternative = tuple(alternative)

            cost = (
                game[alternative]["agent_costs"][i]
                + toll_cache[(beta_state, alternative)][i]
            )

            best = min(best, cost)

        regrets.append(max(0.0, current - best))

    return max(regrets)


def evaluate_cooperation(
    agents,
    seed=999,
    episodes=EVAL_EPISODES,
    steps_per_episode=EVAL_STEPS,
    reference=None,
    toll_cache=None
):
    """
    Evaluate the four greedy policies without exploration.
    All modes use the same original network metrics.
    """

    if reference is None:
        reference = build_reference_games()

    if toll_cache is None:
        toll_cache = build_toll_cache()

    env = RoutingEnvironment(seed=seed)

    social_costs = []
    efficiency_ratios = []
    selfish_regrets = []
    tolled_regrets = []
    switches = []

    low_costs = []
    high_costs = []

    for ep in range(episodes):
        state = env.reset()

        for t in range(steps_per_episode):
            beta_state = state[0]

            actions = tuple(
                agent.greedy_action(state)
                for agent in agents
            )

            next_state, rewards, info = env.step(actions)

            game = reference[beta_state]["game"]
            optimum = reference[beta_state]["optimum_cost"]

            cost = info["social_cost"]

            social_costs.append(cost)
            efficiency_ratios.append(cost / optimum)

            selfish_regrets.append(
                stage_game_regret(
                    actions, BETA_VALUES[beta_state], game
                )
            )

            tolled_regrets.append(
                tolled_stage_regret(
                    actions, beta_state,
                    reference, toll_cache
                )
            )

            switches.append(sum(info["switching"]) / 4)

            if beta_state == 0:
                low_costs.append(cost)
            else:
                high_costs.append(cost)

            state = next_state

    return {
        "mean_social_cost": mean(social_costs),
        "mean_efficiency_ratio": mean(efficiency_ratios),
        "mean_stage_regret": mean(selfish_regrets),
        "mean_tolled_stage_regret": mean(tolled_regrets),
        "switch_rate": mean(switches),
        "low_load_cost": (
            mean(low_costs) if low_costs else None
        ),
        "high_load_cost": (
            mean(high_costs) if high_costs else None
        )
    }


def save_csv(filename, rows):
    """Save results that can be verified independently."""

    if not rows:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def main():
    reference = build_reference_games()
    toll_cache = build_toll_cache()

    all_history = []
    all_results = []

    for mode in REWARD_MODES:
        for seed in SEEDS:

            print("\n" + "=" * 55)
            print(f"MODE: {mode.upper()} | SEED: {seed}")
            print("=" * 55)

            agents, history = train_cooperation(
                mode=mode,
                seed=seed,
                toll_cache=toll_cache
            )

            metrics = evaluate_cooperation(
                agents,
                seed=seed + 2000,
                reference=reference,
                toll_cache=toll_cache
            )

            print("Evaluation:", metrics)

            all_history.extend(history)
            all_results.append({
                "mode": mode,
                "seed": seed,
                **metrics
            })

    save_csv(
        "cooperation_training.csv", all_history
    )

    save_csv(
        "cooperation_evaluation.csv", all_results
    )

    print("\n" + "=" * 55)
    print("SUMMARY ACROSS SEEDS")
    print("=" * 55)

    for mode in REWARD_MODES:
        rows = [
            r for r in all_results
            if r["mode"] == mode
        ]

        print(f"\n{mode.upper()}")

        for metric in (
            "mean_social_cost",
            "mean_efficiency_ratio",
            "mean_stage_regret",
            "mean_tolled_stage_regret",
            "switch_rate"
        ):
            values = [r[metric] for r in rows]

            print(
                f"{metric}: "
                f"{mean(values):.6f} +/- "
                f"{stdev(values):.6f}"
            )


if __name__ == "__main__":
    main()
