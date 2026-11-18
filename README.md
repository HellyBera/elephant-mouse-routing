
# Learning to Route Under Conflicting QoS Objectives

## Elephant and Mouse Agents in a Shared Congested Network

A multi-agent reinforcement learning and game-theoretic
study of congestion-aware routing under conflicting
Quality of Service (QoS) objectives.

## Project Overview

We study a shared network containing four autonomous agents:

- Two elephant agents prioritizing throughput.
- Two mouse agents prioritizing low latency.

Each agent selects one of three parallel network routes.
Agents share limited network capacity, creating congestion
externalities and conflicts between individual incentives
and socially efficient routing.

We analyze strategic routing using game theory, exact
dynamic programming, model-free reinforcement learning,
and cooperative incentive mechanisms.

## Network Configuration

| Parameter | Value |
|---|---|
| Number of agents | 4 |
| Available routes | 3 |
| Route capacities | 50, 100, 40 Mb/s |
| Base delays | 2, 10, 5 ms |
| Agent demands | 20, 20, 6, 6 Mb/s |
| Background loads | 10%, 25% |
| Discount factor | 0.9 |
| Switching penalty | 0.02 |
| Markov states | 162 |

The background traffic follows a two-state Markov process.

## Implemented Methods

### Game Theory
- Pure Nash equilibrium enumeration
- Social optimum
- Price of Anarchy and Price of Stability
- Sequential move-order analysis and pure SPNE
- Numerical mixed-equilibrium computation using PyGambit

### Reinforcement Learning
- Exact policy iteration
- Monte Carlo control
- Q-learning
- Simultaneous independent multi-agent Q-learning

### Cooperation
- Selfish rewards
- Shared team rewards
- Externality-based congestion tolls

### Repeated Games
- Pure Nash-reversion punishment analysis
- Pure-action minmax punishment benchmarks
- Conditional discount-factor incentive thresholds

## Installation

Clone the repository:

```bash
git clone https://github.com/HellyBera/elephant-mouse-routing.git
cd elephant-mouse-routing
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running Tests

```bash
python -m unittest discover -s tests -v
```

## Running Experiments

Stage-game analysis:

```bash
python src/game_theory.py
```

Sequential-game analysis:

```bash
python src/sequential_game.py
```

Mixed equilibrium:

```bash
python src/mixed_equilibrium.py
```

Exact policy iteration:

```bash
python src/policy_iteration.py
```

Monte Carlo control:

```bash
python src/monte_carlo.py
```

Q-learning:

```bash
python src/q_learning.py
```

Simultaneous multi-agent learning:

```bash
python src/multi_agent.py
```

Cooperation experiments:

```bash
python src/cooperation.py
```

Nash-reversion punishment:

```bash
python src/repeated_game.py
```

Pure minmax analysis:

```bash
python src/minmax_punishment.py
```

RL algorithm comparison:

```bash
python scripts/compare_rl_algorithms.py
```

Generate report figures and tables:

```bash
python scripts/generate_all_results.py
```

## Selected Results

### Independent Multi-Agent Q-Learning

Mean social efficiency ratio across three seeds: 1.4479.

### Cooperation Experiments

| Reward mechanism | Mean social cost | Efficiency ratio |
|---|---:|---:|
| Selfish | 0.177140 | 1.447887 |
| Team | 0.144301 | 1.173754 |
| Externality toll | 0.153598 | 1.250589 |

Team rewards achieved the lowest average social cost
across the three evaluated seeds.

### Repeated-Game Analysis

No tested pure Nash-reversion punishment plan sustained
the social optimum.

Pure-action minmax punishment produced conditional
incentive thresholds of:

- Low background load: 0.831169
- High background load: 0.816667

These are incentive benchmarks, not established
subgame-perfect equilibria.

## Results and Figures

Experimental data is stored in `results/data/`.

Publication figures are stored in `results/figures/`.

Summary tables are stored in `results/tables/`.

## Reproducibility

Experiments use specified random seeds and fixed
configurations. The numerical results can be regenerated
using the scripts above.

## Contributors

- Het Bera
- Divisht
- Anshul

See the final report for individual contributions.

## Notes

The mixed-equilibrium numerical solver may return a
limited number of equilibria depending on the configured
enumeration limit. The output should not be interpreted
as exhaustive unless enumeration completes.

The repeated-game punishment analysis uses fixed-load
stage-game approximations and does not establish a full
stochastic-game equilibrium.
