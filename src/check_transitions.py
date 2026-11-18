
from environment import RoutingEnvironment

env = RoutingEnvironment(seed=42)
env.reset()

steps = 100000
high_count = 0

for _ in range(steps):
    state, rewards, info = env.step((0, 1, 0, 2))

    if state[0] == 1:
        high_count += 1

print("Observed high-load fraction:", high_count / steps)
print("Theoretical high-load fraction:", 1 / 3)
