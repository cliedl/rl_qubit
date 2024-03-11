from environment import QubitEnv
from plot_utils import render_episode

# Experiment 1: Initialize in excited state and do nothing
dt = 0.01
length_episode = 3  # in units of natural lifetime
env = QubitEnv(initial_state=[1, 0, 0], target_state=[1, 0, 0], dt=dt)
state = env.reset()

states_epsiode = []
for _ in range(int(length_episode/dt)):
    action = [0, 0, 0]  # Do nothing
    state, reward, done, _ = env.step(action)  # Apply the action
    states_epsiode.append(state)

render_episode(states_epsiode, delay=0.001)


# Experiment 2: Initialize in ground state and apply resonant field
dt = 0.01
length_episode = 3  # in units of natural lifetime
env = QubitEnv(initial_state=[0, 0, 0], target_state=[1, 0, 0], dt=dt)
state = env.reset()

states_epsiode = []
for _ in range(int(length_episode/dt)):
    action = [10, 0, 0]  # Resonant field
    state, reward, done, _ = env.step(action)  # Apply the action
    states_epsiode.append(state)

render_episode(states_epsiode, delay=0.001)
