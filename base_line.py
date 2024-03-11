from environment import QubitEnv
from plot_utils import render_episode
import numpy as np


# initialize state in ground state
env = QubitEnv(
    initial_state=np.array([0., 0., 0.]),
    target_state=np.array([1., 0., 0.]),
    dt=0.001
)
state = env.reset()

# Chose maximum field
Omega = 20
time_pi_pulse = np.pi/Omega  # Time for pi

Delta = 0  # detuning is zero, on resonance
action = [Omega, Delta, 0]  # Apply max field on resonance

states_epsiode = []
rewards_episode = []
i = 0
terminated = False

# Apply max field until time of pi-pulse has been reached, then stop
while not terminated:
    time_passed = i*env.dt
    if time_passed >= time_pi_pulse:
        action[2] = 0.8
    state, reward, terminated, truncated, info = env.step(
        action)  # Apply action
    states_epsiode.append(state)
    rewards_episode.append(reward)
    i += 1


print(f"Total reward: {sum(rewards_episode)}")

render_episode(states_epsiode, delay=0.01)
