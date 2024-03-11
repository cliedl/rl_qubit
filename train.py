from environment import QubitEnv
from plot_utils import render_episode

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from tqdm.auto import tqdm


def env_creator(env_config):
    return QubitEnv()  # return an instance of your environment


# Replace 'QubitEnv-v0' with your desired environment ID
register_env('QubitEnv-v0', env_creator)


ray.init()  # Initialize Ray; include necessary arguments based on your setup


algo = (
    PPOConfig()
    .environment(env="QubitEnv-v0")
    .build()
)

for i in tqdm(range(30)):
    result = algo.train()
    print(f"Mean episode reward: {result['episode_reward_mean']}")

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")


states_episode = []
env = QubitEnv()
terminated = False
while not terminated:
    action = algo.compute_single_action(env.state)
    state, reward, terminated, truncated, info = env.step(action)
    states_episode.append(state)

render_episode(states_episode, delay=0.01)
