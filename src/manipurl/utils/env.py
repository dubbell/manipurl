import gymnasium as gym
import metaworld

import warnings


def create_environment(task, seed, max_episode_step = -1):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env = gym.make(
            "Meta-World/MT1", 
            env_name=task, 
            seed=seed, 
            max_episode_steps=max_episode_step, 
            render_mode="rgb_array", 
            camera_id=2)

    env.observation_space.low[-3:] -= 0.1
    env.observation_space.high[-3:] += 1.0

    return env