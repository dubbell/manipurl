from manipurl.utils.logger import Logger
from manipurl.models.sac import SACAgent
from manipurl.utils.replay_buffer import ReplayBuffer

import gymnasium as gym
import metaworld
import torch

import datetime
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


MAX_ENV_STEP = 500
START_TRAINING = 2500



def start_training(task_name, num_episodes, seed = 0, pb_enable=True):
    logger = Logger("SAC_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    logger.log_parameters({
        "task": task_name,
        "seed": seed,
        "MAX_EP_LENGTH": MAX_ENV_STEP,
        "n_episodes": num_episodes,
        "algorithm": "SAC"
    })

    env = gym.make("Meta-World/MT1", env_name=task_name, seed=seed, max_episode_steps=MAX_ENV_STEP)

    episode_count = 0

    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = SACAgent(state_dim, action_dim, logger=logger)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    total_step = 0
    if pb_enable:
        pb = tqdm(total = num_episodes)

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        while episode_count < num_episodes:
            terminate, truncate = False, False
            env_step = 0
            success = 0

            obs, _ = env.reset()

            while not terminate and not truncate:
                action = agent.sample_action(obs).detach().cpu().numpy()
                next_obs, _, terminate, truncate, info = env.step(action)
                sparse_reward = info["success"]
                
                replay_buffer.insert(
                    obs, 
                    next_obs,
                    action,
                    sparse_reward-1,
                    terminate)
                
                if total_step >= START_TRAINING:
                    agent.train_step(replay_buffer.sample(256))
                
                if sparse_reward > 0:
                    success = 1
                obs = next_obs
                
                env_step += 1
                total_step += 1
                
            logger.log_metrics({
                "success": success,
                "episode_length": env_step})

            logger.increment()
            if pb_enable:
                pb.update(1)
            episode_count += 1

        logger.stop()
        if pb_enable:
            pb.close()
        env.close()