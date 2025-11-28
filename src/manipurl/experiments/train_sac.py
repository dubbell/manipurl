from manipurl.utils.logger import Logger, NoLogger
from manipurl.models.sac import SACAgent
from manipurl.utils.replay_buffer import ReplayBuffer
from manipurl.utils.env import create_environment

import gymnasium as gym
import metaworld
import torch

import datetime
from tqdm import tqdm
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from manipurl.wrappers.profiling import profile
from manipurl.experiments.run_config import RunConfig

from .eval import evaluate



@profile
def start_training(config : RunConfig):
    run_name = f"SAC_{config.task}_s{config.seed}_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = Logger(run_name) if os.environ.get("MANIPURL_ENABLE_LOGGING", 'false') == 'true' else NoLogger()
    logger.log_parameters({
        **config._asdict(),
        "algorithm": "SAC"})

    env = create_environment(config.task, config.seed, config.max_episode_step)
    eval_env = create_environment(config.task, config.seed+100, config.max_episode_step)

    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = SACAgent(state_dim, action_dim, logger=logger)
    replay_buffer = ReplayBuffer(state_dim, action_dim, min(int(1e6), config.n_episodes * config.max_episode_step))

    pb_enable = os.environ.get('MANIPURL_ENABLE_PB', 'false').lower() == 'true'
    if pb_enable:
        pb = tqdm(total = config.n_episodes)

    episode_count = 0
    total_step = 0

    with torch.random.fork_rng():
        torch.manual_seed(config.seed)
        while episode_count <= config.n_episodes:
            terminate, truncate = False, False
            env_step = 0
            success = 0

            obs, _ = env.reset()

            while not terminate and not truncate:
                action = agent.sample_action(obs).cpu().numpy()

                next_obs, _, terminate, truncate, info = env.step(action)
                sparse_reward = info["success"]
                
                replay_buffer.insert(
                    obs, 
                    next_obs,
                    action,
                    sparse_reward-1,
                    terminate)
                
                if total_step >= config.start_training:
                    agent.train_step(replay_buffer.sample(256))
                
                if sparse_reward > 0:
                    success = 1
                
                obs = next_obs
                
                env_step += 1
                total_step += 1
                
            logger.log_metrics({
                "success": success,
                "episode_length": env_step})

            if episode_count % config.eval_freq == 0:
                evaluate(eval_env, agent, config.eval_eps, logger)

            logger.increment()
            if pb_enable:
                pb.update(1)
            episode_count += 1
    
        if pb_enable:
            pb.close()

    logger.stop()
    env.close()
    
    return run_name

