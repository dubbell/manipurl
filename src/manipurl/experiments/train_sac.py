from manipurl.utils.logger import Logger
from manipurl.models.sac import SACAgent
from manipurl.utils.replay_buffer import ReplayBuffer

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


def evaluate(config : RunConfig, env : gym.Env, agent : SACAgent, logger : Logger):
    pinned_action_buffer = torch.empty(agent.act_dim, dtype=torch.float32, pin_memory = True)

    for _ in range(config.eval_eps):
        terminate, truncate = False, False
        success = 0
        obs, _ = env.reset()
        while not terminate and not truncate:
            gpu_action = agent.sample_action(obs)
            pinned_action_buffer.copy_(gpu_action, non_blocking=True)
            torch.cuda.synchronize()
            action = pinned_action_buffer.numpy()

            obs, _, terminate, truncate, info = env.step(action)
            if info['success'] > 0:
                success = 1
        
        logger.log_metric("eval_succes", success)
            

@profile
def start_training(config : RunConfig):
    run_name = f"SAC_{config.task}_s{config.seed}_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = Logger(run_name)
    logger.log_parameters({
        **config._asdict(),
        "algorithm": "SAC"})

    env = gym.make("Meta-World/MT1", env_name=config.task, seed=config.seed, max_episode_steps=config.max_episode_step)
    eval_env = gym.make("Meta-World/MT1", env_name=config.task, seed=config.seed+100, max_episode_steps=config.max_episode_step)

    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = SACAgent(state_dim, action_dim, logger=logger)
    replay_buffer = ReplayBuffer(state_dim, action_dim, min(int(1e6), config.n_episodes * config.max_episode_step))

    pb_enable = os.environ['MANIPURL_ENABLE_LOGGING']
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
                evaluate(eval_env, agent, logger)

            logger.increment()
            if pb_enable:
                pb.update(1)
            episode_count += 1
    
        if pb_enable:
            pb.close()

    logger.stop()
    env.close()

