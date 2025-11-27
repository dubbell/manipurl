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

import cProfile
import os


START_TRAINING = 2500
EVAL_FREQ = 50
EVAL_EPS = 25


def evaluate(env : gym.Env, agent : SACAgent, logger : Logger):
    pinned_action_buffer = torch.empty(agent.act_dim, dtype=torch.float32, pin_memory = True)

    for _ in range(EVAL_EPS):
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
            

def start_training(task_name, seed, n_episodes, max_episode_step, pb_enable=True):
    run_name = f"SAC_{task_name}_s{seed}_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = Logger(run_name)
    logger.log_parameters({
        "task": task_name,
        "seed": seed,
        "max_episode_step": max_episode_step,
        "n_episodes": n_episodes,
        "algorithm": "SAC"})

    env = gym.make("Meta-World/MT1", env_name=task_name, seed=seed, max_episode_steps=max_episode_step)
    eval_env = gym.make("Meta-World/MT1", env_name=task_name, seed=seed+100, max_episode_steps=max_episode_step)

    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = SACAgent(state_dim, action_dim, logger=logger)
    replay_buffer = ReplayBuffer(state_dim, action_dim, min(int(1e6), n_episodes * max_episode_step))
    
    profiler = cProfile.Profile()
    profiler.enable()

    if pb_enable:
        pb = tqdm(total = n_episodes)

    episode_count = 0
    total_step = 0

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        while episode_count <= n_episodes:
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

            if episode_count % EVAL_FREQ == 0:
                evaluate(eval_env, agent, logger)

            logger.increment()
            if pb_enable:
                pb.update(1)
            episode_count += 1
    
        if pb_enable:
            pb.close()

    logger.stop()
    env.close()

    profiler.disable()
    os.makedirs("data/profiling", exist_ok=True)
    profiler.dump_stats(f"data/profiling/{run_name}.prof")

