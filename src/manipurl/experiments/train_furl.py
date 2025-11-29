from manipurl.utils.env import create_environment
from manipurl.utils.logger import Logger, NoLogger
from manipurl.utils.replay_buffer import ReplayBuffer
from manipurl.utils.embedding_buffer import EmbeddingBuffer
from manipurl.wrappers.profiling import profile
from manipurl.experiments.run_config import RunConfig
from manipurl.models.sac import SACAgent
from manipurl.models.clip import project_image, project_text
from manipurl.shaping.furl import FuRLShaper

import torch
import numpy as np
import cv2

from tqdm import tqdm
import os
import datetime
import click


def get_goal_img_embedding(task):
    filepath = f'data/goals/{task}.png'
    img = cv2.imread(filepath)
    assert img is not None, f"Task goal image not found: {filepath}"
    return project_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


@profile
def start_training(config : RunConfig):
    run_name = f"FURL_{config.task}_s{config.seed}_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # logging
    logger = Logger(run_name) if os.environ.get("MANIPURL_ENABLE_LOGGING", 'false') == 'true' else NoLogger()
    logger.log_parameters({
        **config._asdict(),
        "algorithm": "FURL"})
    
    # training and evaluation environments
    env = create_environment(config.task, config.seed, config.max_episode_step)
    eval_env = create_environment(config.task, config.seed+100, config.max_episode_step)

    # goal image and task description text embeddings
    goal_img_embedding = get_goal_img_embedding(config.task)

    # furl and relay SAC agent
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    sac_agent = SACAgent(state_dim, action_dim, logger=logger)
    furl_agent = SACAgent(state_dim, action_dim, logger=logger)
    
    # replay buffer for transitions, embedding buffer for CLIP embeddings
    replay_buffer = ReplayBuffer(
        state_dim, 
        action_dim, 
        min(int(1e6), config.n_episodes * config.max_episode_step), 
        True, 
        goal_img_embedding.shape[0])
    
    embedding_buffer = EmbeddingBuffer()

    # pinned memory for embeddings for efficient transfer
    pinned_embeddings = torch.empty((config.max_episode_step, goal_img_embedding.shape[0])).pin_memory()

    # FuRL reward shaper
    furl_shaper = FuRLShaper(config.task_description)

    # progress bar
    pb_enable = os.environ.get('MANIPURL_ENABLE_PB', 'false').lower() == 'true'
    if pb_enable:
        pb = tqdm(total = config.n_episodes)

    episode_count = 0
    total_step = 0

    rng = np.random.default_rng(config.seed)

    # relay RL
    use_relay = True
    use_furl_agent = True
    relay_count = rng.choice([50, 100, 150, 200])

    with torch.random.fork_rng():
        torch.manual_seed(config.seed)
        while episode_count <= config.n_episodes:
            terminate, truncate = False, False
            ep_step = 0
            success = 0

            obs, _ = env.reset()

            while not terminate and not truncate:
                if total_step < config.start_training:
                    action = env.action_space.sample()
                else:
                    action = (furl_agent if use_furl_agent or not use_relay else sac_agent).sample_action(obs)

                next_obs, _, terminate, truncate, info = env.step(action)
                sparse_reward = info["success"]
                
                image = env.render()[::-1]
                image_embedding = project_image(image)

                replay_buffer.insert(
                    obs, 
                    next_obs,
                    action,
                    sparse_reward-1,
                    terminate,
                    image_embedding)

                pinned_embeddings[ep_step] = image_embedding
                
                if total_step >= config.start_training:
                    replay_batch = replay_buffer.sample(256)
                    vlm_rewards = 


                if use_relay:
                    relay_count -= 1
                    if relay_count <= 0:
                        relay_count = rng.choice([50, 100, 150, 200])
                        use_furl_agent = not use_furl_agent