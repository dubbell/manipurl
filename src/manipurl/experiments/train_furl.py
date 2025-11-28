from manipurl.utils.env import create_environment
from manipurl.utils.logger import Logger, NoLogger
from manipurl.utils.replay_buffer import ReplayBuffer
from manipurl.wrappers.profiling import profile
from manipurl.experiments.run_config import RunConfig
from manipurl.models.sac import SACAgent

import os
import datetime



@profile
def start_training(config : RunConfig):
    run_name = f"FURL_{config.task}_s{config.seed}_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = Logger(run_name) if os.environ.get("MANIPURL_ENABLE_LOGGING", 'false') == 'true' else NoLogger()

    logger.log_parameters({
        **config._asdict(),
        "algorithm": "FURL"})
    
    env = create_environment(config.task, config.seed, config.max_episode_step)
    eval_env = create_environment(config.task, config.seed+100, config.max_episode_step)

    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]

    sac_agent = SACAgent(state_dim, action_dim, logger=logger)
    furl_agent = SACAgent(state_dim, action_dim, logger=logger)
    
    replay_buffer = ReplayBuffer(state_dim, action_dim, min(int(1e6), config.n_episodes * config.max_episode_step))

