from manipurl.utils.logger import Logger
from manipurl.models.sac import SACAgent
from manipurl.utils.tasks import TASKS
from manipurl.utils.replay_buffer import ReplayBuffer

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import CameraConfig, ObservationConfig

import datetime
from tqdm import tqdm


MAX_ENV_STEP = 350
START_TRAINING = 2500



def get_obs_config():
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    obs_config.task_low_dim_state = True
    # obs_config.front_camera = CameraConfig(
    #     rgb=True,
    #     depth=False,
    #     point_cloud=False,
    #     mask=False,
    #     image_size=(512, 512))
    
    return obs_config



def start_training(task_name, num_episodes, seed = 0):
    logger = Logger("SAC_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
    
    env = Environment(action_mode, obs_config=get_obs_config(), headless=True)
    env.launch()
    task_env = env.get_task(TASKS[task_name])
    # task_env.set_variation(0)

    episode_count = 0
    _, obs = task_env.reset()

    state_dim, action_dim = obs.get_low_dim_data().shape[0], env.action_shape[0]
    agent = SACAgent(state_dim, action_dim, logger=logger)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    total_step = 0
    pb = tqdm(total = num_episodes)

    while episode_count < num_episodes:
        terminate = False
        env_step = 0
        success = 0

        while not terminate and env_step < MAX_ENV_STEP:
            action = agent.sample_action(obs.get_low_dim_data()).detach().cpu().numpy()
            next_obs, task_reward, terminate = task_env.step(action)
            
            replay_buffer.insert(
                obs.get_low_dim_data(), 
                next_obs.get_low_dim_data(),
                action,
                task_reward-1,
                terminate)
            
            if total_step >= START_TRAINING:
                agent.train_step(replay_buffer.sample(256))
            
            if task_reward > 0:
                success = 1
            obs = next_obs
            
            env_step += 1
            total_step += 1
            
        logger.log_metrics({
            "success": success,
            "episode_length": env_step})

        logger.increment()
        pb.update(1)
        episode_count += 1

        _, obs = task_env.reset()

    logger.stop()
    pb.close()
    env.shutdown()