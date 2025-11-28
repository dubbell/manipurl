import torch
import gymnasium as gym

from manipurl.utils.logger import Logger
from manipurl.models.sac import SACAgent


def evaluate(env : gym.Env, agent : SACAgent, eval_eps : int, logger : Logger):
    pinned_action_buffer = torch.empty(agent.act_dim, dtype=torch.float32, pin_memory = True)

    for _ in range(eval_eps):
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