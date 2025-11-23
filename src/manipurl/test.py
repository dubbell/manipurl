import numpy as np
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import ReachTarget
from rlbench.observation_config import ObservationConfig, CameraConfig
import matplotlib.pyplot as plt

# Initialize action mode
action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(),
    gripper_action_mode=Discrete()
)

obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.joint_positions = True
obs_config.gripper_open = True
obs_config.task_low_dim_state = True
obs_config.front_camera = CameraConfig(
    rgb=True,
    depth=False,
    point_cloud=False,
    mask=False,
    image_size=(512, 512)
)

# Create and launch environment
print("test1")
env = Environment(
    action_mode, 
    obs_config=obs_config,
    headless=True)
print("test2")
env.launch()
print("test3")

# Load a task
task = env.get_task(ReachTarget)
print("test4")

# Reset the task and get initial observations
descriptions, obs = task.reset()

# Take a random action
print("test5")
obs, reward, terminate = task.step(np.random.normal(size=env.action_shape))

env.shutdown()

print(obs.front_rgb.shape)
plt.imshow(obs.front_rgb)
plt.show()