import os
import cv2
import warnings

import numpy as np

from .env import create_environment

from metaworld.policies import (
    SawyerButtonPressV3Policy,
    SawyerDoorOpenV3Policy,
    SawyerDrawerCloseV3Policy,
    SawyerDrawerOpenV3Policy,
    SawyerPegInsertionSideV3Policy,
    SawyerPickPlaceV3Policy,
    SawyerPushV3Policy,
    SawyerReachV3Policy,
    SawyerWindowOpenV3Policy,
    SawyerWindowCloseV3Policy)


ORACLE_POLICIES = {
    "button-press-v3" : SawyerButtonPressV3Policy,
    "door-open-v3" : SawyerDoorOpenV3Policy,
    "drawer-close-v3" : SawyerDrawerCloseV3Policy,
    "drawer-open-v3" : SawyerDrawerOpenV3Policy,
    "peg-insert-side-v3" : SawyerPegInsertionSideV3Policy,
    "pick-place-v3" : SawyerPickPlaceV3Policy,
    "push-v3" : SawyerPushV3Policy,
    "reach-v3" : SawyerReachV3Policy,
    "window-open-v3" : SawyerWindowOpenV3Policy,
    "window-close-v3" : SawyerWindowCloseV3Policy
}



def write_video(task, frames):
    os.makedirs('data/videos', exist_ok=True)

    video_filename = f'data/videos/{task}.mp4'
    fps = 30.0
    height, width, _ = frames[0].shape
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()

    return video_filename

def write_goal_image(task, goal_frame):
    os.makedirs('data/goals', exist_ok=True)
    goal_image_filename = f'data/goals/{task}.png'
    cv2.imwrite(goal_image_filename, goal_frame)
    return goal_image_filename


def generate_demonstration(task, seed = 0):
    env = create_environment(task, seed)
    oracle_policy = ORACLE_POLICIES[task]()

    frames = []
    success = False
    obs, _ = env.reset(seed=seed)

    print(f"Generating oracle demonstration for {task}...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        while not success:
            action = oracle_policy.get_action(obs)
            obs, _, terminate, truncate, info = env.step(action)
            success = info['success'] > 0

            frame = env.render()[::-1,:,::-1]

            frames.append(frame)

            if terminate or truncate:
                obs, _ = env.reset(seed)
                frames = []
                continue
    
    print("Trajectory generated. Writing goal image...")
    goal_image_filename = write_goal_image(task, frames[-1])
    print(f"Goal image file {goal_image_filename} created. Writing trajectory to video file...")
    video_filename = write_video(task, frames)
    print(f"Video file {video_filename} created.")
