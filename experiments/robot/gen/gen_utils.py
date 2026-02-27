"""
gen_utils.py

Utility functions for evaluating VLA policies in Gen (robosuite) simulation environments.
Mirrors libero_utils.py but adapted for robosuite PouringWater and similar custom environments.
"""

import math
import os

import imageio
import numpy as np

from experiments.robot.robot_utils import DATE, DATE_TIME


def get_gen_image(obs):
    """
    Extract third-person (agentview) image from robosuite observation dict.

    robosuite renders images with OpenGL coordinate system (origin at bottom-left),
    so we flip vertically to standard image coordinates. The horizontal flip
    matches the 180° rotation used in LIBERO preprocessing for training distribution.
    """
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # 180° rotation (same as LIBERO preprocessing)
    return img


def get_gen_wrist_image(obs):
    """
    Extract wrist camera image from robosuite observation dict.
    """
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]
    return img


def get_gen_dummy_action():
    """Return a no-op action for robosuite OSC_POSE controller (7-dim)."""
    return [0, 0, 0, 0, 0, 0, -1]


def quat2axisangle(quat):
    """
    Convert quaternion (x, y, z, w) to axis-angle representation.

    Copied from robosuite transform_utils for standalone usage.
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, save_version=None):
    """Save an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{save_version}/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path
