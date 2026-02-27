"""
run_gen_eval.py

Evaluates a trained VLA-Adapter policy in Gen (robosuite) simulation environments
such as PouringWater.

The simulation environment is created using the same robosuite infrastructure as
Gen/scripts/gen.py (DataCollectionConfig + RoboSuiteDataCollector), ensuring that
evaluation uses the exact same environment dynamics, controller, camera setup,
and object configuration as training data generation.

The model inference pipeline is identical to the one used in run_libero_eval.py
(VLA-Adapter model loading, action chunking, open-loop execution, etc.).

Usage:
    cd /home/ljc/Git/Gen_VLA_Adapter

    CUDA_VISIBLE_DEVICES=0 python experiments/robot/gen/run_gen_eval.py \
        --pretrained_checkpoint <checkpoint_path> \
        --env_config Gen/configs/examples/pouring_water_trajgen.json \
        --task_suite_name pouringwater_generated \
        --num_trials_per_task 50
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import draccus
import numpy as np
import tqdm

import wandb

# Ensure project root is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# --- Gen infrastructure (same as gen.py) ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Gen"))
from configs.config import DataCollectionConfig
from env_interfaces.robosuite_env import RoboSuiteDataCollector

# --- VLA-Adapter inference utilities (same as run_libero_eval.py) ---
from experiments.robot.gen.gen_utils import (
    get_gen_dummy_action,
    get_gen_image,
    get_gen_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GenEvalConfig:
    # fmt: off

    ###########################################################################
    # Model-specific parameters (same as run_libero_eval.py)
    ###########################################################################
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_minivlm: bool = True
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True

    center_crop: bool = True
    num_open_loop_steps: int = 8
    unnorm_key: Union[str, Path] = ""

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    ###########################################################################
    # Gen environment parameters
    ###########################################################################
    env_config: str = "Gen/configs/examples/pouring_water_trajgen.json"
    task_suite_name: str = "pouringwater_generated"     # Used as unnorm_key
    task_description: str = ""                          # Overrides language_instruction from config if set
    num_trials_per_task: int = 50                       # Number of rollouts
    num_steps_wait: int = 10                            # Steps to wait for objects to stabilize
    max_steps: int = 600                                # Max steps per episode
    env_img_res: int = 256                              # Env camera resolution

    ###########################################################################
    # Utils
    ###########################################################################
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    seed: int = 7

    save_version: str = "vla-adapter"
    use_pro_version: bool = True
    phase: str = "Inference"

    # fmt: on


# =============================================================================
# Helper functions
# =============================================================================


def validate_config(cfg: GenEvalConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty!"
    assert os.path.exists(cfg.env_config), f"env_config not found: {cfg.env_config}"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, (
            "Expecting `center_crop==True` because model was trained with image augmentations!"
        )
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), (
        "Cannot use both 8-bit and 4-bit quantization!"
    )


def initialize_model(cfg: GenEvalConfig):
    """Initialize model and associated components (identical to run_libero_eval.py)."""
    model = get_model(cfg)
    model.set_version(cfg.save_version)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dim: eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
        )

    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenEvalConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, (
        f"Action un-norm key '{unnorm_key}' not found in VLA norm_stats! "
        f"Available keys: {list(model.norm_stats.keys())}"
    )
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenEvalConfig):
    """Set up logging to file and optionally to wandb."""
    run_id = f"EVAL-Gen-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def create_gen_env(cfg: GenEvalConfig):
    """
    Create robosuite environment using the same pipeline as gen.py.

    Reads the JSON config used for data generation (DataCollectionConfig),
    builds the robosuite env via RoboSuiteDataCollector, and returns
    the env wrapper + task description.
    """
    # Load same config as gen.py
    gen_config = DataCollectionConfig.from_json(cfg.env_config)

    # Override rendering: we need offscreen renderer for camera obs but no GUI
    gen_config.has_renderer = False
    gen_config.has_offscreen_renderer = True
    gen_config.use_camera_obs = True

    # Override resolution if specified
    gen_config.camera_heights = cfg.env_img_res
    gen_config.camera_widths = cfg.env_img_res

    # Create environment (same path as gen.py)
    env = RoboSuiteDataCollector(gen_config)

    # Task description: use user override or fallback to config language_instruction
    task_description = cfg.task_description if cfg.task_description else gen_config.language_instruction

    return env, task_description, gen_config


def prepare_observation(obs, resize_size):
    """
    Prepare robosuite observation for VLA policy input.

    Maps the raw robosuite observation dict to the standard VLA input format:
      - full_image:   agentview camera (resized)
      - wrist_image:  robot0_eye_in_hand camera (resized)
      - state:        [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
    """
    # Extract and preprocess images (180° rotation to match training preprocessing)
    img = get_gen_image(obs["raw_obs"])
    wrist_img = get_gen_wrist_image(obs["raw_obs"])

    # Resize to model input resolution
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Build proprio state: eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
    raw = obs["raw_obs"]
    eef_pos = raw["robot0_eef_pos"]
    eef_axisangle = quat2axisangle(raw["robot0_eef_quat"].copy())
    gripper_qpos = raw["robot0_gripper_qpos"]
    state = np.concatenate([eef_pos, eef_axisangle, gripper_qpos])

    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": state,
    }

    return observation, img  # Return original img for replay video


def process_action(action, model_family):
    """Process action before sending to environment (same as run_libero_eval.py)."""
    # Normalize gripper action [0,1] -> [-1,+1]
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] Flip gripper sign back: dataloader uses 0=close,1=open
    # but env expects -1=open,+1=close
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


# =============================================================================
# Episode / evaluation loop
# =============================================================================


def run_episode(
    cfg: GenEvalConfig,
    env: RoboSuiteDataCollector,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    log_file=None,
):
    """Run a single evaluation episode in the robosuite environment."""
    # Reset environment (randomises object positions, same as gen.py)
    obs = env.reset()

    # Wait for objects to stabilize (same as LIBERO eval)
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_gen_dummy_action())

    # Action queue for open-loop execution (action chunking)
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        logger.warning(
            f"num_open_loop_steps ({cfg.num_open_loop_steps}) != NUM_ACTIONS_CHUNK "
            f"({NUM_ACTIONS_CHUNK}). For best performance, execute the full action chunk."
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images = []
    success = False

    try:
        while t < cfg.max_steps:
            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # Requery model when action queue is empty
            if len(action_queue) == 0:
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm,
                )
                action_queue.extend(actions)

            # Pop action and process
            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)

            # Step environment
            obs, reward, done, info = env.step(action.tolist())

            # Check success via environment's _check_success (same as gen.py)
            if hasattr(env.unwrapped, "_check_success") and env.unwrapped._check_success():
                success = True
                break
            if done:
                break

            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images


@draccus.wrap()
def eval_gen(cfg: GenEvalConfig) -> float:
    """Main function: evaluate a trained VLA-Adapter policy in Gen robosuite environments."""

    # Validate
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    # Initialize model & components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Create robosuite environment (same pipeline as gen.py)
    env, task_description, gen_config = create_gen_env(cfg)

    log_message(f"Environment: {gen_config.env_name}", log_file)
    log_message(f"Robot: {gen_config.robots}", log_file)
    log_message(f"Controller: {gen_config.controller_type or 'default'}", log_file)
    log_message(f"Task: {task_description}", log_file)
    log_message(f"Num trials: {cfg.num_trials_per_task}", log_file)
    log_message(f"Max steps per episode: {cfg.max_steps}", log_file)
    log_message(f"Checkpoint: {cfg.pretrained_checkpoint}", log_file)

    # Run evaluation
    total_episodes = 0
    total_successes = 0

    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc="Evaluating"):
        log_message(f"\nEpisode {episode_idx + 1}/{cfg.num_trials_per_task}", log_file)
        log_message(f"Task: {task_description}", log_file)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            log_file,
        )

        total_episodes += 1
        if success:
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images,
            total_episodes,
            success=success,
            task_description=task_description,
            log_file=log_file,
            save_version=cfg.save_version,
        )

        # Log results
        current_rate = total_successes / total_episodes * 100
        log_message(f"Success: {success}", log_file)
        log_message(
            f"Progress: {total_episodes} episodes | "
            f"{total_successes} successes ({current_rate:.1f}%)",
            log_file,
        )

    # Close environment
    env.close()

    # Final results
    final_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    log_message("=" * 60, log_file)
    log_message("Final Results", log_file)
    log_message("=" * 60, log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(
        f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)",
        log_file,
    )

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_gen()
