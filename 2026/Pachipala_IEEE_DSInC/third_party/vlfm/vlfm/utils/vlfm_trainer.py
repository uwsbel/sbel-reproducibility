# ---------------------------------------------------------------------------------
# Task: VLFMTrainer
# - Evaluate a Proximal Policy Optimization(PPO)-based Habitat policy on a dataset split using vectorized envs.
# - Optionally load a checkpoint, render videos/GFX replays, and log metrics.
# - Carefully manages recurrent state, previous actions, and env pausing.
# ---------------------------------------------------------------------------------

import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    extract_scalars_from_info as extract_scalars_from_info_habitat,
)
from omegaconf import OmegaConf


# Helper: sanitize info dicts by removing list entries (Habitat's scalar extractor
# doesn't expect lists) and then delegate to the built-in scalar extractor.
def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    info_filtered = {k: v for k, v in info.items() if not isinstance(v, list)}
    return extract_scalars_from_info_habitat(info_filtered)


# Register a custom trainer under the name "vlfm" so it can be created via config.
@baseline_registry.register_trainer(name="vlfm")
class VLFMTrainer(PPOTrainer):
    # Hint to readers and type checkers: the trainer manages a vectorized env.
    envs: VectorEnv

    # Core evaluation entry point: run policy inference over eval episodes,
    # optionally loading a checkpoint and writing videos/metrics.
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # Distributed evaluation isn’t supported by this routine (single-process only).
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Optionally load model weights. Some configs *intentionally* skip loading
        # (e.g., hierarchical policy setups) and only reuse the architecture.
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # map_location="cpu" ensures compatibility and avoids CUDA contexts
            # inside env subprocesses.
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            # Minimal placeholder if we’re not loading weights.
            ckpt_dict = {"config": None}

        # Merge the runtime/ckpt config into an effective evaluation config.
        config = self._get_resume_state_config_or_new_config(ckpt_dict["config"])

        # Ensure the dataset split (e.g., "val" or "test") is set for evaluation.
        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        # If any video option is enabled, wire up additional render sensors and
        # make sure their outputs flow through observations.
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
            with read_write(agent_sensors):
                agent_sensors.update(extra_sensors)
            with read_write(config):
                if config.habitat.gym.obs_keys is not None:
                    for render_view in extra_sensors.values():
                        if render_view.uuid not in config.habitat.gym.obs_keys:
                            config.habitat.gym.obs_keys.append(render_view.uuid)
                # Turn on debug_render so frames are produced for video export.
                config.habitat.simulator.debug_render = True

        # Optional: print the full resolved config (useful for debugging).
        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        # Create vectorized environments in evaluation mode.
        self._init_envs(config, is_eval=True)

        # Build the agent and inspect the action-space traits (shape/type).
        self._agent = self._create_agent(None)
        action_shape, discrete_actions = get_action_space_info(self._agent.policy_action_space)

        # Load model parameters if the policy is expecting them.
        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)

        # Reset envs to get initial observations and apply any obs transforms
        # (resize/normalize/etc.) configured for this run.
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        # Allocate per-env accumulators/buffers used during rollout.
        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        # Recurrent hidden states buffer (if policy is RNN/LSTM/GRU).
        test_recurrent_hidden_states = torch.zeros(
            (
                self.config.habitat_baselines.num_environments,
                *self._agent.hidden_state_shape,
            ),
            device=self.device,
        )
        # Previous actions buffer (dtype depends on discrete vs continuous).
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        # not_done_masks marks envs that are still running (True) vs just finished (False).
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        # Track per-episode stats using (scene_id, episode_id) as a unique key.
        stats_episodes: Dict[Any, Any] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        # Optional per-env RGB frame buffers used for video generation.
        rgb_frames: List[List[np.ndarray]] = [[] for _ in range(self.config.habitat_baselines.num_environments)]
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        # How many episodes to evaluate; -1 means "use all episodes in the dataset".
        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count
        # How many times to eval the same episode (usually 1).
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes, dataset only has {{total_num_eps}}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert number_of_eval_episodes > 0, "You must specify a number of evaluation episodes with test_episode_count"

        # User-facing progress bar across episode evaluations.
        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        # Put the agent into eval mode (no dropout, BatchNorm eval, etc.).
        self._agent.eval()

        # Lazy import: visualizer used only if video collection is enabled.
        from vlfm.utils.habitat_visualizer import HabitatVis

        # Live success tracking for quick terminal feedback.
        num_successes = 0
        num_total = 0
        hab_vis = HabitatVis()
        # Main evaluation loop: continue until we’ve collected all requested
        # episode evaluations, and while at least one env is still active.
        while len(stats_episodes) < (number_of_eval_episodes * evals_per_ep) and self.envs.num_envs > 0:
            # Snapshot of which episode each env slot is currently running.
            current_episodes_info = self.envs.current_episodes()

            # 1) Policy forward pass to get actions (+ optional RNN updates).
            with inference_mode():
                action_data = self._agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                # Optional: record raw action IDs to a file (for debugging/analysis)
                # if the environment variable is provided.
                if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
                    action_id = action_data.actions.cpu()[0].item()
                    filepath = os.path.join(
                        os.environ["VLFM_RECORD_ACTIONS_DIR"],
                        "actions.txt",
                    )
                    # If the file doesn't exist, create it
                    if not os.path.exists(filepath):
                        open(filepath, "w").close()
                    with open(filepath, "a") as f:
                        f.write(f"{action_id}\n")

                # Either update all env slots’ hidden states/prev_actions,
                # or selectively update only those flagged by should_inserts.
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = action_data.rnn_hidden_states
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    for i, should_insert in enumerate(action_data.should_inserts):
                        if should_insert.item():
                            test_recurrent_hidden_states[i] = action_data.rnn_hidden_states[i]
                            prev_actions[i].copy_(action_data.actions[i])  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(self._env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            # 2) Step all environments with their actions, then unpack the batched results.
            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            # Attach policy extras (e.g., option IDs) into infos so visualizers/loggers
            # can access them downstream.
            policy_infos = self._agent.actor_critic.get_extra(action_data, infos, dones)
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])
            # Convert observations back into a device tensor batch and re-apply
            # any transforms (keeps preprocessing consistent).
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            # Build boolean masks indicating which envs are still running.
            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            # Accumulate rewards for currently running episodes.
            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            # Prepare to pause any envs that have finished or should be skipped.
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                # If we’ve already evaluated this (scene, episode) the required number
                # of times, mark this env to be paused.
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                # Special sentinel episode id: forcibly pause this env.
                elif int(next_episodes_info[i].episode_id) == 123123123:
                    envs_to_pause.append(i)

                # If videos are requested, collect per-step visualizer overlays/frames.
                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    hab_vis.collect_data(batch, infos, action_data.policy_info)

                # episode ended
                if not not_done_masks[i].item():
                    # Update progress and build the per-episode metrics dict.
                    pbar.update()
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    # Reset reward accumulator for this env slot.
                    current_episode_reward[i] = 0
                    # Unique key for this episode instance.
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    # Maintain a live success rate printout for sanity checking.
                    if episode_stats["success"] == 1:
                        num_successes += 1
                    num_total += 1
                    print(f"Success rate: {num_successes / num_total * 100:.2f}% ({num_successes} out of {num_total})")

                    # Optionally log richer per-episode diagnostics (e.g., failure causes).
                    from vlfm.utils.episode_stats_logger import (
                        log_episode_stats,
                    )

                    try:
                        failure_cause = log_episode_stats(
                            current_episodes_info[i].episode_id,
                            current_episodes_info[i].scene_id,
                            infos[i],
                        )
                    except Exception:
                        failure_cause = "Unknown"

                    # If video export is on, flush frames for this episode and write
                    # a video/GIF with per-episode metrics encoded in filename/TB.
                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        rgb_frames[i] = hab_vis.flush_frames(failure_cause)
                        generate_video(
                            video_option=self.config.habitat_baselines.eval.video_option,
                            video_dir=self.config.habitat_baselines.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes_info[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(infos[i]),
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        rgb_frames[i] = []

                    # Optionally export a graphics replay string for offline visualization.
                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            # Move masks to model device and compact all per-env buffers by pausing
            # envs that are finished/flagged. This keeps the vectorized rollout tight.
            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        # End of main evaluation loop.
        pbar.close()

        # Optional sentinel file: some pipelines watch for this to detect completion.
        if "ZSOS_DONE_PATH" in os.environ:
            # Create an empty file at ZSOS_DONE_PATH to signal that the
            # evaluation is done
            done_path = os.environ["ZSOS_DONE_PATH"]
            with open(done_path, "w") as f:
                f.write("")

        # Sanity check: ensure we evaluated the expected number of unique episodes.
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        # Aggregate metrics across all finished episode runs by taking the mean.
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean([v[stat_key] for v in stats_episodes.values()])

        # Log per-metric averages to the Habitat logger for easy reading.
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        # Prefer the true training step stored in the checkpoint for TB x-axis.
        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        # Write reward and other metrics to TensorBoard.
        writer.add_scalar("eval_reward/average_reward", aggregated_stats["reward"], step_id)

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        # Cleanly stop all env workers.
        self.envs.close()
