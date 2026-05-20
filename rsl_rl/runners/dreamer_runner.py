# Copyright (c) Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""
DreamerRunner: Training loop for Dreamer Branch.

Handles the full DreamerV3-style training cycle:
1. Real environment rollout (collect data for WM).
2. World model training (on real data).
3. Behavior learning via latent imagination (DreamerBehavior).

Keeps AMP discriminator and privileged critic on the real-rollout side only.
"""

import time
import os
from collections import deque
import statistics

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer
from rsl_rl.modules import DepthPredictor, DreamerActorCritic
from rsl_rl.algorithms.dreamer_behavior import DreamerBehavior
from rsl_rl.storage.dreamer_replay import DreamerReplay

from dreamer.models import WorldModel
from dreamer import tools


class DreamerRunner:
    """Dreamer Branch training runner.

    Shares world model and data collection infrastructure with WMPRunner,
    but replaces PPO-based policy learning with latent imagination.
    """

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 history_length=5,
                 ):
        self.train_cfg = train_cfg  # keep full dict for wm_config access
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg.get("algorithm", {})
        self.policy_cfg = train_cfg.get("policy", {})
        self.depth_predictor_cfg = train_cfg.get("depth_predictor", {})
        self.device = device
        self.env = env
        self.history_length = history_length

        # Build world model (shared with WMP path)
        self._build_world_model()

        # Build depth predictor
        self.depth_predictor = DepthPredictor().to(self._world_model.device)
        self.depth_predictor_opt = optim.Adam(
            self.depth_predictor.parameters(),
            lr=self.depth_predictor_cfg.get("lr", 3e-4),
            weight_decay=self.depth_predictor_cfg.get("weight_decay", 1e-4),
        )

        # --- Dreamer-specific modules ---
        # Feat size for actor/critic: stoch*disc + deter (or stoch + deter if continuous)
        if self.wm_config.dyn_discrete:
            feat_size = self.wm_config.dyn_stoch * self.wm_config.dyn_discrete + self.wm_config.dyn_deter
        else:
            feat_size = self.wm_config.dyn_stoch + self.wm_config.dyn_deter

        self._dreamer_ac = DreamerActorCritic(
            self.wm_config, feat_size, self.wm_config.num_actions,
            use_amp=(self.wm_config.precision == 16),
        )
        self._dreamer_ac.to(self._world_model.device)

        self._dreamer_behavior = DreamerBehavior(
            self._world_model, self._dreamer_ac, self.wm_config,
        )
        # Override imagined horizon from runner config
        imagined_horizon = self.cfg.get("dreamer_imagined_horizon", 16)
        self._dreamer_behavior._imagined_horizon = imagined_horizon

        # --- AMP (real-side only) ---
        self._use_amp_aux = self.cfg.get("dreamer_use_amp_aux", True)
        if self._use_amp_aux:
            amp_data = AMPLoader(
                device, time_between_frames=self.env.dt, preload_transitions=True,
                num_preload_transitions=self.cfg.get("amp_num_preload_transitions", 2000000),
                motion_files=self.cfg.get("amp_motion_files", []),
            )
            amp_normalizer = Normalizer(amp_data.observation_dim)
            self._discriminator = AMPDiscriminator(
                amp_data.observation_dim * 2,
                self.cfg.get("amp_reward_coef", 0.01),
                self.cfg.get("amp_discr_hidden_dims", [1024, 512]),
                device,
                self.cfg.get("amp_task_reward_lerp", 0.3),
            ).to(self.device)
            self._amp_data = amp_data
            self._amp_normalizer = amp_normalizer
            # Separate optimizer for AMP discriminator
            self._amp_opt = optim.Adam(
                self._discriminator.parameters(), lr=self.alg_cfg.get("learning_rate", 1e-3),
            )
        else:
            self._discriminator = None
            self._amp_data = None
            self._amp_normalizer = None
            self._amp_opt = None

        # --- Training state ---
        self.num_steps_per_env = self.cfg.get("num_steps_per_env", 24)
        self.save_interval = self.cfg.get("save_interval", 1000)
        self.wm_update_interval = self.env.cfg.depth.update_interval

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def _build_world_model(self):
        """Build world model from config (same as WMPRunner)."""
        print('Begin construct world model (Dreamer Branch)')
        self.wm_config = self.train_cfg.get("wm_config")
        if self.wm_config is None:
            raise ValueError("wm_config must be provided in train_cfg")

        prop_dim = self.env.num_obs - self.env.privileged_dim - self.env.height_dim - self.env.num_actions
        image_shape = self.env.cfg.depth.resized + (1,)
        obs_shape = {'prop': (prop_dim,), 'image': image_shape}

        self._world_model = WorldModel(self.wm_config, obs_shape, use_camera=self.env.cfg.depth.use_camera)
        self._world_model = self._world_model.to(self._world_model.device)
        print('Finish construct world model')
        if self.wm_config.dyn_discrete:
            self.wm_feature_dim = self.wm_config.dyn_stoch * self.wm_config.dyn_discrete + self.wm_config.dyn_deter
        else:
            self.wm_feature_dim = self.wm_config.dyn_stoch + self.wm_config.dyn_deter

    # ------------------------------------------------------------------
    # Data collection (reuses WMPRunner's wm_dataset infrastructure)
    # ------------------------------------------------------------------

    def init_wm_dataset(self):
        """Initialize world model dataset buffers (same as WMPRunner)."""
        max_len = int(self.env.max_episode_length / self.wm_update_interval) + 3
        self.wm_dataset = {
            "prop": torch.zeros((self.env.num_envs, max_len, self.env.cfg.env.prop_dim),
                                device=self._world_model.device),
            "action": torch.zeros((self.env.num_envs, max_len,
                                   self.env.num_actions * self.wm_update_interval),
                                  device=self._world_model.device),
            "reward": torch.zeros((self.env.num_envs, max_len),
                                  device=self._world_model.device),
        }
        if self.env.cfg.depth.use_camera:
            self.wm_dataset["image"] = torch.zeros(
                (self.env.cfg.depth.camera_num_envs, max_len,) + self.env.cfg.depth.resized + (1,),
                device=self._world_model.device,
            )
            self.wm_dataset["forward_height_map"] = torch.zeros(
                (self.env.num_envs, max_len, self.env.cfg.env.forward_height_dim),
                device=self._world_model.device,
            )

        self.wm_dataset_size = np.zeros(self.env.num_envs)

        # CPU-side buffers for async collection
        self.wm_buffer = {
            "prop": torch.zeros((self.env.num_envs, max_len, self.env.cfg.env.prop_dim), device='cpu'),
            "action": torch.zeros((self.env.num_envs, max_len,
                                   self.env.num_actions * self.wm_update_interval), device='cpu'),
            "reward": torch.zeros((self.env.num_envs, max_len), device='cpu'),
        }
        if self.env.cfg.depth.use_camera:
            self.wm_buffer["image"] = torch.zeros(
                (self.env.cfg.depth.camera_num_envs, max_len,) + self.env.cfg.depth.resized + (1,), device='cpu',
            )
            self.wm_buffer["forward_height_map"] = torch.zeros(
                (self.env.num_envs, max_len, self.env.cfg.env.forward_height_dim), device='cpu',
            )

        self.wm_buffer_index = np.zeros(self.env.num_envs)

        # Create DreamerReplay wrapper
        self._replay = DreamerReplay(
            self.wm_dataset, self.wm_dataset_size, self.wm_config,
            depth_predictor=self.depth_predictor,
            depth_index=self.env.depth_index if hasattr(self.env, 'depth_index') else None,
            depth_index_inverse=self.env.depth_index_inverse if hasattr(self.env, 'depth_index_inverse') else None,
            env=self.env,
        )

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def train_depth_predictor(self):
        """Train depth predictor (same as WMPRunner)."""
        if not self.env.cfg.depth.use_camera:
            return 0.0
        total_mse_loss = 0
        dp_cfg = self.depth_predictor_cfg
        for _ in range(dp_cfg.get("training_iters", 1000)):
            batch_idx = np.random.choice(
                self.env.depth_index_without_crawl_tilt,
                dp_cfg.get("batch_size", 1024), replace=True,
            )
            time_index = [np.random.randint(0, self.wm_dataset_size[idx] + 1) for idx in batch_idx]
            forward_heightmap = self.wm_dataset["forward_height_map"][batch_idx, time_index]
            prop = self.wm_dataset["prop"][batch_idx, time_index]
            depth_image = self.wm_dataset["image"][self.env.depth_index_inverse[batch_idx], time_index]

            predict_depth_image = self.depth_predictor(forward_heightmap, prop)
            depth_predict_loss = (depth_image - predict_depth_image).pow(2).mean() * dp_cfg.get("loss_scale", 100)

            self.depth_predictor_opt.zero_grad()
            depth_predict_loss.backward()
            nn.utils.clip_grad_norm_(self.depth_predictor.parameters(), 1)
            self.depth_predictor_opt.step()
            total_mse_loss += depth_predict_loss.detach() / dp_cfg.get("loss_scale", 100)
        return float(total_mse_loss / dp_cfg.get("training_iters", 1000))

    def train_world_model(self):
        """Train world model on real data."""
        wm_metrics = {}
        for _ in range(self.wm_config.train_steps_per_iter):
            batch_data = self._replay.sample_batch()
            if batch_data is None:
                continue
            post, context, mets = self._world_model._train(batch_data)
            wm_metrics.update(mets)
        return wm_metrics

    def train_amp(self, amp_obs_batch, amp_next_obs_batch):
        """Train AMP discriminator on real transitions (real-side only)."""
        if self._discriminator is None:
            return {}

        # Expert batch
        expert_batch = next(self._amp_data.feed_forward_generator(
            1, amp_obs_batch.shape[0],
        ))
        expert_state, expert_next_state = expert_batch

        if self._amp_normalizer is not None:
            with torch.no_grad():
                amp_obs_batch = self._amp_normalizer.normalize_torch(amp_obs_batch, self.device)
                amp_next_obs_batch = self._amp_normalizer.normalize_torch(amp_next_obs_batch, self.device)
                expert_state = self._amp_normalizer.normalize_torch(expert_state, self.device)
                expert_next_state = self._amp_normalizer.normalize_torch(expert_next_state, self.device)

        policy_d = self._discriminator(torch.cat([amp_obs_batch, amp_next_obs_batch], dim=-1))
        expert_d = self._discriminator(torch.cat([expert_state, expert_next_state], dim=-1))

        expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
        policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
        amp_loss = 0.5 * (expert_loss + policy_loss)
        grad_pen_loss = self._discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=10)

        total_loss = amp_loss + grad_pen_loss
        self._amp_opt.zero_grad()
        total_loss.backward()
        self._amp_opt.step()

        if self._amp_normalizer is not None:
            self._amp_normalizer.update(amp_obs_batch.cpu().numpy())
            self._amp_normalizer.update(expert_state.cpu().numpy())

        return {
            "amp_loss": amp_loss.item(),
            "amp_grad_pen": grad_pen_loss.item(),
            "amp_policy_pred": policy_d.mean().item(),
            "amp_expert_pred": expert_d.mean().item(),
        }

    def train_behavior(self):
        """Train actor and critic via latent imagination."""
        # Sample a batch and get posterior state for imagination seed
        batch_data = self._replay.sample_batch()
        if batch_data is None:
            return {}

        # Get posterior state from WM
        with torch.no_grad():
            embed = self._world_model.encoder(batch_data)
            post, _ = self._world_model.dynamics.observe(
                embed, batch_data["action"], batch_data["is_first"],
            )
            # Use the last step's posterior as imagination seed
            init_state = {k: v[:, -1] for k, v in post.items()}

        # Run behavior update
        metrics = self._dreamer_behavior.update(init_state)
        return metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """Main Dreamer Branch training loop."""
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length),
            )

        obs = self.env.get_observations()
        amp_obs = self.env.get_amp_observations()
        obs = obs.to(self.device)
        amp_obs = amp_obs.to(self.device)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        # Init WM input
        sum_wm_dataset_size = 0
        wm_latent = wm_action = None
        wm_is_first = torch.ones(self.env.num_envs, device=self._world_model.device)
        wm_obs = {
            "prop": obs[:, self.env.privileged_dim:self.env.privileged_dim + self.env.cfg.env.prop_dim].to(
                self._world_model.device),
            "is_first": wm_is_first,
        }
        if self.env.cfg.depth.use_camera:
            wm_obs["image"] = torch.zeros(
                (self.env.num_envs,) + self.env.cfg.depth.resized + (1,),
                device=self._world_model.device,
            )

        wm_action_history = torch.zeros(
            (self.env.num_envs, self.wm_update_interval, self.env.num_actions),
            device=self._world_model.device,
        )
        wm_reward = torch.zeros(self.env.num_envs, device=self._world_model.device)
        wm_feature = torch.zeros((self.env.num_envs, self.wm_feature_dim))

        self.init_wm_dataset()

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # --- Real Rollout ---
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # WM obs step (every wm_update_interval env steps)
                    if self.env.global_counter % self.wm_update_interval == 0:
                        wm_embed = self._world_model.encoder(wm_obs)
                        wm_latent, _ = self._world_model.dynamics.obs_step(
                            wm_latent, wm_action, wm_embed, wm_obs["is_first"],
                        )
                        wm_feature = self._world_model.dynamics.get_feat(wm_latent)
                        wm_is_first[:] = 0

                    # Use Dreamer actor for action selection (from WM feature)
                    actions = self._dreamer_ac.act(
                        wm_feature.to(self._world_model.device), eval_mode=False,
                    )
                    # Map to env device
                    actions_env = actions.to(self.device)

                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = \
                        self.env.step(actions_env)
                    next_amp_obs = self.env.get_amp_observations()

                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    next_amp_obs = next_amp_obs.to(self.device)

                    # Update WM input
                    wm_action_history = torch.concat(
                        (wm_action_history[:, 1:], actions.unsqueeze(1).to(self._world_model.device)), dim=1,
                    )
                    wm_obs = {
                        "prop": obs[:, self.env.privileged_dim:self.env.privileged_dim + self.env.cfg.env.prop_dim].to(
                            self._world_model.device),
                        "is_first": wm_is_first,
                    }

                    # Handle resets
                    reset_env_ids_np = reset_env_ids.cpu().numpy()
                    if len(reset_env_ids_np) > 0:
                        for k, v in self.wm_dataset.items():
                            if k == "image":
                                for rid in reset_env_ids_np:
                                    idx_in_buffer = np.where(self.env.depth_index == rid)[0]
                                    if len(idx_in_buffer) > 0:
                                        v[idx_in_buffer, :] = self.wm_buffer[k][idx_in_buffer].to(
                                            self._world_model.device)
                            else:
                                v[reset_env_ids_np, :] = self.wm_buffer[k][reset_env_ids_np].to(
                                    self._world_model.device)

                        self.wm_dataset_size[reset_env_ids_np] = self.wm_buffer_index[reset_env_ids_np]
                        self.wm_buffer_index[reset_env_ids_np] = 0
                        sum_wm_dataset_size = np.sum(self.wm_dataset_size)

                        wm_action_history[reset_env_ids_np, :] = 0
                        wm_is_first[reset_env_ids_np] = 1

                    wm_action = wm_action_history.flatten(1)
                    wm_reward += rewards.to(self._world_model.device)

                    # Store into buffer
                    if self.env.global_counter % self.wm_update_interval == 0:
                        if self.env.cfg.depth.use_camera:
                            forward_heightmap = self.env.get_forward_map().to(self._world_model.device)
                            pred_depth_image = self.depth_predictor(forward_heightmap, wm_obs["prop"])
                            wm_obs["image"] = pred_depth_image
                            self.wm_buffer["forward_height_map"][
                                range(self.env.num_envs), self.wm_buffer_index, :
                            ] = forward_heightmap[:].to('cpu')
                            wm_obs["image"][self.env.depth_index] = infos["depth"].unsqueeze(-1).to(
                                self._world_model.device)
                            self.wm_buffer["image"][
                                range(self.env.cfg.depth.camera_num_envs),
                                self.wm_buffer_index[self.env.depth_index], :
                            ] = wm_obs["image"][self.env.depth_index].to('cpu')

                        not_reset_env_ids = (1 - wm_is_first).nonzero(as_tuple=False).flatten().cpu().numpy()
                        if len(not_reset_env_ids) > 0:
                            for k, v in wm_obs.items():
                                if k != "is_first" and k != "image":
                                    self.wm_buffer[k][
                                        not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :
                                    ] = v[not_reset_env_ids].to('cpu')
                            self.wm_buffer["action"][
                                not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :
                            ] = wm_action[not_reset_env_ids, :].to('cpu')
                            self.wm_buffer["reward"][
                                not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids]
                            ] = wm_reward[not_reset_env_ids].to('cpu')
                            self.wm_buffer_index[not_reset_env_ids] += 1

                        wm_reward[:] = 0

                    # AMP reward (real-side only)
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states
                    if self._discriminator is not None:
                        rewards = self._discriminator.predict_amp_reward(
                            amp_obs, next_amp_obs_with_term, rewards,
                            normalizer=self._amp_normalizer,
                        )[0]
                    amp_obs = torch.clone(next_amp_obs)

                    # Bookkeeping
                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            collection_time = time.time() - start

            # --- Training ---
            train_start = time.time()

            # Train world model
            if sum_wm_dataset_size > self.wm_config.train_start_steps:
                # Depth predictor
                if it % self.depth_predictor_cfg.get("training_interval", 10) == 0:
                    depth_mse_loss = self.train_depth_predictor()
                    if self.writer is not None:
                        self.writer.add_scalar('DepthPredictor/loss', depth_mse_loss, it)

                # World model
                wm_metrics = self.train_world_model()
                if self.writer is not None:
                    for name, values in wm_metrics.items():
                        self.writer.add_scalar('World_model/' + name, float(np.mean(values)), it)

                # Behavior learning (latent imagination)
                behavior_metrics = self.train_behavior()
                if self.writer is not None:
                    for name, value in behavior_metrics.items():
                        if isinstance(value, (int, float, np.floating)):
                            self.writer.add_scalar('Behavior/' + name, float(value), it)

            train_time = time.time() - train_start

            # Log
            if self.log_dir is not None:
                self.log(locals())

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))

            ep_infos.clear()

            if it == 0:
                os.system("cp ./legged_gym/envs/a1/a1_amp_config.py " + self.log_dir + "/")

            print(f"Iter {it}: collect={collection_time:.1f}s, train={train_time:.1f}s, "
                  f"wm_data={sum_wm_dataset_size}")

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    # ------------------------------------------------------------------
    # Logging and checkpoint
    # ------------------------------------------------------------------

    def log(self, locs, width=200):
        """Log training metrics to tensorboard."""
        it = locs['it']
        if it % 10 != 0:
            return

        # Episode stats
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Episode/mean_reward', statistics.mean(locs['rewbuffer']), it)
            self.writer.add_scalar('Episode/mean_length', statistics.mean(locs['lenbuffer']), it)

        self.writer.add_scalar('Time/collection', locs['collection_time'], it)
        self.writer.add_scalar('Time/train', locs['train_time'], it)

    def save(self, path):
        """Save checkpoint."""
        torch.save({
            'world_model': self._world_model.state_dict(),
            'dreamer_ac': self._dreamer_ac.state_dict(),
            'depth_predictor': self.depth_predictor.state_dict(),
            'discriminator': self._discriminator.state_dict() if self._discriminator is not None else None,
            'current_learning_iteration': self.current_learning_iteration,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self._world_model.load_state_dict(checkpoint['world_model'])
        self._dreamer_ac.load_state_dict(checkpoint['dreamer_ac'])
        self.depth_predictor.load_state_dict(checkpoint['depth_predictor'])
        if self._discriminator is not None and checkpoint.get('discriminator') is not None:
            self._discriminator.load_state_dict(checkpoint['discriminator'])
        self.current_learning_iteration = checkpoint.get('current_learning_iteration', 0)
        print(f"Loaded checkpoint from {path}")
