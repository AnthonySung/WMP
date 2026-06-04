# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import time
import os
from collections import deque
import statistics

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import AMPPPO, PPO
from rsl_rl.modules import ActorCritic, ActorCriticWMP, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer
from rsl_rl.modules import DepthPredictor
import torch.optim as optim

from dreamer.models import *
from dreamer.behavior import ImagBehavior
import ruamel.yaml as yaml
import argparse
import pathlib
import sys
import collections
from dreamer import tools
import datetime
import uuid
class WMPRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 history_length=5,
                 ):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.depth_predictor_cfg = train_cfg["depth_predictor"]
        self.device = device
        self.env = env
        self.history_length = history_length
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        if self.env.include_history_steps is not None:
            num_actor_obs = self.env.num_obs * self.env.include_history_steps
        else:
            num_actor_obs = self.env.num_obs

        self.training_mode = self.cfg.get("wmp_training_mode", "wmp")
        if self.training_mode not in ("wmp", "align", "takeover"):
            raise ValueError(f"Unsupported wmp_training_mode: {self.training_mode}")
        self.is_dreamer_mode = self.training_mode in ("align", "takeover")
        self.dreamer_use_image = self.cfg.get("dreamer_use_image", False)
        self.dreamer_control_start_after = self.cfg.get("dreamer_control_start_after", 10000)
        self.dreamer_takeover_iters = self.cfg.get("dreamer_takeover_iters", 5000)
        self.dreamer_reward_mode = self.cfg.get("dreamer_reward_mode", "env")
        self.dreamer_distill_coef = self.cfg.get("dreamer_distill_coef", 0.0)
        self._takeover_start_it = None

        # build world model
        self._build_world_model()
        self.use_imagination_learning = self.cfg.get("use_imagination_learning", False) or self.is_dreamer_mode
        self.imagination_replace_ppo = self.cfg.get("imagination_replace_ppo", False)
        self.imagination_start_after = self.cfg.get("imagination_start_after", self.wm_config.train_start_steps)
        self.imagination_updates_per_iter = self.cfg.get("imagination_updates_per_iter", 1)
        self._imag_behavior = None
        if self.use_imagination_learning:
            self._build_imag_behavior()

        # build depth predictor
        self.depth_predictor = None
        self.depth_predictor_opt = None
        if not self.is_dreamer_mode or self.dreamer_use_image:
            self.depth_predictor = DepthPredictor().to(self._world_model.device)
            self.depth_predictor_opt = optim.Adam(self.depth_predictor.parameters(), lr=self.depth_predictor_cfg["lr"],
                                                  weight_decay=self.depth_predictor_cfg["weight_decay"])

        self.history_dim = history_length * (self.env.num_obs - self.env.privileged_dim - self.env.height_dim-3) #exclude command
        actor_critic = ActorCriticWMP(num_actor_obs=num_actor_obs,
                                          num_critic_obs=num_critic_obs,
                                          num_actions=self.env.num_actions,
                                          height_dim=self.env.height_dim,
                                          privileged_dim=self.env.privileged_dim,
                                          history_dim=self.history_dim,
                                          wm_feature_dim=self.wm_feature_dim,
                                          **self.policy_cfg).to(self.device)

        amp_data = AMPLoader(
            device, time_between_frames=self.env.dt, preload_transitions=True,
            num_preload_transitions=train_cfg['runner']['amp_num_preload_transitions'],
            motion_files=self.cfg["amp_motion_files"])
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            train_cfg['runner']['amp_reward_coef'],
            train_cfg['runner']['amp_discr_hidden_dims'], device,
            train_cfg['runner']['amp_task_reward_lerp']).to(self.device)

        # self.discr: AMPDiscriminator = AMPDiscriminator()
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        min_std = (
                torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
                (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])))
        self.alg: PPO = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device,
                                  min_std=min_std, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs],
                              [self.env.num_privileged_obs], [self.env.num_actions], self.history_dim, self.wm_feature_dim)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()


    def _build_world_model(self):
        # world model
        print('Begin construct world model')
        configs = yaml.safe_load(
            (pathlib.Path(sys.argv[0]).parent.parent.parent / "dreamer/configs.yaml").read_text()
        )

        def recursive_update(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    recursive_update(base[key], value)
                else:
                    base[key] = value

        name_list = ["defaults"]
        defaults = {}
        for name in name_list:
            recursive_update(defaults, configs[name])
        parser = argparse.ArgumentParser()
        parser.add_argument("--headless", action="store_true", default=False)
        parser.add_argument("--sim_device", default='cuda:0')
        parser.add_argument("--wm_device", default='None')
        parser.add_argument("--terrain", default='climb')
        for key, value in sorted(defaults.items(), key=lambda x: x[0]):
            arg_type = tools.args_type(value)
            parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
        self.wm_config, _ = parser.parse_known_args()
        # allow world model and rl env on different device
        if (self.wm_config.wm_device != 'None'):
            self.wm_config.device = self.wm_config.wm_device
        if self.is_dreamer_mode:
            self.wm_config.num_actions = self.env.num_actions
            self.wm_config.use_cont_head = True
        else:
            self.wm_config.num_actions = self.wm_config.num_actions * self.env.cfg.depth.update_interval
        prop_dim = self.env.num_obs - self.env.privileged_dim - self.env.height_dim - self.env.num_actions
        image_shape = self.env.cfg.depth.resized + (1,)
        obs_shape = {'prop': (prop_dim,)}
        if (not self.is_dreamer_mode and self.env.cfg.depth.use_camera) or (self.is_dreamer_mode and self.dreamer_use_image):
            obs_shape["image"] = image_shape

        use_camera = self.env.cfg.depth.use_camera if not self.is_dreamer_mode else self.dreamer_use_image
        self._world_model = WorldModel(self.wm_config, obs_shape, use_camera=use_camera)
        self._world_model = self._world_model.to(self._world_model.device)
        print('Finish construct world model')
        self.wm_feature_dim = self.wm_config.dyn_deter #+ self.wm_config.dyn_stoch * self.wm_config.dyn_discrete

    def _build_imag_behavior(self):
        print('Begin construct Dreamer Branch imagination behavior')
        self._imag_behavior = ImagBehavior(self.wm_config, self._world_model)
        self._imag_behavior = self._imag_behavior.to(self._world_model.device)
        print('Finish construct Dreamer Branch imagination behavior')

    def _get_wm_prop(self, obs):
        return obs[:, self.env.privileged_dim: self.env.privileged_dim + self.env.cfg.env.prop_dim].to(self._world_model.device)

    def _get_history_obs(self, obs):
        return torch.concat((obs[:, self.env.privileged_dim:self.env.privileged_dim + 6],
                             obs[:, self.env.privileged_dim + 9:-self.env.height_dim]), dim=1)

    def _log_scalar(self, name, value, step):
        if self.writer is not None:
            self.writer.add_scalar(name, float(np.mean(value)), step)

    def _takeover_probability(self, it, dataset_size):
        if self.training_mode == "align":
            return 1.0
        if dataset_size < self.dreamer_control_start_after:
            return 0.0
        if self._takeover_start_it is None:
            self._takeover_start_it = it
        if self.dreamer_takeover_iters <= 0:
            return 1.0
        progress = (it - self._takeover_start_it) / float(self.dreamer_takeover_iters)
        return float(np.clip(progress, 0.0, 1.0))

    def init_dreamer_dataset(self):
        horizon = int(self.env.max_episode_length) + 3
        self.dreamer_dataset = {
            "prop": torch.zeros((self.env.num_envs, horizon, self.env.cfg.env.prop_dim), device=self._world_model.device),
            "action": torch.zeros((self.env.num_envs, horizon, self.env.num_actions), device=self._world_model.device),
            "reward": torch.zeros((self.env.num_envs, horizon), device=self._world_model.device),
            "is_terminal": torch.zeros((self.env.num_envs, horizon), device=self._world_model.device),
            "is_first": torch.zeros((self.env.num_envs, horizon), device=self._world_model.device),
        }
        self.dreamer_buffer = {
            "prop": torch.zeros((self.env.num_envs, horizon, self.env.cfg.env.prop_dim), device='cpu'),
            "action": torch.zeros((self.env.num_envs, horizon, self.env.num_actions), device='cpu'),
            "reward": torch.zeros((self.env.num_envs, horizon), device='cpu'),
            "is_terminal": torch.zeros((self.env.num_envs, horizon), device='cpu'),
            "is_first": torch.zeros((self.env.num_envs, horizon), device='cpu'),
        }
        if self.dreamer_use_image:
            image_shape = self.env.cfg.depth.resized + (1,)
            self.dreamer_dataset["image"] = torch.zeros(
                (self.env.num_envs, horizon) + image_shape, device=self._world_model.device)
            self.dreamer_buffer["image"] = torch.zeros(
                (self.env.num_envs, horizon) + image_shape, device='cpu')
        self.dreamer_dataset_size = np.zeros(self.env.num_envs)
        self.dreamer_buffer_index = np.zeros(self.env.num_envs, dtype=np.int64)

    def store_dreamer_step(self, prop, action, reward, done, is_first, image=None):
        indices = np.arange(self.env.num_envs)
        step = self.dreamer_buffer_index
        capacity = self.dreamer_buffer["reward"].shape[1]
        valid = step < capacity
        if not np.any(valid):
            return
        env_ids = indices[valid]
        step_ids = step[valid]
        self.dreamer_buffer["prop"][env_ids, step_ids, :] = prop[env_ids].detach().to('cpu')
        self.dreamer_buffer["action"][env_ids, step_ids, :] = action[env_ids].detach().to('cpu')
        self.dreamer_buffer["reward"][env_ids, step_ids] = reward[env_ids].detach().to('cpu')
        self.dreamer_buffer["is_terminal"][env_ids, step_ids] = done[env_ids].float().detach().to('cpu')
        self.dreamer_buffer["is_first"][env_ids, step_ids] = is_first[env_ids].float().detach().to('cpu')
        if self.dreamer_use_image:
            if image is None:
                raise ValueError("dreamer_use_image=True requires image data in store_dreamer_step().")
            self.dreamer_buffer["image"][env_ids, step_ids, :] = image[env_ids].detach().to('cpu')
        self.dreamer_buffer_index[env_ids] += 1

    def _make_dreamer_image_obs(self, infos=None, fallback=None):
        if not self.dreamer_use_image:
            return None
        image = torch.zeros(
            ((self.env.num_envs,) + self.env.cfg.depth.resized + (1,)),
            device=self._world_model.device)
        if fallback is not None:
            image.copy_(fallback.to(self._world_model.device))
        depth = None if infos is None else infos.get("depth", None)
        if depth is not None:
            image[self.env.depth_index] = depth.unsqueeze(-1).to(self._world_model.device)
        return image

    def flush_dreamer_resets(self, reset_env_ids):
        if len(reset_env_ids) == 0:
            return
        for k, v in self.dreamer_dataset.items():
            v[reset_env_ids, :] = self.dreamer_buffer[k][reset_env_ids].to(self._world_model.device)
        self.dreamer_dataset_size[reset_env_ids] = self.dreamer_buffer_index[reset_env_ids]
        for k, v in self.dreamer_buffer.items():
            v[reset_env_ids].zero_()
        self.dreamer_buffer_index[reset_env_ids] = 0

    def sample_dreamer_batch(self):
        total = np.sum(self.dreamer_dataset_size)
        if total <= 0:
            return None
        p = self.dreamer_dataset_size / total
        batch_idx = np.random.choice(range(self.env.num_envs), self.wm_config.batch_size, replace=True, p=p)
        batch_length = min(int(self.dreamer_dataset_size[batch_idx].min()), self.wm_config.batch_length)
        if batch_length <= 1:
            return None
        batch_end_idx = [np.random.randint(batch_length, self.dreamer_dataset_size[idx] + 1) for idx in batch_idx]
        batch_data = {}
        for k, v in self.dreamer_dataset.items():
            batch_data[k] = torch.stack([
                v[idx, end_idx - batch_length:end_idx] for idx, end_idx in zip(batch_idx, batch_end_idx)
            ])
        return batch_data

    def _merge_metrics(self, accumulator, metrics):
        for name, value in metrics.items():
            accumulator.setdefault(name, []).append(value)

    def _mean_metrics(self, accumulator):
        return {name: np.mean(values, axis=0) for name, values in accumulator.items()}

    def train_dreamer_world_model(self):
        metrics = {}
        for _ in range(self.wm_config.train_steps_per_iter):
            batch_data = self.sample_dreamer_batch()
            if batch_data is None:
                continue
            post, context, mets = self._world_model._train(batch_data)
            self._merge_metrics(metrics, mets)
        return self._mean_metrics(metrics)

    def train_dreamer_behavior(self):
        metrics = {}
        if self._imag_behavior is None:
            return metrics
        for _ in range(self.imagination_updates_per_iter):
            batch_data = self.sample_dreamer_batch()
            if batch_data is None:
                continue
            with torch.no_grad():
                data = self._world_model.preprocess(batch_data)
                embed = self._world_model.encoder(data)
                post, _ = self._world_model.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )

            def reward_fn(feat, state, action):
                del feat, action
                state_feat = self._world_model.dynamics.get_feat(state)
                return self._world_model.heads["reward"](state_feat).mode()

            self._merge_metrics(metrics, self._imag_behavior.train_from_posterior(post, reward_fn))
        return self._mean_metrics(metrics)

    def learn_dreamer_modes(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        obs = self.env.get_observations().to(self.device)
        last_image_obs = self._make_dreamer_image_obs()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations().to(self.device)
        critic_obs = (privileged_obs if privileged_obs is not None else obs).to(self.device)
        self.alg.actor_critic.train()
        self.alg.discriminator.train()

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.trajectory_history = torch.zeros(size=(self.env.num_envs, self.history_length, self.env.num_obs -
                                                    self.env.privileged_dim - self.env.height_dim - 3),
                                              device=self.device)
        self.trajectory_history = torch.concat(
            (self.trajectory_history[:, 1:], self._get_history_obs(obs).unsqueeze(1)), dim=1)

        self.init_dreamer_dataset()
        dreamer_latent = dreamer_action = None
        dreamer_is_first = torch.ones(self.env.num_envs, device=self._world_model.device)
        dreamer_controller_mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        wm_feature = torch.zeros((self.env.num_envs, self.wm_feature_dim), device=self._world_model.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            if self.env.cfg.rewards.reward_curriculum:
                self.env.update_reward_curriculum(it)
            p_dreamer = self._takeover_probability(it, np.sum(self.dreamer_dataset_size))
            start = time.time()
            ep_infos = []
            if self.training_mode == "align":
                dreamer_controller_mask[:] = True
            elif self.training_mode == "takeover":
                first_env_ids = dreamer_is_first.bool().nonzero(as_tuple=False).flatten()
                if len(first_env_ids) > 0:
                    dreamer_controller_mask[first_env_ids] = (
                        torch.rand(len(first_env_ids), device=self.device) < p_dreamer)

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    prop = self._get_wm_prop(obs)
                    wm_obs = {"prop": prop, "is_first": dreamer_is_first}
                    if self.dreamer_use_image:
                        wm_obs["image"] = last_image_obs
                    transition_prop = prop
                    transition_image = last_image_obs
                    transition_is_first = dreamer_is_first.clone()
                    wm_embed = self._world_model.encoder(wm_obs)
                    dreamer_latent, _ = self._world_model.dynamics.obs_step(
                        dreamer_latent, dreamer_action, wm_embed, dreamer_is_first)
                    wm_feature = self._world_model.dynamics.get_deter_feat(dreamer_latent)
                    dreamer_is_first[:] = 0

                    history = self.trajectory_history.flatten(1).to(self.device)
                    collect_ppo_this_step = self.training_mode == "takeover" and bool((~dreamer_controller_mask).any().item())
                    if collect_ppo_this_step:
                        ppo_actions = self.alg.act(obs, critic_obs, amp_obs, history, wm_feature.to(self.device))
                    else:
                        ppo_actions = torch.zeros((self.env.num_envs, self.env.num_actions), device=self.device)
                    dreamer_actions = self._imag_behavior.act_from_state(
                        dreamer_latent, deterministic=False).to(self.device)
                    if self.training_mode == "align":
                        dreamer_controller_mask[:] = True
                    actions = torch.where(dreamer_controller_mask.unsqueeze(-1), dreamer_actions, ppo_actions)

                    obs, privileged_obs, env_rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actions)
                    next_amp_obs = self.env.get_amp_observations()
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
                    next_amp_obs, env_rewards, dones = next_amp_obs.to(self.device), env_rewards.to(self.device), dones.to(self.device)

                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    reset_env_ids_np = reset_env_ids.cpu().numpy()
                    next_amp_obs_with_term[reset_env_ids_np] = terminal_amp_states
                    amp_rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, env_rewards, normalizer=self.alg.amp_normalizer)[0]
                    train_rewards = env_rewards if self.dreamer_reward_mode == "env" else amp_rewards

                    last_image_obs = self._make_dreamer_image_obs(infos, fallback=last_image_obs)
                    self.store_dreamer_step(transition_prop, actions.to(self._world_model.device),
                                            train_rewards.to(self._world_model.device),
                                            dones.to(self._world_model.device),
                                            transition_is_first,
                                            image=transition_image)
                    self.flush_dreamer_resets(reset_env_ids_np)

                    ppo_valid_mask = (~dreamer_controller_mask).to(self.device)
                    if collect_ppo_this_step:
                        self.alg.process_env_step(amp_rewards, dones, infos, next_amp_obs_with_term, valid_mask=ppo_valid_mask)
                    amp_obs = torch.clone(next_amp_obs)
                    dreamer_action = actions.to(self._world_model.device)

                    env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.trajectory_history[env_ids] = 0
                    self.trajectory_history = torch.concat(
                        (self.trajectory_history[:, 1:], self._get_history_obs(obs).unsqueeze(1)), dim=1)
                    if len(reset_env_ids_np) > 0:
                        dreamer_is_first[reset_env_ids_np] = 1
                        dreamer_action[reset_env_ids_np] = 0
                        if self.training_mode == "takeover":
                            dreamer_controller_mask[reset_env_ids_np] = (
                                torch.rand(len(reset_env_ids_np), device=self.device) < p_dreamer)
                        elif self.training_mode == "align":
                            dreamer_controller_mask[reset_env_ids_np] = True

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += train_rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - start
                learn_start = time.time()
                if self.training_mode == "takeover" and self.alg.storage.step > 0 and torch.any(self.alg.storage.valid_masks):
                    self.alg.compute_returns(critic_obs, wm_feature.to(self.device))
            if self.training_mode == "takeover" and self.alg.storage.step > 0 and torch.any(self.alg.storage.valid_masks):
                ppo_metrics = self.alg.update()
            else:
                ppo_metrics = (0, 0, 0, 0, 0, 0, 0)
            learn_time = time.time() - learn_start

            wm_metrics = {}
            imag_metrics = {}
            dataset_size = np.sum(self.dreamer_dataset_size)
            if dataset_size > self.wm_config.train_start_steps:
                wm_metrics = self.train_dreamer_world_model()
                imag_metrics = self.train_dreamer_behavior()

            if self.writer is not None:
                self._log_scalar('DreamerMode/p_dreamer', p_dreamer, it)
                self._log_scalar('DreamerMode/dataset_size', dataset_size, it)
                for name, values in wm_metrics.items():
                    self._log_scalar('World_model/' + name, values, it)
                for name, values in imag_metrics.items():
                    self._log_scalar('ImagBehavior/' + name, values, it)
                if len(rewbuffer) > 0:
                    self._log_scalar('Train/mean_reward', statistics.mean(rewbuffer), it)
                    self._log_scalar('Train/mean_episode_length', statistics.mean(lenbuffer), it)

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            print(f"Dreamer mode {self.training_mode} iter {it}: p_dreamer={p_dreamer:.3f}, "
                  f"dataset={dataset_size:.0f}, collection={collection_time:.2f}s, learning={learn_time:.2f}s")

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.is_dreamer_mode:
            return self.learn_dreamer_modes(num_learning_iterations, init_at_random_ep_len)

        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        # process trajectory history
        self.trajectory_history = torch.zeros(size=(self.env.num_envs, self.history_length, self.env.num_obs -
                                                    self.env.privileged_dim - self.env.height_dim - 3),
                                              device=self.device)
        obs_without_command = torch.concat((obs[:, self.env.privileged_dim:self.env.privileged_dim + 6],
                                            obs[:, self.env.privileged_dim + 9:-self.env.height_dim]), dim=1)
        self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)),
                                               dim=1)

        # init world model input
        sum_wm_dataset_size = 0
        wm_latent = wm_action = None
        wm_is_first = torch.ones(self.env.num_envs, device=self._world_model.device)
        wm_obs = {
            "prop": obs[:, self.env.privileged_dim: self.env.privileged_dim + self.env.cfg.env.prop_dim].to(self._world_model.device),
            "is_first": wm_is_first,
        }

        if(self.env.cfg.depth.use_camera):
            wm_obs["image"] = torch.zeros(((self.env.num_envs,) + self.env.cfg.depth.resized + (1,)), device=self._world_model.device)

        wm_metrics = None
        self.wm_update_interval = self.env.cfg.depth.update_interval
        wm_action_history = torch.zeros(size=(self.env.num_envs, self.wm_update_interval, self.env.num_actions),
                                        device=self._world_model.device)
        wm_reward = torch.zeros(self.env.num_envs, device=self._world_model.device)
        wm_feature = torch.zeros((self.env.num_envs, self.wm_feature_dim))

        self.init_wm_dataset()


        for it in range(self.current_learning_iteration, tot_iter):
            if (self.env.cfg.rewards.reward_curriculum):
                self.env.update_reward_curriculum(it)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if (self.env.global_counter % self.wm_update_interval == 0):
                        # world model obs step
                        wm_embed = self._world_model.encoder(wm_obs)
                        wm_latent, _ = self._world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed,
                                                                           wm_obs["is_first"])
                        wm_feature = self._world_model.dynamics.get_deter_feat(wm_latent)
                        wm_is_first[:] = 0

                    history = self.trajectory_history.flatten(1).to(self.device)
                    actions = self.alg.act(obs, critic_obs, amp_obs, history, wm_feature.to(self.env.device))
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(
                        actions)
                    next_amp_obs = self.env.get_amp_observations()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs, rewards, dones = obs.to(self.device), critic_obs.to(
                        self.device), next_amp_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # update world model input
                    wm_action_history = torch.concat(
                        (wm_action_history[:, 1:], actions.unsqueeze(1).to(self._world_model.device)), dim=1)
                    wm_obs = {
                        "prop": obs[:, self.env.privileged_dim: self.env.privileged_dim + self.env.cfg.env.prop_dim].to(self._world_model.device),
                        "is_first": wm_is_first,
                    }

                    # store the data in buffer into the dataset before reset
                    reset_env_ids = reset_env_ids.cpu().numpy()
                    if (len(reset_env_ids) > 0):
                        for k, v in self.wm_dataset.items():
                            if(k == "image"):
                                for id in reset_env_ids:
                                    idx_in_buffer = np.where(self.env.depth_index == id)[0]
                                    if(len(idx_in_buffer) > 0):
                                        v[idx_in_buffer, :] = self.wm_buffer[k][idx_in_buffer].to(self._world_model.device)
                            else:
                                v[reset_env_ids, :] = self.wm_buffer[k][reset_env_ids].to(self._world_model.device)

                        self.wm_dataset_size[reset_env_ids] = self.wm_buffer_index[reset_env_ids]
                        self.wm_buffer_index[reset_env_ids] = 0
                        sum_wm_dataset_size = np.sum(self.wm_dataset_size)

                        wm_action_history[reset_env_ids, :] = 0
                        wm_is_first[reset_env_ids] = 1

                    wm_action = wm_action_history.flatten(1)
                    wm_reward += rewards.to(self._world_model.device)

                    # store current step into buffer
                    if (self.env.global_counter % self.wm_update_interval == 0):
                        if (self.env.cfg.depth.use_camera):
                            forward_heightmap = self.env.get_forward_map().to(self._world_model.device)
                            pred_depth_image = self.depth_predictor(forward_heightmap, wm_obs["prop"])
                            wm_obs["image"] = pred_depth_image
                            self.wm_buffer["forward_height_map"][range(self.env.num_envs), self.wm_buffer_index,:] = forward_heightmap[:].to('cpu')
                            wm_obs["image"][self.env.depth_index] = infos["depth"].unsqueeze(-1).to(self._world_model.device)
                            self.wm_buffer["image"][range(self.env.cfg.depth.camera_num_envs),
                            self.wm_buffer_index[self.env.depth_index], :] = wm_obs["image"][self.env.depth_index].to(
                                'cpu')
                        # not_reset_env_ids = (~dones).nonzero(as_tuple=False).flatten().cpu().numpy()
                        not_reset_env_ids = (1 - wm_is_first).nonzero(as_tuple=False).flatten().cpu().numpy()
                        if (len(not_reset_env_ids) > 0):
                            for k, v in wm_obs.items():
                                if(k != "is_first" and k != "image"):
                                    self.wm_buffer[k][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = v[not_reset_env_ids].to('cpu')
                            self.wm_buffer["action"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = \
                                wm_action[not_reset_env_ids, :].to('cpu')
                            self.wm_buffer["reward"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids]] = \
                                wm_reward[not_reset_env_ids].to('cpu')
                            self.wm_buffer_index[not_reset_env_ids] += 1

                        wm_reward[:] = 0

                    # Account for terminal states.
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)[0]
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)

                    # process trajectory history
                    env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.trajectory_history[env_ids] = 0
                    obs_without_command = torch.concat((obs[:, self.env.privileged_dim:self.env.privileged_dim + 6],
                                                        obs[:, self.env.privileged_dim + 9:-self.env.height_dim]),
                                                       dim=1)
                    self.trajectory_history = torch.concat(
                        (self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, wm_feature.to(self.env.device))
            mean_value_loss, mean_surrogate_loss, mean_vel_predict_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()


            start_time = time.time()
            if (sum_wm_dataset_size > self.wm_config.train_start_steps):

                if(self.depth_predictor is not None and it % self.depth_predictor_cfg["training_interval"] == 0):
                # Train Depth Predictor
                    depth_mse_loss = self.train_depth_predictor()
                    self.writer.add_scalar('DepthPredictor/loss', depth_mse_loss, it)

                # Train World Model
                wm_metrics = self.train_world_model()
                for name, values in wm_metrics.items():
                    self.writer.add_scalar('World_model/' + name, float(np.mean(values)), it)

                if self.use_imagination_learning and sum_wm_dataset_size > self.imagination_start_after:
                    imag_metrics = self.train_imagination_behavior()
                    for name, values in imag_metrics.items():
                        self.writer.add_scalar('ImagBehavior/' + name, float(np.mean(values)), it)
            print('training world model time:', time.time() - start_time)

            # copy the config file
            if(it == 0):
                os.system("cp ./legged_gym/envs/a1/a1_amp_config.py " + self.log_dir + "/")

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def init_wm_dataset(self):
        self.wm_dataset = {
            "prop": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, self.env.cfg.env.prop_dim),
                                device=self._world_model.device),
            "action": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   self.env.num_actions * self.wm_update_interval), device=self._world_model.device),
            "reward": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,),
                                  device=self._world_model.device),
        }
        if(self.env.cfg.depth.use_camera):
            self.wm_dataset["image"] = torch.zeros(((self.env.cfg.depth.camera_num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                               + self.env.cfg.depth.resized + (1,)), device=self._world_model.device)
            self.wm_dataset["forward_height_map"] = torch.zeros(
                (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                 self.env.cfg.env.forward_height_dim), device=self._world_model.device)

        self.wm_dataset_size = np.zeros(self.env.num_envs)

        self.wm_buffer = {
            "prop": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, self.env.cfg.env.prop_dim),
                                device='cpu'),
            "action": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   self.env.num_actions * self.wm_update_interval), device='cpu'),
            "reward": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,),
                                  device='cpu'),
        }
        if(self.env.cfg.depth.use_camera):
            self.wm_buffer["image"] = torch.zeros(((self.env.cfg.depth.camera_num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                               + self.env.cfg.depth.resized + (1,)), device='cpu')
            self.wm_buffer["forward_height_map"] = torch.zeros(
                (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                 self.env.cfg.env.forward_height_dim), device='cpu')

        self.wm_buffer_index = np.zeros(self.env.num_envs)

    def train_depth_predictor(self):
        total_mse_loss = 0
        for _ in range(self.depth_predictor_cfg["training_iters"]):
            batch_idx = np.random.choice(self.env.depth_index_without_crawl_tilt, self.depth_predictor_cfg["batch_size"],
                                         replace=True)
            time_index = [np.random.randint(0, self.wm_dataset_size[idx] + 1) for idx in batch_idx]
            forward_heightmap = self.wm_dataset["forward_height_map"][batch_idx, time_index]
            prop = self.wm_dataset["prop"][batch_idx, time_index]
            depth_image = self.wm_dataset["image"][self.env.depth_index_inverse[batch_idx], time_index]

            predict_depth_image = self.depth_predictor(forward_heightmap, prop)
            depth_predict_loss = (depth_image - predict_depth_image).pow(2).mean() * self.depth_predictor_cfg[
                "loss_scale"]
            # Gradient step
            self.depth_predictor_opt.zero_grad()
            depth_predict_loss.backward()
            nn.utils.clip_grad_norm_(self.depth_predictor.parameters(), 1)
            self.depth_predictor_opt.step()
            total_mse_loss += depth_predict_loss.detach() / self.depth_predictor_cfg["loss_scale"]
        return float(total_mse_loss / self.depth_predictor_cfg["training_iters"])

    def train_world_model(self):
        wm_metrics = {}
        mets = {}
        for i in range(self.wm_config.train_steps_per_iter):
            batch_data = self.sample_world_model_batch()
            if batch_data is None:
                continue
            post, context, mets = self._world_model._train(batch_data)
        wm_metrics.update(mets)
        return wm_metrics

    def sample_world_model_batch(self):
        p = self.wm_dataset_size / np.sum(self.wm_dataset_size)
        batch_idx = np.random.choice(range(self.env.num_envs), self.wm_config.batch_size, replace=True,
                                     p=p)
        batch_length = min(int(self.wm_dataset_size[batch_idx].min()), self.wm_config.batch_length)
        if (batch_length <= 1):
            return None
        batch_end_idx = [np.random.randint(batch_length, self.wm_dataset_size[idx] + 1) for idx in batch_idx]
        batch_data = {}
        for k, v in self.wm_dataset.items():
            if (k == "forward_height_map"):
                continue
            value = []
            for idx, end_idx in zip(batch_idx, batch_end_idx):
                if (k == "image"):
                    idx_in_buffer = np.where(self.env.depth_index == idx)[0]
                    if (len(idx_in_buffer) == 0):
                        tmp_forward_heightmap = self.wm_dataset["forward_height_map"][idx,
                                                end_idx - batch_length: end_idx]
                        tmp_prop = self.wm_dataset["prop"][idx, end_idx - batch_length: end_idx]
                        pred_depth_image = self.depth_predictor(tmp_forward_heightmap, tmp_prop)
                        value.append(pred_depth_image)
                    else:
                        value.append(v[idx_in_buffer[0], end_idx - batch_length: end_idx])
                else:
                    value.append(v[idx, end_idx - batch_length: end_idx])
            value = torch.stack(value)
            batch_data[k] = value
        is_first = torch.zeros((self.wm_config.batch_size, batch_length))
        is_first[:, 0] = 1
        batch_data["is_first"] = is_first
        return batch_data

    def train_imagination_behavior(self):
        imag_metrics = {}
        if self._imag_behavior is None:
            return imag_metrics
        for _ in range(self.imagination_updates_per_iter):
            batch_data = self.sample_world_model_batch()
            if batch_data is None:
                continue
            with torch.no_grad():
                data = self._world_model.preprocess(batch_data)
                embed = self._world_model.encoder(data)
                post, _ = self._world_model.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )

            def reward_fn(feat, state, action):
                del feat, action
                state_feat = self._world_model.dynamics.get_feat(state)
                return self._world_model.heads["reward"](state_feat).mode()

            mets = self._imag_behavior.train_from_posterior(post, reward_fn)
            imag_metrics.update(mets)
        return imag_metrics

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/vel_predict', locs['mean_vel_predict_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/AMP_mean_policy_pred', locs['mean_policy_pred'], locs['it'])
        self.writer.add_scalar('Loss/AMP_mean_expert_pred', locs['mean_expert_pred'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Vel predict loss:':>{pad}} {locs['mean_vel_predict_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'world_model_dict': self._world_model.state_dict(),
            'wm_optimizer_state_dict': self._world_model._model_opt._opt.state_dict(),
            'imag_behavior_dict': self._imag_behavior.state_dict() if self._imag_behavior is not None else None,
            'imag_actor_optimizer_state_dict': self._imag_behavior._actor_opt._opt.state_dict() if self._imag_behavior is not None else None,
            'imag_value_optimizer_state_dict': self._imag_behavior._value_opt._opt.state_dict() if self._imag_behavior is not None else None,
            'depth_predictor': self.depth_predictor.state_dict() if self.depth_predictor is not None else None,
            # 'discriminator_state_dict': self.alg.discriminator.state_dict(),
            # 'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True, load_wm_optimizer = False):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=False)
        self._world_model.load_state_dict(loaded_dict['world_model_dict'], strict=False)
        if self._imag_behavior is not None and loaded_dict.get('imag_behavior_dict') is not None:
            self._imag_behavior.load_state_dict(loaded_dict['imag_behavior_dict'], strict=False)
            if load_wm_optimizer:
                actor_opt_state = loaded_dict.get('imag_actor_optimizer_state_dict')
                value_opt_state = loaded_dict.get('imag_value_optimizer_state_dict')
                if actor_opt_state is not None:
                    self._imag_behavior._actor_opt._opt.load_state_dict(actor_opt_state)
                if value_opt_state is not None:
                    self._imag_behavior._value_opt._opt.load_state_dict(value_opt_state)
        if(load_wm_optimizer):
            self._world_model._model_opt._opt.load_state_dict(loaded_dict['wm_optimizer_state_dict'])
        # self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'], strict=False)
        # self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
