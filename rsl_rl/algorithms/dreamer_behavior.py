# Copyright (c) Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""
DreamerBehavior: Imagined rollout + actor/critic update for Dreamer Branch.

This module handles the core DreamerV3-style behavior learning loop:
1. Start from a posterior latent state (encoded from real data).
2. Roll out imagined trajectory using RSSM img_step + actor.
3. Compute lambda-return targets.
4. Update actor and latent critic.
"""

import torch
import torch.nn as nn

from dreamer import tools


class DreamerBehavior:
    """Dreamer-style behavior learning via latent imagination.

    Trains actor and latent critic entirely in the imagined latent space.
    Does NOT consume privileged observations or AMP rewards.
    """

    def __init__(self, world_model, actor_critic, config):
        """
        Args:
            world_model: WorldModel instance (provides RSSM dynamics and heads).
            actor_critic: DreamerActorCritic instance.
            config: WM config namespace.
        """
        self._wm = world_model
        self._ac = actor_critic
        self._config = config

        # Behavior hyperparams
        self._imagined_horizon = getattr(config, 'imagined_horizon', 16)
        self._discount = getattr(config, 'discount', 0.997)
        self._lambda = getattr(config, 'lambda_return', 0.95)
        actor_cfg = config.actor if hasattr(config, 'actor') else config['actor']
        self._entropy_scale = actor_cfg.get('entropy', 3e-4) if isinstance(actor_cfg, dict) else getattr(actor_cfg, 'entropy', 3e-4)

    def imagine_trajectory(self, post_state, horizon=None):
        """Perform imagined rollout from a posterior state.

        Args:
            post_state: Initial posterior state dict from RSSM.
            horizon: Rollout length. Defaults to self._imagined_horizon.

        Returns:
            feats: (horizon, batch, feat_size) latent features along trajectory.
            states: (horizon, batch, ...) prior states along trajectory.
            actions: (horizon, batch, num_actions) actions taken.
        """
        horizon = horizon or self._imagined_horizon
        batch_size = post_state["stoch"].shape[0]
        device = post_state["stoch"].device

        state = {k: v for k, v in post_state.items()}
        feats = []
        states = []
        actions = []

        for _ in range(horizon):
            feat = self._wm.dynamics.get_feat(state)
            action = self._ac.act(feat)
            # img_step: prior from previous state + action
            state = self._wm.dynamics.img_step(state, action, sample=True)

            feats.append(feat)
            states.append(state)
            actions.append(action)

        # Stack along time dim: (horizon, batch, ...)
        feats = torch.stack(feats, dim=0)
        actions = torch.stack(actions, dim=0)
        # states is a list of dicts; keep as list for now
        return feats, states, actions

    def compute_imagined_rewards(self, feats):
        """Predict rewards from latent features.

        Args:
            feats: (horizon, batch, feat_size) latent features.

        Returns:
            rewards: (horizon, batch) predicted rewards.
        """
        # Reshape to (horizon * batch, feat_size) for batched forward
        flat_feats = feats.reshape(-1, feats.shape[-1])
        reward_dist = self._wm.heads["reward"](flat_feats)
        rewards = reward_dist.mode()
        # Reshape back to (horizon, batch)
        rewards = rewards.reshape(feats.shape[0], feats.shape[1])
        return rewards

    def compute_imagined_continuation(self, feats):
        """Predict continuation (not-terminal) probabilities.

        Args:
            feats: (horizon, batch, feat_size) latent features.

        Returns:
            cont: (horizon, batch) continuation probabilities in [0, 1].
        """
        flat_feats = feats.reshape(-1, feats.shape[-1])
        cont_dist = self._wm.heads["cont"](flat_feats)
        cont = cont_dist.mode()
        cont = cont.reshape(feats.shape[0], feats.shape[1])
        return cont

    def compute_value(self, feats, use_slow=False):
        """Predict values from latent features.

        Args:
            feats: (horizon, batch, feat_size) latent features.
            use_slow: Use slow target critic if True.

        Returns:
            values: (horizon, batch) predicted values.
        """
        flat_feats = feats.reshape(-1, feats.shape[-1])
        values = self._ac.get_value(flat_feats, use_slow=use_slow)
        values = values.reshape(feats.shape[0], feats.shape[1])
        return values

    def compute_lambda_target(self, rewards, values, continuation):
        """Compute lambda-return targets for value learning.

        Args:
            rewards: (horizon, batch) predicted rewards.
            values: (horizon, batch) predicted values (from slow critic).
            continuation: (horizon, batch) continuation probabilities.

        Returns:
            lambda_target: (horizon, batch) lambda-return targets.
        """
        # pcont = discount * continuation
        pcont = self._discount * continuation

        # Bootstrap: value of the step after horizon (zero if terminal)
        bootstrap = torch.zeros_like(values[-1])

        lambda_target = tools.lambda_return(
            rewards, values, pcont, bootstrap, self._lambda, axis=0
        )
        return lambda_target

    def update_actor(self, feats, actions, advantages):
        """Update actor using imagined trajectory.

        Args:
            feats: (horizon, batch, feat_size) latent features.
            actions: (horizon, batch, num_actions) actions taken.
            advantages: (horizon, batch) advantage values.

        Returns:
            metrics: dict of training metrics.
        """
        flat_feats = feats.reshape(-1, feats.shape[-1])
        flat_actions = actions.reshape(-1, actions.shape[-1])
        flat_advantages = advantages.reshape(-1)

        return self._ac.actor_loss(
            flat_feats, flat_actions, flat_advantages, self._entropy_scale
        )

    def update_critic(self, feats, lambda_target):
        """Update critic using lambda-return targets.

        Args:
            feats: (horizon, batch, feat_size) latent features.
            lambda_target: (horizon, batch) lambda-return targets.

        Returns:
            metrics: dict of training metrics.
        """
        flat_feats = feats.reshape(-1, feats.shape[-1])
        flat_target = lambda_target.reshape(-1)

        return self._ac.critic_loss(flat_feats, flat_target)

    def update(self, post_state):
        """Full behavior update: imagine → compute targets → update actor & critic.

        Args:
            post_state: Initial posterior state dict from RSSM.

        Returns:
            metrics: dict of all training metrics.
        """
        # 1. Imagine trajectory
        feats, states, actions = self.imagine_trajectory(post_state)

        # 2. Predict rewards and continuation
        rewards = self.compute_imagined_rewards(feats)
        continuation = self.compute_imagined_continuation(feats)

        # 3. Compute values (slow critic for bootstrap)
        with torch.no_grad():
            slow_values = self.compute_value(feats, use_slow=True)

        # 4. Compute lambda-return targets
        lambda_target = self.compute_lambda_target(rewards, slow_values, continuation)

        # 5. Compute advantages
        with torch.no_grad():
            online_values = self.compute_value(feats, use_slow=False)
        advantages = lambda_target - online_values

        # 6. Update critic
        critic_metrics = self.update_critic(feats, lambda_target)

        # 7. Update actor
        actor_metrics = self.update_actor(feats.detach(), actions, advantages)

        # 8. Update slow critic
        self._ac.update_slow_critic()

        # Merge metrics
        metrics = {}
        metrics.update(critic_metrics)
        metrics.update(actor_metrics)
        return metrics
