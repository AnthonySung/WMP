# Copyright (c) Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""
DreamerActorCritic: Actor and latent critic for Dreamer Branch behavior learning.

Both actor and critic consume only RSSM latent features (no privileged observation).
The critic uses a symlog-discretized distribution (DreamerV3 style).
A slow-moving target critic is maintained via EMA for stable value bootstrapping.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from dreamer import tools
from dreamer import networks


class DreamerActorCritic(nn.Module):
    """Actor and latent critic for imagined rollout in Dreamer Branch.

    Actor: latent_feat -> action distribution (Normal with learned std).
    Latent Critic: latent_feat -> symlog-discretized value distribution.
    Slow Critic: EMA copy of latent critic for stable bootstrapping.
    """

    def __init__(self, config, feat_size, num_actions, use_amp=False):
        """
        Args:
            config: WM config namespace (has actor.* and critic.* sub-configs).
            feat_size: Dimension of RSSM latent feature (stoch + deter).
            num_actions: Action dimension.
            use_amp: Whether to use AMP mixed precision.
        """
        super(DreamerActorCritic, self).__init__()
        self._config = config
        self._use_amp = use_amp
        self._feat_size = feat_size
        self._num_actions = num_actions

        # --- Actor ---
        self.actor = networks.MLP(
            feat_size,
            (num_actions,),
            config.actor.layers,
            config.units,
            config.act,
            config.norm,
            dist=config.actor.dist,
            std=config.actor.std,
            min_std=config.actor.min_std,
            max_std=config.actor.max_std,
            absmax=1.0,
            temp=config.actor.temp,
            unimix_ratio=config.actor.unimix_ratio,
            outscale=config.actor.outscale,
            device=config.device,
            name="Actor",
        )

        # --- Latent Critic ---
        self.critic = networks.MLP(
            feat_size,
            (255,),  # symlog_disc uses 255 bins
            config.critic.layers,
            config.units,
            config.act,
            config.norm,
            dist=config.critic.dist,
            outscale=config.critic.outscale,
            device=config.device,
            name="Critic",
        )

        # --- Slow Target Critic ---
        if config.critic.slow_target:
            self.slow_critic = networks.MLP(
                feat_size,
                (255,),
                config.critic.layers,
                config.units,
                config.act,
                config.norm,
                dist=config.critic.dist,
                outscale=config.critic.outscale,
                device=config.device,
                name="SlowCritic",
            )
            # Initialize slow critic with same weights
            self.slow_critic.load_state_dict(self.critic.state_dict())
            self._slow_target_update = config.critic.slow_target_update
            self._slow_target_fraction = config.critic.slow_target_fraction
        else:
            self.slow_critic = None

        # --- Optimizers ---
        self.actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor.lr,
            config.actor.eps,
            config.actor.grad_clip,
            opt="adam",
            use_amp=self._use_amp,
        )
        self.critic_opt = tools.Optimizer(
            "critic",
            self.critic.parameters(),
            config.critic.lr,
            config.critic.eps,
            config.critic.grad_clip,
            opt="adam",
            use_amp=self._use_amp,
        )

    def update_slow_critic(self):
        """EMA update of slow target critic."""
        if self.slow_critic is None:
            return
        with torch.no_grad():
            for slow_param, param in zip(
                self.slow_critic.parameters(), self.critic.parameters()
            ):
                slow_param.data.copy_(
                    self._slow_target_fraction * param.data
                    + (1.0 - self._slow_target_fraction) * slow_param.data
                )

    def act(self, feat, eval_mode=False):
        """Sample action from actor given latent feature.

        Args:
            feat: (batch, feat_size) latent feature.
            eval_mode: If True, return mode instead of sample.

        Returns:
            action: (batch, num_actions) sampled action.
        """
        policy = self.actor(feat)
        if eval_mode:
            action = policy.mode()
        else:
            action = policy.sample()
        return action

    def get_value(self, feat, use_slow=False):
        """Get value prediction from critic.

        Args:
            feat: (batch, feat_size) latent feature.
            use_slow: If True, use slow target critic.

        Returns:
            value: (batch,) predicted value (in symlog space, use symexp to decode).
        """
        critic = self.slow_critic if (use_slow and self.slow_critic is not None) else self.critic
        value_dist = critic(feat)
        return value_dist.mode()

    def actor_loss(self, feat, action, advantage, entropy_scale):
        """Compute actor (policy) loss.

        Args:
            feat: (batch, feat_size) latent feature.
            action: (batch, num_actions) action taken.
            advantage: (batch,) advantage values (already normalized).
            entropy_scale: float, entropy bonus coefficient.

        Returns:
            metrics: dict with actor_loss, actor_entropy.
        """
        policy = self.actor(feat)
        log_prob = policy.log_prob(action)
        # REINFORCE-style loss
        actor_loss = -(log_prob * advantage.detach()).mean()
        entropy = policy.entropy().mean()
        total_loss = actor_loss - entropy_scale * entropy

        metrics = {
            "actor_loss": actor_loss.detach().cpu().numpy(),
            "actor_entropy": entropy.detach().cpu().numpy(),
        }
        actor_metrics = self.actor_opt(total_loss, self.actor.parameters())
        metrics.update(actor_metrics)
        return metrics

    def critic_loss(self, feat, target_value):
        """Compute critic (value) loss.

        Args:
            feat: (batch, feat_size) latent feature.
            target_value: (batch,) target value (in original/real space).

        Returns:
            metrics: dict with critic_loss.
        """
        value_dist = self.critic(feat)
        # log_prob expects original-scale value, symlog_disc handles symlog internally
        critic_loss = -value_dist.log_prob(target_value).mean()

        metrics = {}
        critic_metrics = self.critic_opt(critic_loss, self.critic.parameters())
        metrics.update(critic_metrics)
        return metrics
