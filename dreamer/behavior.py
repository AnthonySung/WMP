# MIT License
#
# Adapted from NM512/dreamerv3-torch for this repository's WMP Dreamer Branch.

import copy

import torch
from torch import nn

from . import networks
from . import tools


to_np = lambda x: x.detach().cpu().numpy()


class ImagBehavior(nn.Module):
    """Dreamer-style actor/value learning from imagined RSSM rollouts."""

    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        object.__setattr__(self, "_world_model", world_model)
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            device=config.device,
            name="ImagActor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="ImagValue",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0

        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "imag_actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        self._value_opt = tools.Optimizer(
            "imag_value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        self.requires_grad_(False)

    def train_from_posterior(self, start, objective):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat, imag_action, target, weights, base
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat.detach()

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1])
                target = torch.stack(target, dim=1)
                value_loss = -value.log_prob(target.detach())
                if self._config.critic["slow_target"]:
                    slow_target = self._slow_value(value_input[:-1])
                    value_loss -= value.log_prob(slow_target.mode().detach())
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "imag_value"))
        metrics.update(tools.tensorstats(target, "imag_target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["imag_actor_entropy"] = to_np(torch.mean(actor_ent))

        with tools.RequiresGrad(self.actor):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
        with tools.RequiresGrad(self.value):
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            action = policy(feat.detach()).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon, device=self._world_model.device)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(self, imag_feat, imag_action, target, weights, base):
        metrics = {}
        policy = self.actor(imag_feat.detach())
        target = torch.stack(target, dim=1)

        if self._config.imag_gradient == "dynamics":
            actor_target = target - base
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            reinforce_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * (target - base) + (1 - mix) * reinforce_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for src, dst in zip(self.value.parameters(), self._slow_value.parameters()):
                    dst.data = mix * src.data + (1 - mix) * dst.data
            self._updates += 1
