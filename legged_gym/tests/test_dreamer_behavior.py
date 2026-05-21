import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from rsl_rl.algorithms.dreamer_behavior import DreamerBehavior  # noqa: E402


class _DummyDist:
    def __init__(self, value):
        self._value = value

    def mode(self):
        return self._value


class _DummyDynamics:
    def get_feat(self, state):
        return state["feat"]

    def img_step(self, state, action, sample=True):
        return {"feat": state["feat"] + 1.0}


class _DummyWorldModel:
    def __init__(self, reward_value, cont_value):
        self.dynamics = _DummyDynamics()
        self.heads = {
            "reward": lambda feat: _DummyDist(torch.full((feat.shape[0], 1), reward_value)),
            "cont": lambda feat: _DummyDist(torch.full((feat.shape[0], 1), cont_value)),
        }


class _DummyActorCritic:
    def act(self, feat, eval_mode=False):
        return torch.zeros((feat.shape[0], 3))

    def get_value(self, feat, use_slow=False):
        value = feat[:, :1]
        return value + (10.0 if use_slow else 0.0)

    def critic_loss(self, feat, target_value):
        return {"critic_loss": float(target_value.mean())}

    def actor_loss(self, feat, action, advantage, entropy_scale):
        return {"actor_loss": float(advantage.mean())}

    def update_slow_critic(self):
        return None


def test_lambda_target_uses_final_bootstrap_value():
    config = SimpleNamespace(
        imagined_horizon=3,
        discount=1.0,
        lambda_return=1.0,
        actor={"entropy": 0.0},
    )
    behavior = DreamerBehavior(
        _DummyWorldModel(reward_value=1.0, cont_value=1.0),
        _DummyActorCritic(),
        config,
    )

    post_state = {"feat": torch.zeros((2, 1))}

    metrics = behavior.update(post_state)

    # For horizon 3 with rewards all ones and final slow bootstrap value 13,
    # the first-step lambda target is 1 + 1 + 1 + 13 = 16.
    assert metrics["critic_loss"] == 15.0
