# Dreamer Branch Imagination Learning Task

## Goal

Change WMP from "world model features as PPO inputs" toward DreamerV3-style learning in imagination. The first implementation keeps the existing WMP path as the baseline and adds a Mode Switch that trains an imagined actor/value branch from RSSM latent states.

## References

- `NM512/dreamerv3-torch` is the closest code reference. This repository already uses the same PyTorch Dreamer lineage for RSSM, world model heads, lambda returns, symlog distributions, and optimizers.
- `danijar/dreamerv3` is the semantic reference for DreamerV3: train a world model from real experience, then train behavior from imagined latent rollouts.
- `carlosferrazza/humanoid-bench` is the robotics benchmark reference. It is useful for continuous-control task framing, but its DreamerV3 training delegates to the upstream Dreamer stack rather than providing a directly portable PyTorch behavior learner.

## Current WMP Path

The current `WMPRunner` collects real Isaac Gym rollouts, updates PPO/AMP from real transitions, stores episode fragments in `wm_dataset`, trains the world model, and feeds `wm_feature` into `ActorCriticWMP`. This is a representation-augmented PPO path.

## Target Dreamer Branch

The new Dreamer Branch keeps real rollouts for data collection, trains the world model from those trajectories, then starts imagined rollouts from posterior RSSM states:

1. Encode sampled real sequences with `WorldModel`.
2. Use posterior latent states as imagined rollout starts.
3. Sample actions from an imagined actor.
4. Step the RSSM forward with `dynamics.img_step`.
5. Predict rewards with `world_model.heads["reward"]`.
6. Train actor and value with lambda returns over imagined features.

## MVP Scope

The first code change intentionally does not replace the environment-control policy. It trains the Dreamer Branch in parallel when enabled, logs imagined behavior metrics, and keeps the existing PPO policy as the behavior policy. This makes the original WMP path a clean comparison.

Replacing PPO control with the imagined actor is a follow-up switch once metrics show the reward head and imagined value learning are stable.

## Mode Switch

Add runner configuration fields:

- `use_imagination_learning`: enables Dreamer Branch training.
- `imagination_replace_ppo`: reserved for the later step where imagined actor controls the environment.
- `imagination_start_after`: minimum world-model dataset size before behavior updates.
- `imagination_updates_per_iter`: number of imagined behavior updates per training iteration.

The MVP can also be enabled from the command line:

```bash
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0 --use_imagination_learning
```

## Required Code Changes

- `dreamer/configs.yaml`: enable reward head learning and add behavior-learning fields.
- `dreamer/behavior.py`: add `ImagBehavior`, adapted from the local Dreamer utilities and RSSM interfaces.
- `dreamer/__init__.py`: export `ImagBehavior`.
- `rsl_rl/runners/wmp_runner.py`: construct `ImagBehavior`, train it after world-model updates, save/load its parameters and optimizers.
- `legged_gym/envs/a1/a1_amp_config.py`: expose the Mode Switch in task config.

## Validation

Lightweight validation:

```bash
python -m compileall dreamer rsl_rl legged_gym
```

Hardware validation:

```bash
python legged_gym/tests/test_env.py --task=a1_amp
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0
```

Expected early signals when `use_imagination_learning=True`:

- `World_model/reward_loss` is finite and trends down.
- `ImagBehavior/actor_loss`, `ImagBehavior/value_loss`, and `ImagBehavior/imag_reward_mean` are finite.
- Existing PPO metrics remain comparable while `imagination_replace_ppo=False`.
