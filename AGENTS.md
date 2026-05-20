# Repository Guidelines

## Project Structure & Module Organization
`legged_gym/` contains environments, configs, utilities, and the main entry scripts in `legged_gym/scripts/`. `rsl_rl/` holds RL algorithms, runners, storage, and model modules. `dreamer/` contains the world-model and Dreamer-specific components. Use `datasets/` for motion data, `resources/robots/` for robot assets, and `docs/` plus `CONTEXT.md` for design notes and project terminology.

## Build, Test, and Development Commands
Use Python 3.8 when possible.

- `pip install -r requirements.txt`: install Python dependencies after PyTorch and Isaac Gym Preview 3 are installed.
- `python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0`: start WMP training.
- `python legged_gym/scripts/play.py --task=a1_amp --sim_device=cuda:0 --terrain=climb`: run visualization against a trained policy.
- `python legged_gym/tests/test_env.py --task=a1_amp`: smoke-test environment creation and stepping.

This project does not ship a `Makefile` or pinned formatter config; prefer small, reproducible commands in PR descriptions.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, and concise inline comments only where logic is not obvious. Match current naming patterns such as `*_config.py` for environment/config modules and `*_runner.py` for training flows. Keep imports explicit and changes localized; avoid broad refactors unless required by the task.

## Testing Guidelines
Tests are lightweight and hardware-aware. Add focused smoke or regression tests near the affected module, using `test_*.py` naming under `legged_gym/tests/` or the nearest package test directory you introduce. For training-path changes, include the exact command used to validate behavior; if Isaac Gym or CUDA prevents full execution, state that clearly in the PR.

## Commit & Pull Request Guidelines
Recent history uses short Conventional Commit-style subjects, especially `fix: ...`. Prefer `fix:`, `feat:`, or `refactor:` with an imperative summary. PRs should include scope, linked issues, reproduction or validation commands, and screenshots or short videos for visualization changes. Call out config changes, GPU assumptions, and any dataset/checkpoint files that must not be committed.

## Domain Language
Use the terms in `CONTEXT.md` consistently: `WMP` for the primary path, `Dreamer Branch` for the experimental alternative, and `Mode Switch` for explicit path selection.
