# WMP / Dreamer 三模式实验指令

## Goal

验证三种训练方案都能训练出可评估策略：

1. `wmp`：原始 WMP，world model 作为表征，生成 feature 给 PPO/AMP。
2. `align`：DreamerV3 风格，基于 world model latent 在 imagination 中训练 actor，并由 Dreamer actor 控制真实环境。
3. `takeover`：先 PPO warmup，再逐步提高 Dreamer actor 控制概率，最终接管。

`train.py` 现在默认 `--task=a1_amp`，Isaac Gym 参数默认解析为 `--sim_device=cuda:0`。五卡服务器建议用 `CUDA_VISIBLE_DEVICES` 绑定物理卡；绑定后进程内部仍写或默认使用 `cuda:0`，这是正常的重映射。

## Prop-Only 三组主实验

默认建议先跑这三组。`align` 和 `takeover` 默认 `dreamer_use_image=False`，不会采集 depth image，适合验证“在想象中学习”的核心路径。

### 1. 原始 WMP baseline

```bash
CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py \
  --headless \
  --wmp_training_mode=wmp
```

日志目录：`logs/a1_amp_example/WMP_wmp`

### 2. DreamerV3 aligned

```bash
CUDA_VISIBLE_DEVICES=1 python legged_gym/scripts/train.py \
  --headless \
  --wmp_training_mode=align
```

日志目录：`logs/a1_amp_example/WMP_align`

### 3. PPO-to-Dreamer takeover

```bash
CUDA_VISIBLE_DEVICES=2 python legged_gym/scripts/train.py \
  --headless \
  --wmp_training_mode=takeover
```

日志目录：`logs/a1_amp_example/WMP_takeover`

## Prop-Only 评估

```bash
CUDA_VISIBLE_DEVICES=3 python legged_gym/scripts/play.py \
  --terrain=climb \
  --wmp_training_mode=wmp
```

```bash
CUDA_VISIBLE_DEVICES=3 python legged_gym/scripts/play.py \
  --terrain=climb \
  --wmp_training_mode=align
```

```bash
CUDA_VISIBLE_DEVICES=3 python legged_gym/scripts/play.py \
  --terrain=climb \
  --wmp_training_mode=takeover
```

## 视觉版训练

视觉版只建议作为第二阶段对比，因为 depth camera 采集会明显变慢。`--dreamer_use_image` 会保留相机并让 Dreamer world model 使用 `prop + image` 输入。代码会把 `align/takeover` 视觉版的 `num_envs` 限制到 `camera_num_envs`，避免没有相机的环境写入空图像轨迹。

```bash
CUDA_VISIBLE_DEVICES=3 python legged_gym/scripts/train.py \
  --headless \
  --wmp_training_mode=align \
  --dreamer_use_image
```

日志目录：`logs/a1_amp_example/WMP_align_image`

```bash
CUDA_VISIBLE_DEVICES=4 python legged_gym/scripts/train.py \
  --headless \
  --wmp_training_mode=takeover \
  --dreamer_use_image
```

日志目录：`logs/a1_amp_example/WMP_takeover_image`

原始 `wmp` 本身就是视觉 WMP baseline；如果想显式放到第五张卡重跑：

```bash
CUDA_VISIBLE_DEVICES=4 python legged_gym/scripts/train.py \
  --headless \
  --wmp_training_mode=wmp
```

## 视觉版评估

play 已支持三类模型：

1. `wmp`：加载 PPO/WMP policy，默认打开 depth camera。
2. `align`：加载 Dreamer actor，默认 prop-only；加 `--dreamer_use_image` 后加载 `WMP_align_image` 并输入 depth image。
3. `takeover`：加载 Dreamer actor，默认 prop-only；加 `--dreamer_use_image` 后加载 `WMP_takeover_image` 并输入 depth image。

```bash
CUDA_VISIBLE_DEVICES=3 python legged_gym/scripts/play.py \
  --terrain=climb \
  --wmp_training_mode=align \
  --dreamer_use_image
```

```bash
CUDA_VISIBLE_DEVICES=3 python legged_gym/scripts/play.py \
  --terrain=climb \
  --wmp_training_mode=takeover \
  --dreamer_use_image
```

如需评估自定义目录：

```bash
CUDA_VISIBLE_DEVICES=3 python legged_gym/scripts/play.py \
  --terrain=climb \
  --wmp_training_mode=align \
  --dreamer_use_image \
  --load_run=WMP_align_image
```

## 建议记录指标

- TensorBoard `Train/mean_reward`
- TensorBoard `Train/mean_episode_length`
- `World_model/reward_loss`
- `World_model/cont_loss`
- `ImagBehavior/imag_actor_loss`
- `ImagBehavior/imag_value_loss`
- `DreamerMode/p_dreamer`
- play 输出的 `total reward`

## Notes

- 如果命令不带 `--dreamer_use_image`，`align`/`takeover` 不应再出现 `acquiring depth image time`。
- 如果带 `--dreamer_use_image`，出现 `acquiring depth image time` 是预期行为。
- 三组主实验建议优先跑 `wmp/align/takeover` prop-only，对齐后再跑视觉版 `align_image/takeover_image`。

## Code Review Fixes In Dreamer Modes

当前 `dreamer` 分支已按 code review 修复 Dreamer replay 和 takeover 语义：

- Dreamer replay 存储的是 pre-step transition：`obs[t]`, `action[t]`, `reward[t]`, `done[t]`，不再把 `obs[t+1]` 和 `action[t]` 错位配对。
- replay 中保存真实 `is_first`，batch 采样不再人为把每个窗口第 0 步都标成 episode start。
- takeover 使用 episode-level controller assignment，reset 时按 `p_dreamer` 采样该 episode 由 PPO 或 Dreamer 控制。
- takeover 的 PPO 更新使用 `valid_mask`，只用 PPO-controlled transition 更新 PPO/AMP，Dreamer-controlled transition 只进入 Dreamer world model replay。
- follow-up review 中提到的 RolloutStorage overflow 是误报：`AMPPPO.update()` 正常路径和空 mask 路径都会调用 `storage.clear()`，云端多轮 takeover 已验证无 overflow。
- Dreamer world model / behavior metrics 现在按本轮所有 update 求均值记录，不再只保留最后一个 batch。
- takeover 在当前 step 没有 PPO-controlled env 时会跳过 PPO actor action 计算，减少无用开销。

## 单 GPU 云端 Smoke Test

云端路径：`/home/WMP`。该云 GPU 只有一张卡，使用 `CUDA_VISIBLE_DEVICES=0`，Python 使用 `/root/miniconda3/bin/python`。

同步代码：

```bash
cd /home/WMP
git fetch origin dreamer
git reset --hard origin/dreamer
```

语法检查：

```bash
cd /home/WMP
/root/miniconda3/bin/python -m compileall \
  legged_gym/scripts/train.py \
  legged_gym/scripts/play.py \
  legged_gym/utils/helpers.py \
  legged_gym/utils/task_registry.py \
  rsl_rl/runners/wmp_runner.py
```

非视觉 Dreamer aligned 短跑：

```bash
cd /home/WMP
CUDA_VISIBLE_DEVICES=0 timeout 180 /root/miniconda3/bin/python legged_gym/scripts/train.py \
  --headless \
  --wmp_training_mode=align \
  --num_envs=32 \
  --max_iterations=2 \
  --sim_device=cuda:0
```

已验证输出包含：

```text
Dreamer mode align iter 0: p_dreamer=1.000
Dreamer mode align iter 1: p_dreamer=1.000, dataset=166
```

非视觉 takeover 短跑：

```bash
cd /home/WMP
CUDA_VISIBLE_DEVICES=0 timeout 180 /root/miniconda3/bin/python legged_gym/scripts/train.py \
  --headless \
  --wmp_training_mode=takeover \
  --num_envs=32 \
  --max_iterations=2 \
  --sim_device=cuda:0
```

已验证输出包含：

```text
Dreamer mode takeover iter 0: p_dreamer=0.000
Dreamer mode takeover iter 1: p_dreamer=0.000, dataset=28
```

强制 takeover mask / 接管路径短跑：

```bash
cd /home/WMP
CUDA_VISIBLE_DEVICES=0 timeout 240 /root/miniconda3/bin/python legged_gym/scripts/train.py \
  --headless \
  --wmp_training_mode=takeover \
  --num_envs=32 \
  --max_iterations=3 \
  --dreamer_control_start_after=0 \
  --dreamer_takeover_iters=1 \
  --sim_device=cuda:0 \
  --run_name=WMP_takeover_mask_smoke
```

已验证输出包含：

```text
Dreamer mode takeover iter 0: p_dreamer=0.000
Dreamer mode takeover iter 1: p_dreamer=1.000
Dreamer mode takeover iter 2: p_dreamer=1.000
```

非视觉 aligned play：

```bash
cd /home/WMP
CUDA_VISIBLE_DEVICES=0 timeout 180 /root/miniconda3/bin/python legged_gym/scripts/play.py \
  --headless \
  --wmp_training_mode=align \
  --terrain=climb \
  --sim_device=cuda:0
```

已验证会加载 `logs/a1_amp_example/WMP_align/model_2.pt` 并输出 `total reward`。

非视觉 takeover play：

```bash
cd /home/WMP
CUDA_VISIBLE_DEVICES=0 timeout 180 /root/miniconda3/bin/python legged_gym/scripts/play.py \
  --headless \
  --wmp_training_mode=takeover \
  --terrain=climb \
  --sim_device=cuda:0
```

已验证会加载 `logs/a1_amp_example/WMP_takeover/model_2.pt` 并输出 `total reward`。

该云 GPU 的 Isaac Gym depth camera/headless 图形栈会在视觉路径 core dump，原始 `wmp` 视觉和 `--dreamer_use_image` 都会触发。因此云端 smoke test 只验证 prop-only 的 `align/takeover` 路径；视觉实验留给本地可正常采集 depth 的机器。
