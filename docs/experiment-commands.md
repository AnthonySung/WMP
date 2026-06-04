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
