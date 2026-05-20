# Dreamer Branch 实现文档

> 版本：v1.0  
> 日期：2026-05-20  
> 基于：WMP 项目 master 分支，commit `b5954cf`

---

## 1. 概述

本项目在保留原始 **WMP**（World Model-based Perception）训练主线的同时，新增了一条独立的 **Dreamer Branch** 训练路线。

### 两条路线的关系

| | WMP（原版） | Dreamer Branch（新增） |
|---|---|---|
| **策略优化** | PPO + AMP，在真实环境 rollout 上更新 | Latent imagination 中训练 actor/critic |
| **Critic** | Privileged critic（消费 privileged obs + WM feature） | Latent critic（仅消费 RSSM latent feature） |
| **AMP** | 参与 PPO 联合 loss | 仅 real rollout side，独立 optimizer |
| **视觉** | DepthPredictor → WM encoder | 同 WMP（复用） |
| **World Model** | 提供 latent feature 辅助控制器 | 提供 latent dynamics 用于 imagination |

### 切换方式

通过配置或 CLI 一键切换，**不需要修改任何代码**：

```bash
# 方式 1：CLI（推荐）
python legged_gym/scripts/train.py --task=a1_amp --training_mode dreamerv3 --headless --sim_device=cuda:0

# 方式 2：配置文件
# 编辑 legged_gym/envs/a1/a1_amp_config.py，设置 training_mode = 'dreamerv3'
```

切回原版：

```bash
python legged_gym/scripts/train.py --task=a1_amp --training_mode wmp --headless --sim_device=cuda:0
```

---

## 2. 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                      train.py 入口                           │
│                 --training_mode wmp|dreamerv3                │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   ┌─────────────┐         ┌────────────────┐
   │  WMPRunner  │         │ DreamerRunner  │
   │  (原版不变)  │         │   (新增)        │
   └─────────────┘         └───────┬────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
           ┌─────────────┐ ┌────────────┐ ┌──────────────┐
           │DreamerReplay│ │Dreamer     │ │DreamerActor  │
           │ (数据采样)   │ │Behavior    │ │Critic        │
           └─────────────┘ │(想象训练)  │ │(latent actor │
                           └────────────┘ │ + critic)    │
                                          └──────────────┘
```

### 新增文件

| 文件 | 作用 |
|------|------|
| `rsl_rl/runners/dreamer_runner.py` | Dreamer 训练主循环 |
| `rsl_rl/algorithms/dreamer_behavior.py` | Latent imagination + actor/critic 更新 |
| `rsl_rl/modules/dreamer_actor_critic.py` | Latent-only actor + symlog critic + slow target |
| `rsl_rl/storage/dreamer_replay.py` | Chunk-step replay 采样 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `legged_gym/envs/base/legged_robot_config.py` | 新增 `training_mode` 字段 |
| `legged_gym/envs/a1/a1_amp_config.py` | 新增 Dreamer 专用配置块 |
| `legged_gym/scripts/train.py` | 自动检测 training_mode，设置 run_name |
| `legged_gym/utils/task_registry.py` | WM config 加载 + mode switch 分发 |
| `legged_gym/utils/helpers.py` | 新增 `--training_mode` / `--use_camera` CLI 参数 |
| `dreamer/configs.yaml` | reward_head loss_scale 0→1，grad_heads 加入 cont |
| `dreamer/models.py` | 启用 cont_head，cont 张量 shape 修复 |
| `rsl_rl/runners/wmp_runner.py` | WM config 去 sys.argv 化，is_terminal 推导，depth predictor guard |
| `rsl_rl/modules/__init__.py` | 导出 DreamerActorCritic |
| `rsl_rl/runners/__init__.py` | 导出 DreamerRunner |

---

## 3. 训练流程详解

### 3.1 DreamerRunner 主循环

每个 training iteration 执行：

```
1. Real Rollout（收集数据）
   ├── Dreamer actor 从 WM latent feature 采样 action
   ├── env.step() 执行动作
   ├── WM obs_step 更新 latent state
   ├── 写入 wm_dataset（chunk-step 粒度）
   └── AMP discriminator 计算 reward（real side only）

2. World Model Training（在真实数据上）
   ├── DreamerReplay.sample_batch() 采样序列
   ├── WM._train()：encoder → RSSM → decoder/reward/cont heads
   └── 更新 WM optimizer

3. Behavior Learning（在 latent 空间中）
   ├── 从 replay batch 获取 posterior state
   ├── DreamerBehavior.update()：
   │   ├── imagine_trajectory()：RSSM img_step 展开 imagined rollout
   │   ├── 预测 imagined reward / continuation / value
   │   ├── compute_lambda_target()：λ-return
   │   ├── update_critic()：latent critic → λ-target
   │   ├── update_actor()：REINFORCE + entropy bonus
   │   └── update_slow_critic()：EMA
   └── (可选) AMP discriminator 更新（real side）
```

### 3.2 关键设计决策

| 决策 | 说明 |
|------|------|
| **Imagined phase 不走 privileged obs** | Actor/Critic 只消费 RSSM latent feature |
| **AMP 仅 real side** | AMP discriminator 独立 optimizer，imagined reward 来自 WM reward_head |
| **Chunk-step 时间语义** | 1 chunk-step = 5 env-steps（`wm_update_interval=5`），horizon 按 chunk-step 计 |
| **Latent critic 为主 critic** | Symlog-discretized 分布，EMA slow target |
| **视觉链路保留** | DepthPredictor 继续作为 WM 输入预处理器 |

---

## 4. 运行指令

### 4.1 基础运行

```bash
# WMP 原版（默认）
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0

# Dreamer Branch
python legged_gym/scripts/train.py --task=a1_amp --training_mode dreamerv3 --headless --sim_device=cuda:0
```

### 4.2 常用 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | `a1_amp` | 任务名称 |
| `--training_mode` | `wmp` | `wmp` 或 `dreamerv3` |
| `--use_camera` | `True` | 启用/关闭深度相机视觉链路 |
| `--headless` | `False` | 无头模式（不渲染） |
| `--sim_device` | `cuda:0` | 仿真设备 |
| `--num_envs` | 配置文件值 | 并行环境数 |
| `--max_iterations` | 配置文件值 | 最大训练迭代数 |
| `--seed` | 配置文件值 | 随机种子 |

### 4.3 实验矩阵（推荐运行组合）

```bash
# 实验 1：WMP 原版（baseline）
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0

# 实验 2：Dreamer Branch（有视觉）
python legged_gym/scripts/train.py --task=a1_amp --training_mode dreamerv3 --headless --sim_device=cuda:0

# 实验 3：Dreamer Branch（无视觉，纯 proprioception）
python legged_gym/scripts/train.py --task=a1_amp --training_mode dreamerv3 --use_camera False --headless --sim_device=cuda:0

# 实验 4：WMP 原版（无视觉对照）
python legged_gym/scripts/train.py --task=a1_amp --use_camera False --headless --sim_device=cuda:0
```

---

## 5. 评测方案

### 5.1 评测指标

| 指标类别 | 具体指标 | 数据来源 |
|----------|---------|---------|
| **Locomotion 性能** | Mean episode reward, mean episode length | TensorBoard scalar |
| **速度跟踪** | tracking_lin_vel, tracking_ang_vel reward | TensorBoard scalar |
| **能耗** | torques reward, dof_acc reward | TensorBoard scalar |
| **AMP 模仿质量** | AMP policy pred, AMP expert pred | TensorBoard scalar |
| **WM 预测质量** | reward_loss, cont_loss, kl, image_loss | TensorBoard scalar |
| **Behavior 训练** | actor_loss, actor_entropy, critic_loss | TensorBoard scalar（仅 Dreamer） |
| **训练效率** | Computation steps/s, iteration time | 控制台输出 |

### 5.2 对比实验设计

| 对比 | 命令 | 目的 |
|------|------|------|
| **WMP vs Dreamer（有视觉）** | 实验 1 vs 实验 2 | 核心对比：PPO+AMP vs Latent Imagination |
| **Dreamer 有/无视觉** | 实验 2 vs 实验 3 | 消融：视觉对 Dreamer Branch 的贡献 |
| **WMP 有/无视觉** | 实验 1 vs 实验 4 | 消融：视觉对 WMP 原版的贡献 |
| **跨方法无视觉对比** | 实验 3 vs 实验 4 | 纯 proprioception 下两种范式的对比 |

### 5.3 训练时长建议

- 每个实验至少 **10,000 iterations**（约 11 小时 @ 3.9s/iter）
- 推荐 **20,000 iterations** 以获得稳定收敛
- 前 100 iterations 的 WM 数据积累期（`train_start_steps=10000` 对应约 100 iter × 24 steps × 4096 envs / 5 interval ≈ 200K chunk-steps），此时 behavior learning 不会启动

### 5.4 结果记录模板

```
实验编号: ___
训练模式: wmp / dreamerv3
视觉: 有 / 无
随机种子: ___
训练迭代数: ___

最终指标（最后 100 iter 平均）:
- Mean episode reward: ___
- Mean episode length: ___
- tracking_lin_vel reward: ___
- tracking_ang_vel reward: ___
- torques reward: ___
- AMP expert pred: ___
- WM reward_loss: ___
- WM cont_loss: ___
- (Dreamer only) actor_loss: ___
- (Dreamer only) critic_loss: ___

训练耗时: ___ 小时
GPU 显存: ___ GB
```

---

## 6. 已知限制与注意事项

### 6.1 当前限制

1. **reward_head 刚启用**：`loss_scale` 从 0 改为 1，需要观察 reward prediction 是否收敛
2. **cont_head 刚启用**：首次训练 continuation prediction，可能需要调 `loss_scale`
3. **Dreamer actor 用于 real rollout**：当前 DreamerRunner 在 real rollout 中用 Dreamer actor（从 WM latent 采样），而非 PPO actor。这在 WM 训练早期可能不稳定
4. **无 privileged bootstrap**：当前 Dreamer Branch 的 latent critic 完全不走 privileged info，bootstrap 功能尚未实现
5. **Chunk-step 语义**：imagined horizon=16 对应 80 env-steps ≈ 0.4s，可能偏短

### 6.2 如果训练不稳定

1. **降低 behavior learning rate**：编辑 `dreamer/configs.yaml` 中 `actor.lr` 和 `critic.lr`
2. **增加 imagined horizon**：`a1_amp_config.py` 中 `dreamer_imagined_horizon`
3. **调整 entropy scale**：`dreamer/configs.yaml` 中 `actor.entropy`
4. **暂时关闭 cont_head**：`grad_heads` 中移除 `cont`，`loss_scale` 改回 0
5. **回退到 WMP**：`--training_mode wmp`

---

## 7. 后续工作计划

### Phase 1：基础验证（当前阶段）

- [x] WMP 原版 `use_camera=False` 训练验证 ✅
- [ ] Dreamer Branch `use_camera=False` 训练验证
- [ ] Dreamer Branch `use_camera=True` 训练验证
- [ ] 观察 reward_head / cont_head loss 收敛曲线
- [ ] 对比 WMP vs Dreamer 的 reward 曲线

### Phase 2：消融实验

- [ ] AMP 消融：`dreamer_use_amp_aux=False` vs `True`
- [ ] 视觉消融：有/无 camera 对比
- [ ] Horizon 消融：imagined_horizon=8/16/32
- [ ] Entropy scale 调参

### Phase 3：功能增强

- [ ] 实现 privileged bootstrap（real rollout 端用 privileged critic 提供更好的 value target）
- [ ] 实现 AMP reward 的 latent-space 版本（可选）
- [ ] 添加 Dreamer Branch 专用的 TensorBoard 日志面板
- [ ] Checkpoint 兼容性：支持从 WMP checkpoint 热启动 Dreamer Branch

### Phase 4：性能优化与发布

- [ ] 多 GPU 训练支持
- [ ] 推理速度 benchmark（actor 在 real rollout 中的延迟）
- [ ] 整理最终实验报告
- [ ] 更新 README 和论文

---

## 8. 附录：关键配置速查

### dreamer/configs.yaml（行为学习相关）

```yaml
actor:
  layers: 2
  dist: 'normal'
  entropy: 3e-4       # 熵系数，调大 → 更多探索
  lr: 3e-5            # actor 学习率
  grad_clip: 100.0

critic:
  layers: 2
  dist: 'symlog_disc'
  slow_target: True
  slow_target_fraction: 0.02  # EMA 系数
  lr: 3e-5            # critic 学习率

reward_head:
  loss_scale: 1.0     # 刚启用，观察收敛

cont_head:
  loss_scale: 1.0     # 刚启用，观察收敛
```

### a1_amp_config.py（Dreamer 专用）

```python
training_mode = 'dreamerv3'   # 或 'wmp'
dreamer_imagined_horizon = 16 # imagined rollout 长度（chunk-steps）
dreamer_use_amp_aux = True    # 保留 AMP 在 real side
dreamer_use_privileged_bootstrap = True  # 保留 privileged critic（待实现）
dreamer_use_camera = True     # 保留视觉链路
```
