# Dreamer Branch 实现文档

> 版本：v1.3  
> 日期：2026-05-21  
> 基于：WMP 项目 master 分支  
> 状态：核心路径已修正，全部已知小问题已修复，待在 Isaac Gym 环境下验证收敛

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
| `rsl_rl/runners/dreamer_runner.py` | 修正 chunk-step 动作执行语义，接入 AMP discriminator 训练 |
| `rsl_rl/algorithms/dreamer_behavior.py` | 修正 imagined lambda-return 的 slow-critic bootstrap |
| `rsl_rl/storage/dreamer_replay.py` | 补充 episode 尾部 is_terminal 标记 |
| `legged_gym/tests/test_dreamer_behavior.py` | 新增纯 PyTorch 回归测试，锁定 imagined bootstrap 语义 |
| `rsl_rl/modules/__init__.py` | 导出 DreamerActorCritic |
| `rsl_rl/runners/__init__.py` | 导出 DreamerRunner |

---

## 3. 训练流程详解

### 3.1 DreamerRunner 主循环

每个 training iteration 执行：

```
1. Real Rollout（收集数据）
   ├── 每个 chunk-step 起点由 Dreamer actor 从 WM latent feature 采样一整段 chunk action
   ├── 在 chunk 内按顺序执行每个 env-step 对应的 action slice
   ├── WM obs_step 更新 latent state
   ├── 将真实执行过的整段 chunk action 写入 wm_dataset（chunk-step 粒度）
   └── AMP discriminator 在 real side 计算 reward

2. World Model Training（在真实数据上）
   ├── DreamerReplay.sample_batch() 采样序列
   ├── WM._train()：encoder → RSSM → decoder/reward/cont heads
   └── 更新 WM optimizer

3. AMP Auxiliary Update（真实 transition 上）
   ├── 收集 rollout 期间的 `amp_obs -> next_amp_obs`
   ├── 更新独立的 AMP discriminator optimizer
   └── 记录 AMP loss / grad penalty / policy pred / expert pred

4. Behavior Learning（在 latent 空间中）
   ├── 从 replay batch 获取 posterior state
   ├── DreamerBehavior.update()：
   │   ├── imagine_trajectory()：RSSM img_step 展开 imagined rollout
   │   ├── 预测 imagined reward / continuation / value
   │   ├── 用 final imagined state 的 slow critic 作为 λ-return bootstrap
   │   ├── compute_lambda_target()：chunk-step λ-return
   │   ├── update_critic()：latent critic → λ-target
   │   ├── update_actor()：REINFORCE + entropy bonus
   │   └── update_slow_critic()：EMA
   └── 记录 imagined reward / imagined value 等行为学习指标
```

### 3.2 关键设计决策

| 决策 | 说明 |
|------|------|
| **Imagined phase 不走 privileged obs** | Actor/Critic 只消费 RSSM latent feature |
| **AMP 仅 real side** | AMP discriminator 独立 optimizer，imagined reward 来自 WM reward_head |
| **Chunk-step 时间语义** | 1 chunk-step = 5 env-steps（`wm_update_interval=5`），horizon 按 chunk-step 计 |
| **Chunk action 与 dataset 对齐** | 每个 chunk-step 只采样一次整段动作，环境执行与 replay 写入保持一致 |
| **Latent critic 为主 critic** | Symlog-discretized 分布，EMA slow target |
| **视觉链路保留** | DepthPredictor 继续作为 WM 输入预处理器 |

### 3.3 Replay 与 bootstrap 语义

- `DreamerReplay.sample_batch()` 仍以当前 `wm_dataset` 的单 env-slot 单 episode 设计为基础。
- 每个采样子序列的第一个时间步都标记为 `is_first=1`，表示该子序列从独立的 latent reset 开始。
- 当采样窗口命中该 env slot 当前 episode 的尾部时，最后一个时间步会标记为 `is_terminal=1`，用于训练 `cont_head`。
- `DreamerBehavior` 计算 λ-return 时，不再把 horizon 末端 bootstrap 强行置零，而是使用 final imagined state 的 slow critic value。

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
| `--use_camera` | `False`（a1_amp_config） | 启用/关闭深度相机视觉链路。注意：当前 `a1_amp_config.py` 中 `depth.use_camera = False`，开启需先改配置 |
| `--headless` | `False` | 无头模式（不渲染） |
| `--sim_device` | `cuda:0` | 仿真设备 |
| `--num_envs` | 配置文件值 | 并行环境数 |
| `--max_iterations` | 配置文件值 | 最大训练迭代数 |
| `--seed` | 配置文件值 | 随机种子 |

### 4.3 实验矩阵（推荐运行组合）

> **注意**：当前 `a1_amp_config.py` 中 `depth.use_camera = False`，以下命令均为无视觉模式。如需视觉实验，先改配置 `use_camera = True`。

```bash
# 实验 1：WMP 原版（baseline，无视觉）
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0

# 实验 2：Dreamer Branch（无视觉，当前默认配置）
python legged_gym/scripts/train.py --task=a1_amp --training_mode dreamerv3 --headless --sim_device=cuda:0

# 实验 3：Dreamer Branch（有视觉，需先改 a1_amp_config.py 中 depth.use_camera = True）
python legged_gym/scripts/train.py --task=a1_amp --training_mode dreamerv3 --headless --sim_device=cuda:0

# 实验 4：WMP 原版（有视觉对照，需先改 a1_amp_config.py 中 depth.use_camera = True）
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0
```

### 4.4 快速验证命令（确认无 bug）

```bash
# 克隆并运行，观察是否能跑到 Iter 100+ 无 crash
git clone https://github.com/AnthonySung/WMP.git
cd WMP
pip install -r requirements.txt
python legged_gym/scripts/train.py --task=a1_amp --training_mode dreamerv3 --headless --sim_device=cuda:0

# 预期：Iter 3-4 开始 train 时间 > 0，Iter 50+ 稳定无 crash
# 日志格式：Iter N: collect=X.Xs, train=X.Xs, wm_data=XXXXX.X
```

### 4.5 无 Isaac Gym 的逻辑回归测试

如果当前机器没有 GPU 或 Isaac Gym，可以先验证 Dreamer Branch 的纯 PyTorch 核心逻辑：

```bash
# 需要一个可导入 torch 的 Python 环境
python -m pytest legged_gym/tests/test_dreamer_behavior.py -q
```

该测试不依赖 Isaac Gym，只验证 imagined lambda-return 是否正确使用 final imagined state 的 slow-critic bootstrap。

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
| **Behavior 训练** | actor_loss, actor_entropy, critic_loss, imagined_reward, imagined_value | TensorBoard scalar（仅 Dreamer） |
| **AMP 辅助训练** | amp_loss, amp_grad_pen | TensorBoard scalar（仅 Dreamer） |
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
2. **cont_head 刚启用**：首次训练 continuation prediction，`is_terminal` 仅在 episode 末尾标记，中间 chunk-step 的 terminal 语义较粗
3. **Dreamer actor 用于 real rollout**：每 chunk（5 env-steps）采样一次 60-dim 动作，reshape 为 (5,12) 后逐步执行。WM 训练早期 latent 质量不足时可能不稳定
4. **无 privileged bootstrap**：latent critic 已使用 slow critic 对 final imagined state 做 bootstrap，但未引入 privileged critic 作为 real-side support
5. **Chunk-step 语义**：imagined horizon=16 对应 80 env-steps ≈ 0.4s，可能偏短
6. **Actor 每 chunk 采样一次**：chunk 内 5 个 env-step 使用同一 WM feature 采样的不同 action slice，但 feature 本身不更新

### 6.2 已知修复记录（v1.3）

| 问题 | 修复 |
|------|------|
| `critic_loss` 未记录到 metrics | 补充 `metrics["critic_loss"]` |
| `actor_loss` key 与 Optimizer 冲突 | 重命名为 `actor_reinforce_loss` |
| 重复 `if self._discriminator` 检查 | 合并为一个 if 块 |
| `discount`/`lambda_return` 未在 yaml 定义 | 补充到 `configs.yaml` |

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
- [x] Dreamer Branch `use_camera=False` 训练验证 ✅（Iter 137+ 无 crash，核心路径跑通）
- [x] 纯 PyTorch 行为学习回归测试 ✅（imagined bootstrap 逻辑已锁定）
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
# Behavior learning
discount: 0.997          # 折扣因子
lambda_return: 0.95      # λ-return 系数

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
