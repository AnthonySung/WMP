# 控制范式对比：WMP vs DreamerV3 vs MPPI(SimDist)

## 1. 文档定位

本文档系统对比三种控制范式，并分析它们在"world model residual adaptation"这个问题上的差异和适用性。

- WMP (当前项目)：PPO + AMP + wm_feature
- DreamerV3 / DayDreamer：imagination-based actor-critic
- MPPI (SimDist)：model predictive path integral control

### 参考论文与代码

| 方法 | 论文 | 代码 |
|---|---|---|
| **DreamerV3** | [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (Hafner et al., 2023) | [github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3) |
| **DayDreamer** | [DayDreamer: World Models for Physical Robot Learning](https://arxiv.org/abs/2206.14176) (Wu et al., 2022) | [github.com/danijar/daydreamer](https://github.com/danijar/daydreamer) |
| **MPPI (原始)** | [Model Predictive Path Integral Control: From Theory to Parallel Computation](https://arc.aiaa.org/doi/10.2514/1.G003921) (Williams et al., 2017) | — (算法标准，无单一官方实现) |
| **MPPI + World Model (SimDist)** | [Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation](https://arxiv.org/abs/2603.15759) (Levy et al., 2026) | [github.com/CLeARoboticsLab/simdist](https://github.com/CLeARoboticsLab/simdist) |
| **ReDRAW** | [Adapting World Models with Latent-State Dynamics Residuals](https://arxiv.org/abs/2504.02252) (Lanier et al., 2025) | [redraw.jblanier.net](https://redraw.jblanier.net) |

**论文关系说明**：
- DreamerV3 是本项目 WMP 的 world model 架构来源（RSSM）
- DayDreamer 是 DreamerV3 在真机上的应用，证明了 imagination-based 方法在物理机器人上的可行性
- MPPI 原始论文提出了采样-based 模型预测控制算法，SimDist 将其与 world model 结合用于 sim-to-real
- ReDRAW 是本项目 residual adaptation 思路的直接灵感来源

---

## 2. 三种控制方式的架构对比

### 2.1 WMP：PPO + wm_feature（当前方案）

```
训练流程：
┌─────────────────────────────────────────────────────────┐
│  PPO Rollout（真实环境交互）                              │
│    obs → history_encoder → latent_vector(32)             │
│    obs → world_model.encoder → RSSM → wm_feature(256)   │
│    wm_feature → wm_encoder → wm_latent(16)               │
│    actor_input = concat(latent, command, wm_latent)      │
│    action = actor(actor_input)                           │
│    next_obs, reward = env.step(action)  ← 真实环境        │
│                                                          │
│  PPO Update:                                             │
│    advantage = real_reward + γ*V(next) - V(now)          │
│    actor_loss = PPO_clip(log_prob, advantage)            │
│    critic_loss = MSE(V(obs), return)                     │
│                                                          │
│  World Model 训练（独立于 PPO）:                          │
│    wm_buffer → wm_dataset → train_world_model()          │
│    loss = reconstruction + KL + reward_prediction        │
└─────────────────────────────────────────────────────────┘

部署流程：
  obs → world_model → wm_feature → wm_encoder → actor → action
```

**关键特征**：
- actor 输入中 world model 信息被压缩到 16 维
- PPO 用真实环境 reward 计算 advantage
- world model 只是"特征提取器"，不参与 reward 预测和策略评估
- AMP discriminator 提供运动风格奖励（需要真实运动数据）

---

### 2.2 DreamerV3 / DayDreamer：Imagination-based Actor-Critic

```
训练流程：
┌─────────────────────────────────────────────────────────┐
│  Step 1: 在线收集数据                                     │
│    obs → encoder → latent_state                          │
│    action = actor(latent_state)                          │
│    next_obs, reward = env.step(action)  ← 真实环境        │
│    存入 replay buffer                                    │
│                                                          │
│  Step 2: 训练 world model（从 replay buffer 采样）        │
│    obs → encoder → latent_state                          │
│    dynamics: (latent_state, action) → next_latent_state  │
│    decoder: latent_state → reconstructed_obs              │
│    reward_head: latent_state → predicted_reward          │
│    loss = recon_loss + KL_loss + reward_loss             │
│                                                          │
│  Step 3: Imagination 训练 actor-critic                   │
│    start_latent = replay_buffer.sample()                 │
│    for t in range(H):  # H=16                            │
│        action = actor(latent_state)                      │
│        next_latent = dynamics(latent_state, action)      │
│        predicted_reward = reward_head(latent_state)      │
│        predicted_value = critic(latent_state)            │
│                                                          │
│    imagined_return = Σ λᵗ * predicted_reward             │
│    actor_loss = -log_prob * (return - value)  # Reinforce│
│    critic_loss = MSE(value, return)                      │
└─────────────────────────────────────────────────────────┘

部署流程：
  obs → encoder → latent_state → actor → action
```

**关键特征**：
- actor 直接使用原始 latent state（stoch+deter，~288 维），信息量最大
- actor-critic **在 world model 的想象中训练**，不需要真实环境交互
- reward 来自 world model 的 reward head 预测
- 可以"预见"未来多步（imagination horizon H=16）
- DayDreamer = DreamerV3 + 真机 online RL（不需要仿真）

---

### 2.3 MPPI (SimDist)：Model Predictive Path Integral Control

```
训练流程：
┌─────────────────────────────────────────────────────────┐
│  Step 1: 仿真中训练 expert policy（PPO）                  │
│    obs → actor → action → env.step() → reward            │
│    保存多个 checkpoint                                   │
│                                                          │
│  Step 2: 数据生成（mixed-quality）                        │
│    expert policy + non-expert policies + action noise    │
│    记录 (obs, action, reward, value, expert_flag)        │
│                                                          │
│  Step 3: 训练 world model（离线，从数据集）               │
│    obs → encoder → latent_state                          │
│    dynamics: TransformerDecoder(latent, fut_actions)     │
│    reward_head, value_head, policy_head                  │
│    loss = latent_dynamics_MSE + reward_MSE + value_MSE   │
│           + action_MSE (masked by expert_flag)           │
│                                                          │
│  Step 4: 真机 dynamics finetuning                        │
│    冻结 encoder + reward + value + policy                │
│    只训练 dynamics Transformer                           │
│    loss = latent_dynamics_MSE (on real data)             │
└─────────────────────────────────────────────────────────┘

部署流程（MPPI 在线规划）：
  for each control step:
      obs → encoder → latent_state
      # 采样 N 条动作序列
      for i in range(N):
          noise_actions = base_policy_actions + noise
          # 在 world model 中 rollout
          imagined_trajectory = world_model(latent_state, noise_actions)
          score[i] = Σ γᵗ * predicted_reward
      # 选择 elite 动作序列，更新分布
      action = weighted_average(elite_actions)
      执行 action
```

**关键特征**：
- 部署时使用 MPPI **在线规划**（每个控制周期重新规划）
- world model 使用 Transformer Decoder（非 RSSM）
- dynamics 是确定性多步预测（输入未来动作序列，输出未来 latent 序列）
- 不需要在 world model 中训练 actor-critic（policy head 在 world model 内部）
- adaptation 只 finetune dynamics，冻结其他所有组件

---

## 3. 三种控制范式的核心差异总结

| 维度 | WMP (PPO+wm_feature) | DreamerV3 (imagination) | MPPI (SimDist) |
|---|---|---|---|
| **World Model 架构** | RSSM (DreamerV3 同款) | RSSM | Transformer Decoder |
| **World Model 输出** | wm_feature (deter) | latent_state (stoch+deter) | 多步 latent 序列 |
| **Actor 输入** | wm_latent (16维, 压缩后) | latent_state (288维, 原始) | 无独立 actor（MPPI 规划） |
| **Actor 训练方式** | PPO + 真实环境 reward | Reinforce + WM 预测 reward | 不需要训练（WM 内含 policy head） |
| **Critic 训练方式** | PPO + 真实环境 return | MSE + WM 预测 return | 不需要训练（WM 内含 value head） |
| **是否需要环境交互** | 是（PPO on-policy） | 是（数据收集）+ 否（imagination） | 否（离线数据 + MPPI 规划） |
| **多步前瞻** | 无（只看当前 wm_feature） | 有（imagination horizon H=16） | 有（MPPI horizon T，通常更长） |
| **AMP 兼容性** | ✅ 原生支持 | ❌ 需要额外设计 | ❌ 不适用 |
| **Sim-to-Real** | 仿真训练 → 真机部署 | 仿真 pretrain → 真机 adaptation | 仿真 pretrain → 真机 dynamics finetune |
| **residual 传导路径** | dynamics → wm_feature(256→16) → actor | dynamics → imagination → actor | dynamics → MPPI rollout → action |

---

## 4. MPPI vs DreamerV3 在 Residual Adaptation 问题上的区别

这是本项目的核心问题：**哪种控制范式更适合"用 residual 修正 dynamics"这个目标？**

### 4.1 对 World Model 质量的需求不同

```
MPPI:
  world model 需要预测:
    1. future latent states（多步，确定性）
    2. future rewards（用于 scoring）
    3. future values（用于 terminal value estimation）
  
  residual 修正 dynamics 后:
    → MPPI rollout 更准确
    → action scoring 更准确
    → 直接改善 final action 质量
  
  传导路径: residual → dynamics → rollout → action
  传导效率: ★★★★★（最直接）

DreamerV3:
  world model 需要预测:
    1. next latent state（单步，随机）
    2. next reward（单步）
  
  residual 修正 dynamics 后:
    → imagination rollout 更准确
    → imagined return 更准确
    → actor-critic 训练信号更好
    → policy 改善
  
  传导路径: residual → dynamics → imagination → actor/critic → action
  传导效率: ★★★★（较直接）

WMP (PPO+wm_feature):
  world model 需要预测:
    1. next latent state（单步）
    2. next observation（用于 reconstruction loss）
  
  residual 修正 dynamics 后:
    → wm_feature 更准确
    → wm_latent(16维) 更准确
    → actor 输入更好
    → policy 改善
  
  传导路径: residual → dynamics → wm_feature(256) → wm_encoder → wm_latent(16) → actor
  传导效率: ★★（最间接，有信息瓶颈）
```

### 4.2 对 Residual 过拟合的鲁棒性不同

```
MPPI:
  - 使用开环 rollout（给定动作序列，预测未来）
  - residual 错误会在多步 rollout 中累积
  - 对 residual 过拟合非常敏感
  - 需要大量 diverse target data 来保证 residual 泛化

DreamerV3:
  - 使用闭环 imagination（actor 参与 rollout）
  - actor 可以在 imagination 中"修正" residual 的错误
  - 对 residual 过拟合有一定容忍度
  - 数据需求相对较低

WMP:
  - residual 只影响 wm_feature（单步表征）
  - 没有多步累积问题
  - 对 residual 过拟合最不敏感
  - 但 residual 的改善也可能被压缩层削弱
```

### 4.3 数据效率对比

| 维度 | MPPI | DreamerV3 | WMP |
|---|---|---|---|
| **需要 target domain reward** | 不需要 | 不需要（imagination 用 source reward head） | 需要（PPO 用真实 reward） |
| **target domain 数据量** | 几十到几百 episodes | 几十到几百 episodes | 需要在线交互（大量） |
| **offline adaptation 可行性** | ✅ 天然支持 | ✅ 天然支持 | ❌ 需要在线环境 |
| **真机部署复杂度** | 需要实时 MPPI 规划（计算量大） | 只需 encoder + actor（轻量） | 只需 encoder + wm_encoder + actor（轻量） |

### 4.4 在四足机器人场景下的适用性

```
MPPI 的优势：
  + 不需要训练 actor-critic（WM 内含 policy/value head）
  + dynamics finetune 后立即改善规划质量
  + 适合需要精确控制的场景（如 manipulation）
  
MPPI 的劣势：
  - 每个控制周期需要多次 WM 推理（N×T 次，计算量大）
  - 四足机器人控制频率高（50-100Hz），实时 MPPI 有挑战
  - 对 WM 质量要求极高（多步 rollout 不能漂移）
  - SimDist 在四足上的实验有限（仅 slippery slope 和 foam）

DreamerV3 的优势：
  + 部署时只需 encoder + actor（推理快）
  + imagination 训练可以离线进行
  + DayDreamer 已在真机上验证（虽非四足）
  + 闭环 imagination 对 WM 误差有一定容忍度
  
DreamerV3 的劣势：
  - 需要实现完整的 imagination training loop
  - 需要 source domain 有 reward（用于 pretrain reward head）
  - AMP 风格的运动奖励难以在 imagination 中实现

WMP 的优势：
  + 已有完整的 PPO + AMP 训练链路
  + 部署简单（encoder + wm_encoder + actor）
  + AMP 提供运动风格约束
  
WMP 的劣势：
  - residual 改善传导路径最长
  - 需要 target domain 在线交互（PPO 是 on-policy）
  - wm_feature 压缩（256→16）可能丢失信息
```

---

## 5. 建议的演进路径

基于以上分析，建议的演进路径是：

```
Phase A（当前）:
  WMP (PPO + AMP + wm_feature)
  验证 residual 结构本身是否有效
  
Phase A 对照:
  DreamerV3 imagination（独立脚本）
  验证 residual 在直接控制路径中的效果
  判断 wm_feature 传导是否有瓶颈

Phase B:
  如果 imagination 显著优于 wm_feature:
    → 逐步将 imagination 集成到主训练链路
    → 保留 AMP（作为辅助奖励或去掉）
  如果 wm_feature 已经足够好:
    → 继续使用当前架构
    → 重点优化 residual adaptation 流程

Phase C（真机）:
  优先使用 DreamerV3 部署模式（encoder + actor）
  如果计算资源允许，可以尝试 MPPI
  但 MPPI 在四足上的实时性需要验证
```

---

## 6. 一句话总结

> **DreamerV3 的 imagination 模式是 WMP 向 world model-based control 演进的最自然路径**（因为已经共享 RSSM 架构），而 **MPPI 更适合对 WM 质量有极高信心且计算资源充足的场景**。对于当前的 residual adaptation 研究，DreamerV3 imagination 对照实验是性价比最高的下一步。
