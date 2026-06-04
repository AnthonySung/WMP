# WMP Hybrid Dreamer Takeover 实验方案（第二阶段）

## Goal

这份文档不再作为第一版主方案，而是重新定位为：

- 在 `docs/wmp-dreamer-aligned-plan.md` 跑通之后
- 基于已经可工作的 Dreamer 主学习闭环
- 进一步探索如何与当前 WMP / PPO / AMP 体系做更平滑的接管实验

换句话说：

- `wmp-dreamer-aligned-plan.md` 是 **Dreamer 对齐主线 / MVP**
- 本文档是 **后续 hybrid takeover experiment**

## Positioning

第一版不追求：

- PPO / Dreamer 同 run 混合控制
- controller schedule
- PPO teacher distillation
- AMP final reward 直接作为 world model reward label

这些内容都不属于官方 DreamerV3 主线，更适合作为第二阶段实验。

因此，本文档只回答一个问题：

> 当 Dreamer 对齐主线已经稳定后，如何进一步做“从现有 WMP/PPO 平滑过渡到 Dreamer 主控制”的扩展实验？

## Relationship To The Mainline Plan

主线方案见：

- [docs/wmp-dreamer-aligned-plan.md](docs/wmp-dreamer-aligned-plan.md)

主线 MVP 的边界应保持为：

1. `prop-only`
2. per-step action world model
3. per-step dataset：`prop`, `action`, `reward`, `is_first`, `is_terminal`
4. `cont_head`
5. 真实数据训练 world model
6. posterior latent 上 imagined actor/value 学习
7. Dreamer actor 直接控制真实环境

只有当这条主线稳定后，再开始本文档中的 takeover 实验。

## Preconditions

进入 hybrid takeover experiment 之前，建议至少满足以下条件：

1. `dreamer_control_mode=True` 的 Dreamer-only 训练可以正常运行
2. world model 的 per-step 建模稳定
3. reward head 对稳定环境 reward 的拟合稳定
4. `cont_head` 正常工作
5. Dreamer actor 已经能直接驱动真实环境，不依赖 PPO mixed-control

如果这些前提还未满足，不建议进入本文档的实验阶段。

## Problems This Experiment Must Solve

hybrid takeover 不是简单地把 PPO action 和 Dreamer action 混在一个 batch 里。它至少会引入下面几个新问题：

1. PPO 的 on-policy 假设会被 Dreamer-controlled transition 破坏。
2. Dreamer actor 控制真实环境后，world model replay 的数据分布会快速变化。
3. AMP discriminator 如果继续更新，会让 AMP reward label 非平稳。
4. PPO teacher distillation 可能压制 Dreamer actor 自己通过 imagined return 学到的策略。
5. episode-level controller mixing 会让日志、checkpoint、失败归因都更复杂。

这些问题是第二阶段实验的核心，不应被隐藏在主线 MVP 里。

## Experiment Objectives

第二阶段实验的目标不是“对齐官方 Dreamer”，而是回答下面这些工程问题：

1. 是否可以在保留现有 PPO 基础设施的同时，让 Dreamer 更平滑地接管控制？
2. controller mixing 是否比直接切 Dreamer-only 更稳？
3. PPO teacher distillation 是否能降低接管早期的不稳定？
4. AMP reward 是否值得迁移进 Dreamer 主线？

这些目标都属于 **工程迁移优化**，而不是第一版算法对齐。

## Recommended Experiment Scope

## 1. Episode-level Controller Mixing

如果要做 hybrid takeover，仍建议优先采用 **按 episode 切 controller**，而不是按 step 切换。

原因：

- locomotion 对控制连续性敏感
- 每步切 controller 更容易引入 gait 抖动
- episode 级切换更利于统计 PPO vs Dreamer 的真实表现

实验形式：

- 某些 episode 由 PPO 控制
- 某些 episode 由 Dreamer 控制
- 不建议第一版实验就做 step-level mixing

## 2. Takeover Schedule

在 hybrid 实验中，可以引入 `p_dreamer` 之类的接管概率调度，例如：

- 前期 PPO 占主导
- 中期 PPO / Dreamer 混合
- 后期 Dreamer 占主导

但这类 schedule 只应在第二阶段中出现，而不进入主线 MVP。

建议新增实验配置，例如：

```python
dreamer_takeover_experiment = False
dreamer_control_start_after = 10000
dreamer_takeover_iters = 5000
dreamer_control_schedule = "linear"
```

默认应保持关闭，避免污染主线配置。

## 3. PPO → Dreamer Distillation

在 hybrid 实验中，可以加入轻量 teacher-student 蒸馏：

- teacher：PPO actor
- student：Dreamer actor

目标是帮助 Dreamer actor 在 takeover 早期更快靠近 PPO 已验证过的有效动作分布。

建议第一版实验只做最简单的形式：

```text
L_distill = ||a_dreamer_mean - a_ppo_mean||^2
```

并让其权重与 takeover schedule 挂钩，例如：

```text
lambda_distill = lambda_distill_init * (1 - p_dreamer)
```

注意：

- distillation 只属于 hybrid experiment
- 不应进入 Dreamer 对齐 MVP

## 4. AMP Reward Migration Experiment

评审结论已经明确：

- **不建议第一版直接使用 AMP final reward 训练 world model reward head**

原因：

- discriminator 持续更新
- 历史 transition 的 reward label 非平稳
- world model 更适合稳定监督目标

但在第二阶段，可以专门设计 AMP migration experiment，探索以下路线：

### 方案 A：冻结 discriminator 后再迁移

做法：

1. 先用环境 reward 跑通 Dreamer 主线
2. 当 PPO baseline 或 AMP 模块稳定后，冻结 discriminator
3. 用冻结后的 AMP reward 作为稳定 label 训练 Dreamer reward head

优点：

- label 稳定
- 更容易判断 AMP shaping 是否带来收益

### 方案 B：半冻结 / 低频更新 discriminator

做法：

- discriminator 不每轮更新，而是按较低频率更新
- 或在若干 window 内固定 reward label

优点：

- 比完全冻结更灵活
- 仍可能保留一定适应性

缺点：

- 实验变量更多
- 更难解释结果

### 方案 C：AMP 作为附加 shaping，而非唯一 reward label

做法：

- world model reward head 继续拟合稳定环境 reward
- AMP reward 只在 behavior loss 中作为附加 shaping 项

优点：

- 不破坏 reward head 监督稳定性
- 更接近“附加技巧”而不是主目标替换

这是我更推荐的第二阶段 AMP 实验路线。

## What Should Stay Out Of The Mainline MVP

以下内容依然不建议回流到 `wmp-dreamer-aligned-plan.md`：

1. episode-level PPO / Dreamer mixed controller
2. `p_dreamer` takeover schedule
3. PPO valid mask 混合更新逻辑
4. PPO → Dreamer distillation
5. AMP final reward 直接监督 world model reward head

这些内容都应该留在 takeover experiment 文档中，作为可选扩展。

## Suggested Config Split

建议明确区分两类配置。

### A. 主线 Dreamer 对齐配置

```python
use_imagination_learning = True
dreamer_control_mode = True
dreamer_use_image = False
wm_update_interval = 1
dreamer_reward_mode = "env"
use_cont_head = True
```

### B. 第二阶段 Hybrid Takeover 实验配置

```python
dreamer_takeover_experiment = False
dreamer_control_start_after = 10000
dreamer_takeover_iters = 5000
dreamer_control_schedule = "linear"
dreamer_distill_coef = 0.1
dreamer_reward_mode = "final_amp"
```

重点是：

- A 是主线必需项
- B 是扩展实验项
- 不要把 A、B 混成一个 MVP

## Implementation Suggestions For Phase 2

如果未来进入 hybrid takeover experiment，建议按下面顺序推进：

1. 保持主线 Dreamer-only mode 不变
2. 增加 experiment flag，而不是污染默认控制路径
3. 先做 episode-level controller mixing
4. 再补 PPO valid mask / partial update
5. 再加 distillation
6. 最后再做 AMP reward migration

也就是说，第二阶段内部也建议分层展开，而不是一次性把所有实验变量塞进去。

## Detailed Implementation Approach

### 1. Keep The Dreamer Mainline Untouched

新增实验开关：

```python
dreamer_takeover_experiment = False
```

只有当这个开关打开时，才允许进入 mixed-controller 逻辑。默认路径必须仍然是：

- PPO baseline
- 或 Dreamer-only mainline

不要让 takeover 实验逻辑成为 `WMPRunner.learn()` 的默认分支。

### 2. Track Controller Per Env And Per Episode

维护：

```python
controller_mode_per_env  # PPO or Dreamer
```

规则：

- env reset 时重新采样 controller
- episode 内 controller 不变
- reset 的 env 单独更新 controller，不影响其他 env

这能避免 step-level controller jitter，也方便统计 PPO-controlled 与 Dreamer-controlled episode 的表现。

### 3. Merge Actions But Preserve Provenance

rollout 中可以分别计算：

- `ppo_actions`
- `dreamer_actions`

然后按 `controller_mode_per_env` merge 成最终 `actions`。

同时每条 transition 必须记录：

- `controller_mask`
- `ppo_valid_mask`

否则后续无法安全地区分 PPO update、Dreamer replay、AMP discriminator 数据。

### 4. PPO Storage Must Filter Dreamer-Controlled Samples

PPO update 只能使用 PPO-controlled transitions。

两种可选实现：

1. PPO storage 只写入 PPO-controlled transition
2. storage 写入全量 transition，但 generator 根据 `ppo_valid_mask` 过滤

第一版实验建议选第 1 种，更简单，也更不容易在 advantage/return 里混入无效样本。

如果选第 2 种，则必须同步修改：

- `RolloutStorage.compute_returns()`
- `RolloutStorage.mini_batch_generator()`
- `AMPPPO.update()`

否则 PPO loss 会悄悄吃到 Dreamer-generated data。

### 5. World Model Replay Should Use All Real Transitions

无论 controller 来自 PPO 还是 Dreamer，真实环境 transition 都应进入 world model replay。

原因：

- Dreamer 后期必须学习自己策略诱导出的状态分布
- 只用 PPO 数据会导致 Dreamer 控制时 distribution shift

但日志必须拆开：

- PPO-controlled replay ratio
- Dreamer-controlled replay ratio

否则 world model 变差时很难判断是数据分布变了，还是模型本身不稳。

### 6. Distillation Should Be A Separate Loss Head

distillation 不要混进 world model loss。

建议放在 `ImagBehavior` 中，作为可选 behavior regularizer：

```text
L_behavior = L_imag_actor + lambda_distill * L_distill
```

第一版只在真实 posterior feature 上做：

```text
L_distill = ||mean_dreamer(feat) - mean_ppo(obs, history, wm_feat)||^2
```

注意事项：

- teacher action 必须 `detach`
- distillation 权重应随 `p_dreamer` 衰减
- 一旦进入 Dreamer-only 后期，默认关闭 distillation

### 7. AMP Reward Migration Should Be An Independent Experiment

不要同时打开：

- takeover schedule
- distillation
- AMP final reward migration

AMP 迁移应单独有开关，例如：

```python
dreamer_amp_reward_mode = "off"  # off, frozen, shaping
```

推荐先做：

- `off`：Dreamer reward head 仍学环境 reward
- `shaping`：AMP 只作为 behavior shaping，不作为 reward head label
- `frozen`：冻结 discriminator 后生成稳定 AMP label

不要第一版就让持续更新的 discriminator 直接成为 reward head 监督源。

### 8. Takeover Schedule Should Have Stop Conditions

`p_dreamer` 不能只按 iteration 盲目线性增长。至少应允许以下保护：

- Dreamer-controlled reward 低于阈值时暂停增长
- fall rate 过高时暂停增长
- world model reward/cont loss 爆炸时暂停增长
- imagined actor/value loss 非有限时回退到 PPO-only

这能避免 schedule 把训练推进到不可恢复的坏分布。

## Failure Criteria

出现以下情况时，应暂停 takeover 实验，回到 Dreamer mainline 或 PPO baseline 排查：

1. `p_dreamer > 0` 后真实环境平均 reward 断崖式下降且无法恢复
2. Dreamer-controlled episode length 明显低于 PPO-controlled episode
3. world model loss 在 Dreamer-controlled 数据占比上升后持续发散
4. PPO update 中有效样本过少，导致 advantage/value 估计不稳定
5. distillation loss 下降但真实 reward 不升，说明 teacher imitation 没有帮助任务目标

这些失败不说明 DreamerV3 主线错误，只说明 hybrid takeover 实验没有带来正收益。

## Validation Plan For The Experiment Phase

### 1. Controller Mixing Validation

关注：

- PPO-controlled episode reward
- Dreamer-controlled episode reward
- episode length
- fall / reset 频率

目标：

- 引入 mixing 后，性能不应显著劣于纯 PPO 或纯 Dreamer-only baseline

### 2. Distillation Validation

关注：

- distillation loss
- Dreamer actor takeover 初期的 reward 曲线
- 是否减少早期训练震荡

目标：

- 证明 distillation 确实帮助接管，而不是仅增加实现复杂度

### 3. AMP Migration Validation

关注：

- reward head 拟合稳定性
- imagined reward 统计量
- Dreamer-only real control 的实际回报

目标：

- 判断 AMP 是否值得迁移入 Dreamer 主线
- 若收益不明显，应保留环境 reward 作为主线默认方案

## Expected Outcome

当主线 Dreamer 对齐方案已经稳定后，本文档中的实验应帮助我们回答：

- 是否需要 PPO 渐进接管
- 是否值得加入 teacher distillation
- AMP reward 是否值得进入 Dreamer 主训练闭环

但在此之前，主线目标始终应保持清晰：

- 先把 Dreamer-only 主学习路径做对
- 再决定是否需要 hybrid takeover

## Current Implementation Boundary

当前代码已经提供 `wmp_training_mode="takeover"` 的第一版骨架：

- `p_dreamer = 0` 时，使用 PPO 控制真实环境并正常更新 PPO。
- dataset 达到 `dreamer_control_start_after` 后，`p_dreamer` 从 0 开始线性增长。
- mixed-control 阶段会按 `p_dreamer` 混合 PPO action 与 Dreamer action。
- 所有真实 transition 都进入 Dreamer per-step dataset。
- mixed-control 阶段暂不做 PPO partial update，避免 PPO storage 混入 Dreamer-controlled transition。

尚未实现的第二阶段细化：

- PPO valid mask / partial PPO update
- PPO -> Dreamer distillation
- AMP reward migration
- takeover schedule 的自动暂停 / 回退条件
