# WMP 对齐官方 DreamerV3 的主线方案

## Goal

在当前 WMP 项目中，优先实现一条尽量贴近官方 DreamerV3 与 humanoid-bench 的训练主线，而不是把 PPO 渐进接管、AMP 奖励对齐、蒸馏和混合控制同时塞进第一版。

这份方案的核心目标是：

- 让训练语义尽量接近 `danijar/dreamerv3`
- 保留当前 WMP 可复用的 world model 基础设施
- 将第一版范围压缩到足够小，方便定位问题
- 把“Dreamer 对齐”与“PPO 渐进接管”明确拆开

## References

- `danijar/dreamerv3`
  - 标准语义：真实经验训练 world model，imagined rollout 训练 actor/value，行为策略本身就是 Dreamer actor。
- `carlosferrazza/humanoid-bench`
  - 机器人任务参考。
  - 其 DreamerV3 训练入口本质上也是直接调用 Dreamer 训练栈，而不是 PPO/Dreamer controller 混合。

## Review Conclusion

对 `docs/wmp-dreamer-takeover-plan.md` 的评审结论如下：

- `per-step action`
- `continuation head`
- `prop-only first`

这些方向和官方 DreamerV3 是对齐的。

但以下内容不属于官方 DreamerV3 主线，应从第一版主方案中移出：

- episode-level PPO / Dreamer controller switching
- `p_dreamer` takeover schedule
- PPO valid mask 混合更新
- PPO -> Dreamer distillation
- world model 直接拟合持续变化的 AMP final reward

这些内容如果要做，更适合作为第二阶段的“hybrid takeover experiment”，而不是 Dreamer 对齐 MVP。

## What To Align

第一版主线应只对齐以下几点：

1. world model 使用 per-step action 建模
2. replay / dataset 使用 per-step transition
3. reward head 拟合稳定 reward target
4. continuation / terminal 建模恢复
5. imagined actor/value 成为主学习策略
6. 真实环境行为策略最终直接来自 Dreamer actor

## What Not To Align Yet

以下内容不进入第一版：

1. PPO 与 Dreamer 在同一个 run 中混合控制真实环境
2. PPO 只更新部分 rollout、Dreamer 更新另一部分 rollout
3. 基于 PPO teacher 的蒸馏损失
4. 使用不断变化的 AMP discriminator 输出作为 world model reward label
5. 视觉主导控制

## Current Gaps To Fix Before Implementation

当前代码已经有 RSSM、reward head、`ImagBehavior` 和旁路 imagined training，但还没有真正完成 DreamerV3 主线。正式进入实现前，需要先承认下面几个缺口：

1. `ImagBehavior` 现在仍是旁路训练模块，真实环境动作仍由 PPO actor 产生。
2. world model 仍与 `depth.update_interval` 绑定，并使用 action chunk。
3. dataset 还不是标准 per-step replay，而是 episode buffer + chunk action 的混合形式。
4. `cont_head` 仍处于注释/未训练状态，imagined return 目前依赖固定 discount。
5. reward head 第一版虽然已经打开，但 reward label 的定义仍需要和 Dreamer 主线明确绑定。
6. `prop-only` 不是完整模式开关，camera/depth predictor 相关逻辑仍会影响 runner 复杂度。
7. checkpoint 中还没有清晰区分 PPO policy、world model、Dreamer actor/value 的主从关系。

这些问题不一定要一次性大改，但实现顺序应围绕它们展开。

## Recommended MVP

推荐把第一版压缩成下面这条主线：

1. 关闭 image，先做 `prop-only`
2. 将 world model 从 action chunk 改成 per-step action
3. dataset 按 step 存 `prop`, `action`, `reward`, `is_first`, `is_terminal`
4. 恢复 `cont_head`
5. 真实数据训练 world model
6. posterior latent 上 imagined rollout 训练 actor/value
7. 真实环境控制逐步切到 Dreamer actor

这里的“逐步切到 Dreamer actor”不是指 PPO/Dreamer 混合控制，而是开发过程上的切换顺序：

- 初期保留 PPO baseline 作为对照
- 当 Dreamer actor 在线推理链路打通后，单独运行 Dreamer control 模式
- 不在同一个训练 run 中做混合 controller 调度

## Reward Recommendation

### 不建议第一版直接使用 AMP final reward 训练 world model reward head

原因：

- AMP discriminator 会持续更新
- 同一条历史 transition 的 reward label 会变得非平稳
- Dreamer 的 world model/replay 训练更适合稳定监督目标

### 推荐顺序

第一版：

- world model reward head 使用稳定的环境 reward
- AMP 仍可保留在 PPO baseline 中做对照

第二版可选：

- 若确认需要将 AMP 迁移进 Dreamer 主线，再设计冻结或半冻结 discriminator 的版本
- 或将 AMP 作为附加 shaping，而不是 world model reward 的唯一标签

## Continuation Recommendation

`cont_head` 建议恢复，并作为第一版核心内容之一。

原因：

- 这是 Dreamer 主线的一部分
- 对 locomotion 中的跌倒 / reset / timeout 很重要
- 比 PPO takeover 或蒸馏更属于“算法对齐项”

第一版建议：

- dataset 增加 `is_terminal`
- world model 增加 `cont_head`
- imagined target 使用 learned continuation

## Prop-Only Recommendation

第一版建议明确采用 `prop-only`。

原因：

- 当前目标是先打通 Dreamer 主训练闭环
- `depth.update_interval` 和 camera pipeline 当前耦合较深
- 视觉路径会把排查难度放大

建议新增：

- `dreamer_use_image = False`

当关闭时：

- 不构建 image buffer
- 不训练 `DepthPredictor`
- `MultiEncoder` 只走 vector 输入

## Runner Recommendation

不建议第一版继续在现有 `WMPRunner` 里叠加越来越多 mixed-control 逻辑。

更稳的做法有两个：

1. 在 `WMPRunner` 内保留最小 Dreamer mode
2. 新建更独立的 `DreamerRunner`

如果只考虑最小改动，可以先选第一个：

- 保留 `WMPRunner`
- 增加 `dreamer_control_mode`
- 当该模式打开时：
  - 不做 PPO update
  - 不走 mixed controller
  - 真实动作直接来自 Dreamer actor

这样虽然仍在同一个 runner 文件里，但语义上已经更接近官方 Dreamer 主线。

## Suggested Config Split

建议把配置分成两类。

### A. Dreamer 对齐主线配置

```python
use_imagination_learning = True
dreamer_control_mode = True
dreamer_use_image = False
wm_update_interval = 1
dreamer_reward_mode = "env"
use_cont_head = True
```

### B. 后续 Hybrid Takeover 实验配置

```python
dreamer_takeover_experiment = False
dreamer_control_start_after = 10000
dreamer_takeover_iters = 5000
dreamer_distill_coef = 0.1
dreamer_reward_mode = "final_amp"
```

重点是不要把 A、B 两类配置混成一个 MVP。

## Implementation Order

推荐实现顺序如下：

1. `per-step action` world model
2. `prop-only` dataset / encoder path
3. `cont_head`
4. Dreamer actor 在线推理接口
5. Dreamer-only control mode
6. 真实环境下跑 Dreamer-only training

只有当这条主线稳定后，再考虑：

1. AMP reward 迁移
2. PPO takeover
3. controller mixing
4. distillation

## Detailed Implementation Plan

### 1. Add A Real Dreamer Mode

新增一个明确的主线开关：

```python
dreamer_control_mode = False
```

语义：

- `False`：保留当前 WMP/PPO 主路径，Dreamer Branch 可作为旁路训练。
- `True`：进入 Dreamer 主线，真实环境动作由 Dreamer actor 产生，PPO update 关闭。

第一版不要在 `dreamer_control_mode=True` 时混入 PPO controller，也不要加入 takeover schedule。

### 2. Convert World Model To Per-Step Actions

当前代码中 `wm_config.num_actions = env.num_actions * depth.update_interval`，这应改成：

```python
wm_config.num_actions = env.num_actions
wm_update_interval = 1
```

需要同步修改：

- `wm_action_history` 删除或只保留单步 action
- `wm_reward` 不再跨 interval 累积
- `wm_buffer["action"]` shape 改成 `[num_envs, time, env.num_actions]`
- rollout 中每个 env step 都写入 world model transition

这是最关键的一步，因为 Dreamer actor 只能输出单步 action，不能自然接管 action chunk。

### 3. Introduce Prop-Only Dataset Path

新增：

```python
dreamer_use_image = False
```

当关闭 image 时：

- `obs_shape = {"prop": (prop_dim,)}`
- 不初始化 `DepthPredictor`
- 不创建 `image` / `forward_height_map` buffer
- `sample_world_model_batch()` 不处理 image fallback

这样可以先让 Dreamer 主线的状态、动作、奖励、终止闭环稳定，再处理视觉。

### 4. Restore Continuation Head

在 `dreamer/models.py` 中恢复 `cont_head`：

- 构建 `self.heads["cont"]`
- `self._scales` 中加入 `cont`
- `preprocess()` 根据 `is_terminal` 构造 `cont = 1 - is_terminal`
- `_train()` 中计算 cont loss
- `ImagBehavior._compute_target()` 使用 learned continuation

dataset 需要增加：

- `is_terminal`

注意区分 timeout 和真实终止。locomotion 中 timeout 不一定代表失败终止，建议保留 `infos["time_outs"]` 语义，避免把正常时间截断当成跌倒。

### 5. Add Dreamer Online Policy State

Dreamer actor 控制真实环境时，需要维护在线 RSSM state：

- `dreamer_latent`
- `dreamer_prev_action`
- `dreamer_is_first`

每步流程：

1. encoder 编码当前 `prop`
2. `dynamics.obs_step()` 更新 posterior
3. `ImagBehavior.actor` 基于 `dynamics.get_feat(latent)` 产生 action
4. `env.step(action)`
5. reset env 的 latent/action 清零或通过 `is_first` 重置

这条在线路径应与 imagined rollout 共享 actor/value，但不共享 PPO policy。

### 6. Disable PPO Update In Dreamer Control Mode

当 `dreamer_control_mode=True`：

- 不调用 `self.alg.act()`
- 不写 PPO rollout storage
- 不调用 `self.alg.compute_returns()`
- 不调用 `self.alg.update()`

仍然可以保留 AMP/PPO baseline 的代码，但这条模式下不要让 PPO loss 参与训练。

### 7. Keep World Model Replay Shared And Simple

第一版可以继续使用现有 tensor dataset/buffer，不必立刻迁移到官方 Dreamer 的 episode replay 文件格式。

但接口语义要改成标准 per-step：

- sample batch 返回 `[batch, time, ...]`
- 每条 batch 内第一个 step 的 `is_first=True`
- `action[t]` 对应从 `obs[t]` 到 `obs[t+1]` 的动作

如果现有 buffer 难以保持这个语义，优先提取一个小的 `DreamerReplay` 类，而不是继续扩大 `WMPRunner`。

### 8. Checkpoint Semantics

Dreamer 主线 checkpoint 应至少保存：

- world model
- world model optimizer
- Dreamer actor/value
- Dreamer actor/value optimizers
- current iteration

PPO checkpoint 可以继续保存，但在 `dreamer_control_mode=True` 下应被视为 baseline/teacher 资产，而不是主策略。

## Validation Plan

### Phase 1: World Model Alignment

目标：

- world model loss 有限
- reward head 有限
- cont head 有限
- posterior / prior entropy 正常

关注：

- `World_model/model_loss`
- `World_model/reward_loss`
- `World_model/cont_loss`
- `World_model/prior_ent`
- `World_model/post_ent`

### Phase 2: Imagined Behavior Alignment

目标：

- imagined actor/value loss 有限
- imagined reward 统计正常
- lambda return 稳定

关注：

- `ImagBehavior/imag_actor_loss`
- `ImagBehavior/imag_value_loss`
- `ImagBehavior/imag_reward_mean`
- `ImagBehavior/imag_target_mean`

### Phase 3: Dreamer-Only Real Control

目标：

- Dreamer actor 可直接驱动真实环境
- 不依赖 PPO mixed-control
- 平均 reward 和 episode length 可持续增长

关注：

- real env reward
- episode length
- reset / fall rate
- world model imagination 指标是否同步稳定

## Relationship To The Existing Takeover Plan

`docs/wmp-dreamer-takeover-plan.md` 不需要删除，但建议重新定位。

建议关系如下：

- `wmp-dreamer-aligned-plan.md`
  - 主线方案
  - 目标是与官方 DreamerV3 / humanoid-bench 尽量对齐

- `wmp-dreamer-takeover-plan.md`
  - 扩展实验方案
  - 目标是最小化对现有 WMP/PPO 体系的破坏，实现平滑接管

两者都可以存在，但不要把它们当成同一件事。

## Expected Outcome

完成本方案后，我们会得到一条更清晰的 Dreamer 主线：

- world model 按 per-step 建模
- reward / continuation / imagined actor-value 闭环完整
- 真实控制策略直接来自 Dreamer actor
- 训练语义尽量靠近官方 DreamerV3

在这条主线稳定后，再单独决定是否值得做 PPO 渐进接管实验。那时如果要做，也会更容易判断收益来自 Dreamer 本身，还是来自混合训练技巧。

## Current Mode Commands

当前实现保留三种可对比模式：

### 1. 原始 WMP baseline

```bash
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0 --wmp_training_mode=wmp
```

语义：

- PPO/AMP 主训练
- world model 作为表征分支
- `wm_feature` 输入 `ActorCriticWMP`

### 2. DreamerV3 aligned mode

```bash
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0 --wmp_training_mode=align
```

语义：

- prop-only
- per-step action world model
- Dreamer actor 控制真实环境
- world model + imagined actor/value 训练
- 不做 PPO update

### 3. Hybrid takeover mode

```bash
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0 --wmp_training_mode=takeover
```

语义：

- 前期 `p_dreamer = 0` 时由 PPO 控制并训练 PPO
- world model dataset 达到 `dreamer_control_start_after` 后开始线性增加 `p_dreamer`
- 后期逐步使用 Dreamer actor 控制真实环境
- 当前版本在 mixed-control 阶段不做 PPO partial update，避免 PPO storage 吃到 Dreamer-controlled 数据

注意：

- `--use_imagination_learning` 仍然兼容；如果当前模式是 `wmp`，它会自动切到 `align`。
- 完整 PPO valid-mask partial update、distillation、AMP reward migration 属于 takeover 第二阶段细化项。
