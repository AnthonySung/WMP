# WMP 与 DreamerV3 的区别，以及本项目的 Dreamer Branch 设计稿

> 更新日期：2026-05-21  
> 注：本文档仍以“设计稿”为主，但其中部分条目已经在当前代码中实现；实现细节请同时参考 [Dreamer Branch 实现文档](./Dreamer%20Branch%20实现文档.md)。

## 1. 目标

本文档回答两个问题：

1. 当前仓库中的 WMP 与 DreamerV3 在训练范式上到底差在哪里；
2. 如何在**不破坏现有 WMP 主线**的前提下，为本项目新增一条可切换的 `Dreamer Branch`。

本文档不是论文综述，也不是 DreamerV3 复现说明。它的目的，是为本项目的第一版 `Dreamer Branch` 给出清晰、可实现、边界明确的设计。

---

## 2. 当前项目与 DreamerV3 的根本区别

### 2.1 训练闭环不同

当前项目的本质是：

- world model 参与表征；
- policy 在真实环境 rollout 上更新；
- 策略优化仍然是 `PPO + value loss + AMP` 路线。

关键代码位置：

- [rsl_rl/runners/wmp_runner.py:242-375](../rsl_rl/runners/wmp_runner.py#L242-L375)
- [rsl_rl/algorithms/amp_ppo.py:172-260](../rsl_rl/algorithms/amp_ppo.py#L172-L260)
- [rsl_rl/modules/actor_critic_wmp.py:169-220](../rsl_rl/modules/actor_critic_wmp.py#L169-L220)

DreamerV3 的本质则是：

- real data 主要用于训练 world model；
- actor / critic 在 latent imagination 中训练；
- reward、continuation、value 都围绕 imagined trajectory 定义。

所以两者的根本差异不是“是否使用 RSSM”，而是：

> 当前 WMP 是“world model 辅助控制器”，DreamerV3 是“world model 内部直接训练行为”。

### 2.2 world model 层已经有可复用基础

当前仓库已经有 Dreamer 风格的 world model 骨架：

- [dreamer/models.py](../dreamer/models.py)
- [dreamer/networks.py](../dreamer/networks.py)
- [dreamer/tools.py:717-743](../dreamer/tools.py#L717-L743)

其中已经具备：

- encoder；
- RSSM；
- decoder；
- `reward_head`；
- `cont_head`；
- `lambda_return()` 工具函数。

这意味着本项目适合新增 `Dreamer Branch`，但**不适合把现有 WMP 直接解释成 DreamerV3**。

---

## 3. 设计结论

### 3.1 总结论

本项目应保留现有 `WMP` 主线，并新增一条独立的 `Dreamer Branch`。

推荐形式：

- `training_mode: wmp`
- `training_mode: dreamerv3`

两条路线共享：

- env；
- world model 主体；
- world model 数据采样框架；
- 日志与 checkpoint 基础设施。

两条路线分叉在：

- behavior learning；
- actor / critic 结构；
- replay 语义；
- 训练闭环。

实现原则上应优先：

- 分 runner；
- 分 behavior 模块；
- 分 actor / critic 模块；
- 共享 world model 与公共工具。

只有当代码天然属于公共基础设施时，才建议放在同一文件中，并且必须由显式 `training_mode` 或等价开关控制。

### 3.2 第一版 Dreamer Branch 的已确认边界

第一版不是纯标准 DreamerV3，而是一个**受约束的 hybrid Dreamer Branch**。

已确认的边界如下：

1. 保留原 `WMP` 主线，不替换现有训练路线。
2. 第一版支持 `CLI / config` 级别切换。
3. 第一版 replay 沿用当前 `chunk-step / world-model-step` 语义。
4. 第一版保留视觉链路。
5. 第一版保留 `AMP / motion prior`，但只保留在 real rollout / auxiliary side。
6. 第一版保留 `privileged critic`，但只保留在 real rollout / bootstrap / auxiliary support side。
7. imagined phase 默认只走 latent-path，不直接使用 AMP reward，也不直接读取 privileged observation。
8. 第一版主 critic 是 `latent critic`。
9. 第一版默认采用**分 runner、分模块、显式 mode switch** 的方式保留原版可回切能力；只有确实值得共享的代码才允许写在一起，并且必须受显式开关保护。

---

## 4. 第一版 Dreamer Branch 的训练边界

### 4.1 imagined phase 包含什么

imagined phase 默认只包含：

- latent state；
- actor 产生的 action；
- RSSM latent rollout；
- reward prediction；
- continuation prediction；
- latent critic value prediction。

也就是说，imagined phase 的核心序列是：

```text
post_state_t -> action_t -> prior_state_{t+1} -> action_{t+1} -> ...
```

这里的 `state` 指 RSSM latent state，而不是环境原始观测。

相关接口：

- [dreamer/networks.py:201-233](../dreamer/networks.py#L201-L233)
- [dreamer/networks.py:235-260](../dreamer/networks.py#L235-L260)
- [dreamer/networks.py:178-186](../dreamer/networks.py#L178-L186)

### 4.2 imagined phase 不包含什么

第一版 imagined phase 默认不直接包含：

- 当前 AMP reward；
- 当前 privileged observation；
- 环境侧 `critic_observations`；
- 真实 `amp_obs -> next_amp_obs` transition。

原因不是这些信息无价值，而是它们都属于 **real-only signals**，当前并不天然存在于 latent rollout 里。

### 4.3 AMP 放在哪里

第一版保留 AMP，但放在 real rollout / auxiliary side：

- AMP 继续基于真实 `amp_obs` 与 `next_amp_obs` 计算；
- 不把当前 AMP 判别器输出直接当 imagined reward；
- AMP discriminator 使用独立 optimizer，在 real rollout 后单独更新；
- 如果以后要进入 imagination，需要单独设计 latent-compatible motion prior。

当前 AMP reward 的实现位置：

- [rsl_rl/algorithms/amp_discriminator.py:86-99](../rsl_rl/algorithms/amp_discriminator.py#L86-L99)
- [rsl_rl/runners/wmp_runner.py:317-324](../rsl_rl/runners/wmp_runner.py#L317-L324)

### 4.4 privileged critic 放在哪里

第一版主 critic 定义为 `latent critic`。

`privileged critic` 如果保留，只允许出现在：

- real rollout supervision；
- posterior 对齐点；
- bootstrap；
- auxiliary value support。

不能把它解释成 imagined future 上仍可直接消费真实 privileged observation 的主 critic。

---

## 5. replay 与时间语义

### 5.1 第一版为什么沿用 chunk-step

当前 world model 数据流已经按 `wm_update_interval` 组织：

- 一个 world-model-step 对应多个 env-step；
- action 使用拼接块；
- reward 使用窗口累计。

关键位置：

- [rsl_rl/runners/wmp_runner.py:228-315](../rsl_rl/runners/wmp_runner.py#L228-L315)
- [rsl_rl/runners/wmp_runner.py:384-477](../rsl_rl/runners/wmp_runner.py#L384-L477)

因此第一版 `Dreamer Branch` 推荐继续沿用 chunk-step 语义，而不是立即改写成逐 env-step replay。

### 5.2 这意味着什么

第一版应统一按下面语义理解：

- horizon 按 chunk-step 计；
- discount 按 chunk-step 定义；
- lambda return 按 chunk-step 定义；
- imagined rollout 的时间尺度与 world-model-step 对齐。
- 每个 chunk-step 只采样一次整段 chunk action，并在 chunk 内顺序执行各个 env-step slice。

这条路线不是标准逐步 Dreamer replay，而是本项目现有数据语义上的 Dreamer-style behavior branch。

---

## 6. 需要新增什么

### 6.1 推荐新增模块

建议新增：

1. `DreamerRunner`
   - 负责 real rollout、world model update、behavior update、checkpoint 管理。
2. `DreamerBehavior`
   - 负责 imagined rollout、actor / latent critic / slow target critic 更新。
3. `DreamerReplay`
   - 单独定义 `Dreamer Branch` 的 replay 语义。
4. `DreamerActorCritic`
   - 明确服务于 latent imagination 路径。
5. `cont_head`
   - continuation / discount 建模头。
6. `DreamerCheckpointIO`
   - 管理 Dreamer 分支额外训练状态。

### 6.2 可以复用什么

可以优先复用：

- `WorldModel` 主体
  [dreamer/models.py](../dreamer/models.py)
- RSSM 与 latent rollout
  [dreamer/networks.py](../dreamer/networks.py)
- `lambda_return()`
  [dreamer/tools.py:717-743](../dreamer/tools.py#L717-L743)

### 6.3 不应直接复用什么

不应乐观假设直接复用：

- 当前 WMP dataset 作为 Dreamer replay；
- 当前 `ActorCriticWMP` 作为 imagined actor / critic；
- 当前 `AMPPPO` 作为 Dreamer behavior learner；
- 当前 reward / continuation 头配置直接满足行为学习。

其中 world model 仍然存在需要持续验证的点：

- `reward_head` 虽已启用，但在当前 chunk-step 数据上的预测质量仍需验证。
- `cont_head` 虽已启用，但 replay terminal 语义仍然较粗。

---

## 7. 不建议的实现方式

不建议做下面这些事：

1. 在 [rsl_rl/algorithms/amp_ppo.py](../rsl_rl/algorithms/amp_ppo.py) 里硬塞 Dreamer 分支。
2. 在 [rsl_rl/modules/actor_critic_wmp.py](../rsl_rl/modules/actor_critic_wmp.py) 里同时兼容 PPO actor 和 imagined actor。
3. 让一个 runner 同时承担 WMP 与 DreamerV3 的所有内部细节。
4. 继续依赖 runner 内部对 `sys.argv` 的二次解析。
5. 把当前 AMP reward 不经改造直接接到 imagined return。
6. 把 support-only privileged critic 重新变成 imagined phase 主 critic。
7. 通过手工改代码、注释分支或改入口来切换 WMP 与 Dreamer Branch。

---

## 8. 实施顺序

### 第一步：拆模式分发与参数链路

目标：

- 增加 `training_mode`；
- 先支持配置切换，再补命令行切换；
- 让切回原版 WMP 只依赖 mode switch，而不是代码修改；
- 去掉或绕开 world model 构造过程中的 `sys.argv` 二次解析；
- 建立 `WMPRunner` / `DreamerRunner` 分发。

### 第二步：补齐 world model 行为学习前提

目标：

- 验证已启用的 `reward_head`；
- 验证已启用的 `cont_head`；
- 明确 Dreamer replay 的 chunk-step 语义；
- 证明 world model 已达到 imagined behavior 可用水平。

### 第三步：做第一版 hybrid Dreamer Branch baseline

目标：

- 保留视觉链路；
- 保留 AMP，但只放在 real rollout side；
- 保留 privileged critic，但只做 support role；
- imagined phase 只走 latent-path；
- 主 critic 使用 latent critic；
- 原版 WMP 路线保持可直接回切；
- 先验证训练闭环数值稳定。

### 第四步：做关键消融

建议至少比较：

- 有无 AMP；
- latent critic only vs latent critic + privileged support；
- 有无视觉链路；
- `Dreamer Branch` vs 原 `WMP`。

### 第五步：再考虑 motion prior 的 imagination 化

只有在第一版稳定之后，再评估是否需要：

- latent-space motion prior；
- latent 解码到 AMP 特征；
- 或其他 imagined imitation 设计。

---

## 9. 风险

### 9.1 算法风险

腿足控制上的 Dreamer-style behavior learning 比标准 benchmark 更敏感：

- horizon 难选；
- reward 尺度更敏感；
- latent rollout 更易积累误差；
- value 更容易不稳定。

### 9.2 world model 风险

当前 world model 还不是现成可用的 Dreamer behavior model：

- reward 训练虽已接入，但是否足够支撑 imagined behavior 仍待验证；
- continuation 已接入，但 terminal 标记仍然粗化；
- replay 语义不是标准逐步版本。

### 9.3 系统耦合风险

第一版保留视觉、AMP 和 privileged support，会降低“彻底换范式”的风险，但会提高系统复杂度：

- loss 来源更多；
- 梯度边界更难维护；
- 日志与调参维度明显增加。

### 9.4 checkpoint 风险

Dreamer 分支通常需要额外保存：

- replay 状态；
- behavior actor / latent critic；
- slow target critic；
- 对应 optimizer；
- 训练计数器。

因此 checkpoint schema 应允许独立扩展，而不应强求与 WMP 完全同构。

---

## 10. 一句话结论

**本项目最合适的方向，不是把当前 WMP 改写成 DreamerV3，而是保留 WMP 主线，新增一条以 chunk-step replay 为基础、以 latent critic 为主、且把 AMP / privileged critic 限制在 imagined phase 之外的 hybrid Dreamer Branch。**

---

## 11. 第一版最小实现清单

### 11.1 总体原则

第一版实现必须同时满足两个目标：

1. `Dreamer Branch` 能独立演进；
2. 原版 `WMP` 能通过显式 mode switch 立即切回。

因此默认策略是：

- 不同训练范式尽量分文件、分 runner、分模块；
- 共享基础设施才放在一起；
- 如果必须写在同一文件，必须有显式开关。

不接受的切换方式：

- 手工注释代码；
- 手工改 import；
- 手工改训练入口；
- 手工删分支。

### 11.2 文件级最小改动

1. 训练入口分发
   - [legged_gym/scripts/train.py](../legged_gym/scripts/train.py)
   - [legged_gym/utils/task_registry.py](../legged_gym/utils/task_registry.py)
   - 增加 `training_mode`
   - 统一按 mode switch 分发到 `WMPRunner` 或 `DreamerRunner`
   - 不再把训练入口硬编码到 `make_wmp_runner()`

2. 配置层
   - [legged_gym/envs/a1/a1_amp_config.py](../legged_gym/envs/a1/a1_amp_config.py)
   - [legged_gym/envs/base/legged_robot_config.py](../legged_gym/envs/base/legged_robot_config.py)
   - 新增 `training_mode`
   - 新增 Dreamer 分支专用配置块
   - 明确 `use_amp_aux`、`use_privileged_bootstrap`、`use_camera`

3. 新增 `DreamerRunner`
   - `rsl_rl/runners/dreamer_runner.py`
   - 负责 real rollout、world model update、behavior update、checkpoint
   - 当前实现已修正 chunk-step 动作执行语义，并接入 AMP auxiliary update
   - 不修改 `WMPRunner` 的原始训练闭环

4. 新增 `DreamerReplay`
   - `rsl_rl/storage/dreamer_replay.py`
   - 独立承载 chunk-step replay 语义
   - 当前实现已补充采样窗口命中 episode 尾部时的 `is_terminal`
   - 不把现有 `wm_dataset` 直接改造成两用结构

5. 补齐 world model 行为学习目标
   - [dreamer/models.py](../dreamer/models.py)
   - [dreamer/configs.yaml](../dreamer/configs.yaml)
   - 启用 `reward_head`
   - 实现 `cont_head`
   - 这部分如果写在现有文件里，必须用显式配置控制，不得影响原 WMP 默认行为

6. 新增 `DreamerBehavior`
   - `rsl_rl/algorithms/dreamer_behavior.py`
   - imagined rollout
   - actor / latent critic / slow critic update
   - 当前实现已使用 final imagined state 的 slow critic 做 λ-return bootstrap
   - 不混入 `AMPPPO`

7. 新增 `DreamerActorCritic`
   - `rsl_rl/modules/dreamer_actor_critic.py`
   - actor 与主 critic 都吃 latent feature
   - 不改造现有 `ActorCriticWMP` 去兼容两套主路径

8. 保留 privileged support
   - 优先放在 `DreamerRunner` real-side 逻辑
   - 明确只做 bootstrap / auxiliary support
   - 如果和 latent critic 同文件存在，必须显式区分主路径与 support 路径

9. 保留 AMP auxiliary
   - 优先复用当前 `AMPDiscriminator`
   - 保持 real transition 训练链路
   - 当前实现为独立 optimizer + 独立 update
   - imagined phase 不消费 AMP reward

10. checkpoint
   - Dreamer 分支允许单独 schema
   - 不强求与 `WMPRunner` 完全同构

### 11.3 哪些地方可以共用，哪些地方不要混

可以共用：

- world model 主体；
- RSSM；
- `lambda_return()`；
- 公共日志工具；
- checkpoint 辅助工具；
- mode switch 分发入口。

不要混在一起：

- `WMPRunner` 与 `DreamerRunner` 主循环；
- `AMPPPO` 与 `DreamerBehavior`；
- `ActorCriticWMP` 与 `DreamerActorCritic`；
- WMP rollout storage 与 Dreamer replay 语义。

### 11.4 如果必须写在一起，必须有的开关

如果某些代码必须留在同一文件，至少要有显式开关：

- `training_mode == "wmp"`
- `training_mode == "dreamerv3"`

必要时再细分：

- `use_amp_aux`
- `use_privileged_bootstrap`
- `use_camera`

原则是：

- 开关决定行为；
- 不是注释决定行为；
- 不是改代码决定行为。
