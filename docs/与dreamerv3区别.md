# WMP 与 DreamerV3 的区别，以及兼容两套训练范式的改造方案

## 1. 目标

本文档回答两个问题：

1. 当前仓库里的 WMP 和 DreamerV3 到底差在哪；
2. 如何在**不破坏原有 WMP 体系**的前提下，引入一套 DreamerV3-style 训练路径，并做到**一键切换 WMP / DreamerV3 两个版本**。

这里的重点不是简单把几个损失函数换掉，而是保证：

- 原本 WMP 路线继续可用；
- DreamerV3 路线独立存在；
- 公共模块尽量复用；
- 训练入口和配置入口统一；
- 切换时不需要改代码，只改配置或命令行参数。

---

## 2. DreamerV3 和 WMP 的核心区别

### 2.1 训练闭环不同

| 维度 | DreamerV3 | 当前 WMP |
| --- | --- | --- |
| world model 作用 | 核心训练中枢 | 主要做环境表征与预测，给控制器提供 feature |
| 策略优化位置 | 在 latent imagination 中优化 actor / critic | 在真实环境 rollout 上优化控制器 |
| real data 用途 | 主要用来训练 world model | 同时用于 world model 与 policy 学习 |
| imagined rollout | 核心组件 | 当前实现不是主策略优化路径 |
| value learning | critic 在 imagined feature 上学习 returns / value | PPO value loss + GAE 路径 |
| imitation / motion prior | 一般不是主路径 | 当前实现中 AMP 是重要组成部分 |

### 2.2 方法论不同

#### DreamerV3

标准范式是：

1. 用真实交互数据训练 world model；
2. 从 posterior latent state 出发做 imagination rollout；
3. 在 imagined trajectory 上训练 actor 和 critic；
4. 用 lambda return、slow target critic、normalization 等手段保证稳定；
5. 尽量用统一配置覆盖不同任务。

#### 当前 WMP

更接近下面这条路线：

1. 学一个能建模环境演化的 world model；
2. 从 world model 提取 feature；
3. 把 feature 作为策略输入的一部分；
4. 仍然在真实环境 rollout 上用 PPO / AMP 更新控制器；
5. 重点是视觉腿足控制、运动先验和 sim-to-real。

---

## 3. 当前仓库里的 WMP 实际训练方式

### 3.1 world model 部分是 Dreamer 风格

当前仓库已经具备 Dreamer 风格的模型学习骨架，关键位置包括：

- [dreamer/models.py](../dreamer/models.py)
- [dreamer/configs.yaml](../dreamer/configs.yaml)
- [dreamer/tools.py](../dreamer/tools.py)

其中：

- [dreamer/models.py:34-180](../dreamer/models.py#L34-L180) 定义了 `WorldModel`；
- 内部包含 encoder、RSSM、decoder、reward head；
- [dreamer/tools.py:717-743](../dreamer/tools.py#L717-L743) 已经有 `lambda_return()`；
- [dreamer/configs.yaml:30-57](../dreamer/configs.yaml#L30-L57) 里已经出现了 Dreamer 风格的部分超参。

这说明仓库底层的 **world model 组件已经具备复用价值**。

### 3.2 但行为学习仍然是 WMP / PPO / AMP 路线

当前真正的策略优化在这些文件里：

- [rsl_rl/runners/wmp_runner.py:237-375](../rsl_rl/runners/wmp_runner.py#L237-L375)
- [rsl_rl/algorithms/amp_ppo.py:169-304](../rsl_rl/algorithms/amp_ppo.py#L169-L304)
- [rsl_rl/modules/actor_critic_wmp.py:73-220](../rsl_rl/modules/actor_critic_wmp.py#L73-L220)

现有逻辑是：

1. 在真实环境里 rollout；
2. 定期用 world model 编码观测并生成 `wm_feature`；
3. policy 使用 `history + command + wm_feature` 输出动作；
4. 用 PPO surrogate loss、value loss、AMP discriminator loss 更新策略。

所以当前实现的本质是：

> world model 参与表征，但 policy optimization 并不发生在 imagination rollout 中。

这正是它和 DreamerV3 的根本区别。

---

## 4. 能不能改成 DreamerV3 训练范式

### 4.1 结论

**可以改，但不建议直接覆盖现有 WMP 体系。**

正确做法是：

- 保留当前 WMP 路线作为 `wmp` 模式；
- 新增一条 `dreamerv3` 模式；
- 两者共享 env、world model、数据采样与日志框架；
- 在 runner / behavior / config 层面分叉；
- 通过配置项或命令行参数一键切换。

### 4.2 为什么可行

因为当前仓库已经有以下基础：

1. 已有 world model 与 RSSM；
2. 已有 world model 训练数据采样逻辑；
3. 已有 `lambda_return()` 工具函数；
4. 已有单独的训练 runner 结构，便于扩展模式选择。

相关位置：

- [dreamer/models.py:117-180](../dreamer/models.py#L117-L180)
- [rsl_rl/runners/wmp_runner.py:384-477](../rsl_rl/runners/wmp_runner.py#L384-L477)
- [dreamer/tools.py:717-743](../dreamer/tools.py#L717-L743)
- [legged_gym/utils/task_registry.py:162-215](../legged_gym/utils/task_registry.py#L162-L215)

### 4.3 为什么不能只改几行

因为缺的不是 world model，而是 **DreamerV3 的 behavior learning 闭环**，主要包括：

- imagined actor update；
- imagined critic update；
- slow target critic；
- continuation / discount 预测；
- 从 posterior latent 出发的 imagination rollout 训练链路。

而当前代码使用的是：

- PPO；
- GAE / returns；
- AMP imitation discriminator；
- 真实环境 step 上的策略更新。

因此这不是一个“改损失函数”的问题，而是“新增第二套训练范式”的问题。

---

## 5. 改造原则：不影响原本体系，并支持一键切版本

### 5.1 总原则

必须遵守下面四条：

1. **保留原 WMP 默认行为不变**；
2. **DreamerV3 作为新增模式接入，不反向污染 WMP 逻辑**；
3. **公共组件只复用，不强行统一行为接口**；
4. **切换方式只依赖配置，不依赖手工改代码**。

### 5.2 推荐切换方式

新增一个统一配置字段，例如：

```yaml
training_mode: wmp
```

可选值：

- `wmp`
- `dreamerv3`

然后在训练入口根据这个字段分支。

如果希望命令行直接切换，也可以增加参数，例如：

```bash
python legged_gym/scripts/train.py --task a1 --training_mode wmp
python legged_gym/scripts/train.py --task a1 --training_mode dreamerv3
```

这样可以做到真正的一键切换。

---

## 6. 推荐的代码改造方案

## 6.1 第一层：保留原 WMP 路线不动

当前这些文件尽量只做最小侵入式改动：

- [rsl_rl/runners/wmp_runner.py](../rsl_rl/runners/wmp_runner.py)
- [rsl_rl/algorithms/amp_ppo.py](../rsl_rl/algorithms/amp_ppo.py)
- [rsl_rl/modules/actor_critic_wmp.py](../rsl_rl/modules/actor_critic_wmp.py)

原则是：

- 现有 WMP 训练逻辑继续保留；
- 不把 DreamerV3 的损失、分支、特殊逻辑硬塞进 `AMPPPO`；
- WMP 模式依然走原路径，保证老实验可复现。

## 6.2 第二层：新增 DreamerV3-style behavior 模块

建议新增一套独立行为学习模块，而不是改造 `ActorCriticWMP` 去兼容两套范式。

推荐新增文件：

- `rsl_rl/modules/dreamer_behavior.py`
- `rsl_rl/modules/dreamer_actor_critic.py`
- 或者放到 `dreamer/` 目录下也可以，但建议放在 `rsl_rl/modules/`，便于和现有 runner 集成。

这个模块负责：

1. 从 posterior latent 初始化 imagined rollout；
2. actor 根据 latent feature 生成 action；
3. RSSM 在 latent 空间里 `imagine_with_action`；
4. critic 在 imagined features 上估值；
5. 计算 lambda return；
6. 更新 actor / critic / slow critic。

## 6.3 第三层：给 world model 补齐 DreamerV3 所需头部

当前 [dreamer/models.py:81-92](../dreamer/models.py#L81-L92) 里的 `cont_head` 是注释状态。

如果要更接近 DreamerV3，应当：

- 恢复或重新实现 continuation head；
- 在 behavior learning 中使用 reward head + cont head + critic target；
- 继续复用 [dreamer/tools.py:717-743](../dreamer/tools.py#L717-L743) 的 `lambda_return()`。

## 6.4 第四层：在 runner 层做模式分发，而不是在算法内部混写

当前训练入口主要经过：

- [legged_gym/scripts/train.py:48-58](../legged_gym/scripts/train.py#L48-L58)
- [legged_gym/utils/task_registry.py:162-215](../legged_gym/utils/task_registry.py#L162-L215)

建议改为：

1. 在 `train.py` 读取 `training_mode`；
2. 在 `task_registry.py` 中统一创建对应 runner；
3. `training_mode == wmp` 时走现有 `WMPRunner`；
4. `training_mode == dreamerv3` 时走新增 `DreamerRunner`；
5. world model 数据采样、日志目录、保存恢复接口尽量保持一致。

即：

- `WMPRunner`：保持原样；
- `DreamerRunner`：新增，不污染原类。

这是“可维护”和“可回滚”成本最低的方案。

---

## 7. 一键切换的最小落地设计（修正版）

### 7.1 先说结论

“一键切换”不能只靠在文档里增加一条命令行参数。

原因是当前参数链路并不是单点入口，而是两段解析：

1. 训练入口通过 [legged_gym/utils/helpers.py:155-185](../legged_gym/utils/helpers.py#L155-L185) 解析固定参数；
2. world model 构建阶段又在 [rsl_rl/runners/wmp_runner.py:142-168](../rsl_rl/runners/wmp_runner.py#L142-L168) 里重新基于 `sys.argv` 做了一次 `argparse.parse_args()`。

因此，如果直接加：

```bash
python legged_gym/scripts/train.py --task a1 --training_mode dreamerv3
```

在现状下大概率会失败，而不是自动生效。

### 7.2 正确的一键切换实现方式

推荐分两步做。

#### 方案 A：先走配置切换，再补命令行切换

第一阶段先只支持配置切换：

```python
class runner:
    training_mode = 'wmp'
```

这样修改范围最可控，因为：

- `task_registry.get_cfgs()` 会拿到训练配置；
- `make_wmp_runner()` 内部可以根据 `train_cfg.runner.training_mode` 分发 runner；
- 不会立刻撞上 `helpers.py` 和 `WMPRunner._build_world_model()` 的双重参数解析问题。

第二阶段再补命令行切换，届时需要同时改三处：

1. 在 [legged_gym/utils/helpers.py:155-185](../legged_gym/utils/helpers.py#L155-L185) 中新增 `--training_mode`；
2. 在 [legged_gym/utils/helpers.py:130-153](../legged_gym/utils/helpers.py#L130-L153) 中把它写回 `cfg_train.runner.training_mode`；
3. 重构 [rsl_rl/runners/wmp_runner.py:142-168](../rsl_rl/runners/wmp_runner.py#L142-L168) 的 world model 配置构建方式，避免它再次对未知参数报错。

### 7.3 world model 参数链路必须一起改

这一点是评审意见里最关键的补充，应该明确写进方案。

当前 `_build_world_model()` 里直接读取 `dreamer/configs.yaml`，然后又从 `sys.argv` 构造 parser 并 `parse_args()`。这意味着：

- CLI 参数不是统一从训练入口一路传递下来的；
- runner 内部又复制了一套参数解释逻辑；
- 未来只要在训练入口新增参数，就有机会在 world model 构建阶段再次报错。

因此推荐把 world model 配置读取改成下面结构：

1. `helpers.py` 负责解析 CLI；
2. `update_cfg_from_args()` 负责把训练相关参数写回 train cfg；
3. `WMPRunner` / `DreamerRunner` 不再直接读 `sys.argv`；
4. world model 所需参数通过显式 config dict 传入 `_build_world_model()`。

也就是说，后续的正确方向不是“给 `_build_world_model()` 再补几个参数”，而是**去掉 runner 内部的二次命令行解析**。

### 7.4 推荐的一键切换落地顺序

#### 第一阶段

只支持配置文件切换：

- `training_mode = 'wmp'`
- `training_mode = 'dreamerv3'`

#### 第二阶段

再补命令行切换：

```bash
python legged_gym/scripts/train.py --task a1 --training_mode wmp
python legged_gym/scripts/train.py --task a1 --training_mode dreamerv3
```

这样文档和实现可以保持一致，不会出现“文档上能一键切换，代码里实际还不支持”的误导。

---

## 8. 具体修改建议（修正版）

### 8.1 可以复用的现有模块

下面这些模块可以复用，但需要重新定义它们在 DreamerV3 路线中的职责：

- [dreamer/models.py](../dreamer/models.py) 中 `WorldModel` 主体；
- [dreamer/networks.py](../dreamer/networks.py) 中 RSSM 与 `imagine_with_action`；
- [dreamer/tools.py:717-743](../dreamer/tools.py#L717-L743) 中 `lambda_return()`；
- [rsl_rl/runners/wmp_runner.py:384-477](../rsl_rl/runners/wmp_runner.py#L384-L477) 中可参考的数据采样框架。

注意，这里只能说“部分结构可参考或复用”，不能再表述成“当前 world model dataset / sampling 逻辑可直接复用”。

### 8.2 不能乐观假设直接复用的部分

#### 1. reward / continuation 头当前并没有准备好

评审意见是对的，这里不是“补个 cont_head”这么轻。

当前状态：

- `cont_head` 在 [dreamer/models.py:81-92](../dreamer/models.py#L81-L92) 整段被注释；
- `reward_head` 虽然定义了，但 [dreamer/configs.yaml:34-35](../dreamer/configs.yaml#L34-L35) 里 `loss_scale: 0.0`；
- 在 [dreamer/models.py:110-115](../dreamer/models.py#L110-L115) 中，reward loss 也确实由这个 scale 控制。

这意味着：

- 当前 reward head 默认并没有被真正训练成 Dreamer behavior 可依赖的 reward predictor；
- continuation head 也没有进入训练闭环；
- 所以 DreamerV3 行为学习最依赖的两个头都需要重新设计、启用并验证。

因此方案应修正为：

1. 补齐 `cont_head`；
2. 重新定义 `reward_head` 的训练目标和 loss scale；
3. 验证 reward / cont 预测质量之后，才能接 Dreamer actor / critic。

#### 2. 现有 WMP dataset 不是 Dreamer 常规 replay

当前 [rsl_rl/runners/wmp_runner.py:228-315](../rsl_rl/runners/wmp_runner.py#L228-L315) 里的 world model 数据流有两个重要特征：

- action 不是逐 env-step 存，而是 `wm_update_interval` 窗口内展平后的动作块；
- reward 也是窗口内累加后的聚合 reward，而不是逐步 reward。

此外视觉部分还有特殊语义：

- 只有 camera 子集 env 使用真实深度；
- 其它 env 的 `image` 由 depth predictor 生成；
- 见 [rsl_rl/runners/wmp_runner.py:293-302](../rsl_rl/runners/wmp_runner.py#L293-L302)。

所以这套数据更像：

> WMP 专用的 world-model 训练集，而不是 DreamerV3 行为学习可直接复用的 replay buffer。

因此文档方案应改成：

- world model 训练数据流可以局部参考；
- DreamerV3 behavior learning 需要重新定义 replay 语义；
- 至少要明确是继续使用 chunked transition，还是新增逐 model-step replay；
- 不能默认拿当前 dataset 直接接 imagined actor / critic。

#### 3. actor / critic 的可观测性假设需要彻底分叉

这不是只新建一个 `DreamerActorCritic` 就完事。

当前 [rsl_rl/modules/actor_critic_wmp.py:73-74](../rsl_rl/modules/actor_critic_wmp.py#L73-L74) 中：

- actor 输入是 `history latent + command + wm_latent`；
- critic 输入是 `num_critic_obs + wm_latent`。

而 [rsl_rl/modules/actor_critic_wmp.py:214-220](../rsl_rl/modules/actor_critic_wmp.py#L214-L220) 明确表明 critic 直接吃的是环境侧 `critic_observations`。在当前任务里这条路径与 privileged observation 强绑定。

DreamerV3 路线则应满足：

- actor 在 posterior / prior latent feature 上工作；
- critic 也主要基于 latent feature 估值；
- imagined rollout 期间不能继续依赖真实环境里的 privileged critic 路径。

因此这里不是“重写一个类名”的问题，而是**训练可观测性假设完全不同，必须独立分叉**。

### 8.3 建议新增的模块

基于上面的修正，建议新增：

1. `DreamerRunner`
   - 负责 real rollout、world model 更新、behavior update、checkpoint 管理；
2. `DreamerBehavior`
   - 负责 imagined rollout、actor / critic / slow target critic 更新；
3. `DreamerReplay`
   - 单独定义 DreamerV3 需要的数据语义，而不是直接复用 WMP dataset；
4. `DreamerActorCritic` 或等价 latent behavior 模块
   - 明确只服务于 latent imagination 路径；
5. `cont_head`
   - continuation / discount 建模头；
6. `DreamerCheckpointIO`
   - 如果不想污染原 `save/load` 逻辑，可单独管理 Dreamer 模式附加状态。

### 8.4 不建议直接修改的点

不建议做以下事情：

1. 在 [rsl_rl/algorithms/amp_ppo.py](../rsl_rl/algorithms/amp_ppo.py) 里硬塞 Dreamer 分支；
2. 在 [rsl_rl/modules/actor_critic_wmp.py](../rsl_rl/modules/actor_critic_wmp.py) 里同时兼容 PPO actor 和 latent imagination actor；
3. 让一个 runner 同时承担 WMP 与 DreamerV3 的所有内部细节；
4. 继续保留 runner 内部对 `sys.argv` 的二次解析；
5. 直接把当前 WMP dataset 当成 Dreamer replay 使用；
6. 默认假设未来可以轻松把 AMP reward 接到 imagined rollout 里。

---

## 9. 风险分析（修正版）

### 9.1 算法风险

腿足控制比很多标准 Dreamer benchmark 更敏感，主要体现在：

- imagination horizon 更难选；
- action repeat 与控制频率耦合更强；
- reward 尺度更容易导致 value 不稳定；
- latent rollout 误差积累更快。

### 9.2 world model 目标风险

这是这次评审补充后必须单列的一项。

DreamerV3 行为学习依赖：

- reward predictor；
- continuation predictor；
- 可用于 imagination 的 latent dynamics。

而当前实现中：

- `reward_head` 默认未有效训练；
- `cont_head` 没启用；
- dataset 语义是 WMP 专用 chunk 形式。

因此不能默认当前 world model 已满足 Dreamer behavior 的前提条件，必须先完成 world model 训练目标重构。

### 9.3 系统风险

当前 WMP 可能部分依赖 AMP 运动先验来保证 gait quality。

如果直接切成纯 DreamerV3：

- reward 可能上升更慢；
- gait 可能不自然；
- 早期训练稳定性可能下降。

### 9.4 视觉风险

当前视觉链路里还有 depth predictor，见：

- [rsl_rl/runners/wmp_runner.py:419-438](../rsl_rl/runners/wmp_runner.py#L419-L438)
- [rsl_rl/runners/wmp_runner.py:440-477](../rsl_rl/runners/wmp_runner.py#L440-L477)

如果 DreamerV3 模式完全依赖 imagined latent，视觉预测误差会更直接传递到策略学习里。

### 9.5 checkpoint 风险

文档之前写“保存/恢复接口尽量一致”，这句话需要收紧。

当前 WMP checkpoint 只保存：

- actor_critic；
- actor optimizer；
- world model；
- world model optimizer；
- depth predictor；

见 [rsl_rl/runners/wmp_runner.py:558-569](../rsl_rl/runners/wmp_runner.py#L558-L569)。

如果新增 DreamerV3 路线，通常还需要保存：

- Dreamer replay 的必要状态；
- behavior actor / critic；
- slow target critic；
- 它们各自的 optimizer；
- 可能还包括行为训练相关计数器。

因此更准确的说法应该是：

- `save()` / `load()` 接口名可以尽量保持一致；
- 但 checkpoint schema 需要为 Dreamer 模式单独扩展，不能假设和 WMP 完全共用。

### 9.6 AMP 兼容风险

文档之前提到“后续可把 AMP discriminator 当 auxiliary reward 接到 imagined return”，这句话也需要收紧。

当前 AMP reward 依赖真实环境里的：

- `amp_obs`
- `next_amp_obs`

计算位置见 [rsl_rl/runners/wmp_runner.py:317-323](../rsl_rl/runners/wmp_runner.py#L317-L323)。

而 imagined latent rollout 中并没有天然对应的 `amp_obs -> next_amp_obs`。除非后续再定义：

- 如何从 latent feature 解码出 AMP 判别器需要的状态；
- 或如何构造一个可在 imagination 中计算的 motion prior reward；

否则 AMP 不能被视为一个低成本的后续增强项。

因此推荐修正为：

- 第一版 DreamerV3 路线默认不接 AMP；
- 是否重新引入 motion prior，作为后续独立课题评估。

---

## 10. 推荐实施顺序（修正版）

为了不影响原体系，建议按下面顺序推进：

### 第一步：拆参数链路和模式分发

目标：

- 增加 `training_mode` 配置字段；
- 先支持配置切换；
- 去掉或绕开 runner 内部对 `sys.argv` 的二次解析；
- 建立 `WMPRunner` / `DreamerRunner` 的模式分发框架；
- 不改原 WMP 行为。

### 第二步：单独重构 Dreamer world model 训练目标

目标：

- 启用并验证 `reward_head`；
- 实现并验证 `cont_head`；
- 明确 Dreamer replay 的数据语义；
- 验证 world model 预测质量达到行为学习可用水平。

### 第三步：先做纯 proprioception 的 DreamerV3 baseline

目标：

- 先不接 camera / depth predictor；
- 先验证 imagined actor / critic 闭环；
- 去掉 privileged critic 假设；
- 确保 loss 与 rollout 数值稳定。

### 第四步：再接回视觉

目标：

- 重新定义视觉 world model 数据流是否沿用当前 WMP 方案；
- 验证视觉输入下的训练稳定性；
- 比较与原 WMP 的收益和代价。

### 第五步：最后再评估 motion prior / AMP

推荐先完全不混。

只有在 DreamerV3 baseline 已稳定后，再单独评估：

- 是否需要 motion prior；
- 如果需要，应采用 latent-compatible 的新设计，还是为其单独解码出判别器输入。

---

## 11. 最终建议（修正版）

如果目标是：

- **保留原 WMP**；
- **同时新增 DreamerV3 版本**；
- **最终支持一键切换**；
- **不影响已有实验和训练流程**；

那么最合适的方案是：

1. 保留原 `WMPRunner + AMPPPO + ActorCriticWMP`；
2. 新增 `DreamerRunner + DreamerBehavior + DreamerReplay (+ DreamerActorCritic)`；
3. 复用 world model 主体、RSSM 和 `lambda_return()`，但**不默认复用**当前 WMP 的 reward/cont 头配置、dataset 语义和 privileged critic 假设；
4. 在配置层先支持 `training_mode`，命令行切换放到参数链路重构之后；
5. 单独扩展 Dreamer 模式的 checkpoint schema；
6. 第一版 DreamerV3 路线默认不接 AMP。

这是当前信息下最稳妥、最不容易误导实现成本的方案。

---

## 12. 已确认的设计选择

根据当前讨论，方案已经收敛为下面 5 点：

1. `training_mode` **第一版就要支持 CLI 切换**；
2. Dreamer replay **第一版沿用当前 `wm_update_interval` 的 chunk 语义**，不立即改成逐 env-step / 逐 model-step replay；
3. 第一版 DreamerV3 **保留视觉链路**；
4. 第一版 DreamerV3 **保留 AMP / motion prior**；
5. checkpoint **允许为 Dreamer 模式扩展 schema**，不要求和 WMP 完全同构。

### 12.1 对第 2 点的具体解释

这里的“沿用 chunk 语义”指的是：

- 一个 world model step 继续对应 `wm_update_interval` 个 env step；
- action 继续采用当前实现中的动作块拼接形式；
- reward 继续采用当前实现中的窗口累计形式；
- DreamerV3 第一版的 imagined rollout、horizon、discount、lambda 也都基于这个 chunk-level time scale 定义。

相关代码位置见：

- [rsl_rl/runners/wmp_runner.py:228-315](../rsl_rl/runners/wmp_runner.py#L228-L315)

这样做的原因是：

1. 与当前 WMP 的视觉 world model 数据流兼容性最好；
2. 不必在第一版同时重写 replay 语义、视觉链路和 motion prior；
3. 更符合“不破坏原体系、可一键切换”的目标。

但这也意味着：

- 这条 DreamerV3 路线在第一版不是标准逐步 replay 版本；
- 后续分析 horizon、discount、return 时，都必须按 chunk-step 而不是 env-step 理解；
- AMP / motion prior 若要保留，也必须按这个 chunk 时间尺度重新设计兼容方式。

### 12.2 由此带来的实现约束

既然第 2 点已经选定为 chunk 语义，那么后续实现里应明确遵守：

1. 不新增第二套逐步 replay 作为第一版基础；
2. `DreamerRunner` 的 real-data 收集逻辑优先复用当前 world model 的 chunk 采样节奏；
3. actor / critic 的 imagined rollout 时间尺度与当前 world model step 对齐；
4. 文档和代码中都要明确区分：
   - env-step
   - chunk-step / world-model-step

避免后面在 reward、cont、discount、lambda 上出现语义混乱。

---

## 13. 一句话总结

**当前 WMP 是“world model 辅助 PPO/AMP 控制器”，而 DreamerV3 是“在 world model 的 imagination 中直接训练 actor/critic”。**

因此，当前确定的改法是：

**保留原 WMP 为 `wmp` 模式，新增 DreamerV3 为 `dreamerv3` 模式；第一版支持 CLI 切换，沿用当前 chunk-level world model 时间尺度，并允许为视觉、AMP / motion prior 与 checkpoint 做独立扩展。**