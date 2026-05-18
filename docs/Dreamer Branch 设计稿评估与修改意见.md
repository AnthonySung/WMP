# Dreamer Branch 设计稿评估与修改意见

> 评估对象：[与dreamerv3区别.md](./与dreamerv3区别.md)  
> 评估日期：2026-05-18  
> 评估方式：逐条对照代码验证，覆盖 `rsl_rl/`、`dreamer/`、`legged_gym/` 三个主要模块

---

## 1. 总体评价

**方案方向正确，但对代码层面的耦合深度和 world model 当前训练状态偏乐观。**

设计稿的核心判断——"分 runner、分模块、显式 mode switch"——是唯一可行的架构路线。但在以下三个关键维度上需要修正预期：

| 维度 | 设计稿判断 | 代码实际 | 偏差 |
|------|-----------|----------|------|
| WM reward head 状态 | "基础定义存在" | `loss_scale: 0.0`，完全不参与训练 | 高 |
| WM cont head 状态 | "仍未启用" | 整段代码被注释掉 | 高 |
| WMP Runner 可拆分性 | 未具体评估 | 477 行单文件，PPO rollout 与 WM 逻辑深度交织 | 中 |

---

## 2. 逐项代码验证

### 2.1 world model "已有可复用基础" — 部分正确

**相关文件**：`dreamer/models.py`、`dreamer/networks.py`、`dreamer/configs.yaml`

✅ 已有的：
- `WorldModel` 类（含 encoder、RSSM、decoder）
- `RSSM.obs_step()` / `RSSM.img_step()` / `RSSM.imagine_with_action()`
- `lambda_return()`（`dreamer/tools.py:717-743`）
- `MultiEncoder` / `MultiDecoder`

❌ 缺失的（设计稿承认但低估了工作量）：

**`reward_head` 虽然定义但 `loss_scale=0`**：

```yaml
# dreamer/configs.yaml:34-35
reward_head:
  {layers: 2, dist: 'symlog_disc', loss_scale: 0.0, outscale: 0.0}
```

这意味着 reward head 虽然参与前向计算（`models.py:125-133`），但 loss 被乘 0 后不影响梯度。实际上 reward prediction 从未被训练过。

**`cont_head` 整段被注释掉**：

```python
# dreamer/models.py:81-92
# self.heads["cont"] = networks.MLP(
#     feat_size,
#     (),
#     config.cont_head["layers"],
#     ...
# )
```

`_scales` 字典里也没有 `cont` 的 key。continuation prediction 完全不存在。

**修改意见**：

> 将设计稿"第二步：补齐 world model 行为学习前提"拆分为两步：
> - **第 1.5 步**（spike）：单独验证 `reward_head` 和 `cont_head` 在当前 chunk-step 数据上的预测能力
> - **第二步**：正式补齐并调参

### 2.2 "不应直接复用 ActorCriticWMP" — 完全正确

**相关文件**：`rsl_rl/modules/actor_critic_wmp.py`

当前 `ActorCriticWMP.evaluate()` 的实现：

```python
# actor_critic_wmp.py:205-210
def evaluate(self, critic_observations, wm_feature, **kwargs):
    wm_latent_vector = self.critic_wm_feature_encoder(wm_feature)
    concat_observations = torch.concat((critic_observations, wm_latent_vector), dim=-1)
    value = self.critic(concat_observations)
```

`critic_observations` 包含 privileged info（基座线速度、外扰等），这在 imagined latent rollout 中不存在。**必须新增只消费 latent feature 的 critic**。

**修改意见**：设计稿中"新增 `DreamerActorCritic`"的判断无需修改。

### 2.3 "分 runner" — 正确且必要

**相关文件**：`rsl_rl/runners/wmp_runner.py`

当前 `WMPRunner.learn()` 的主循环（`wmp_runner.py:228-375`）约 150 行代码，内部包含且**无函数分解**：

- PPO rollout（`alg.act` → `env.step` → `alg.process_env_step`）
- GAE 计算（`alg.compute_returns`）
- WM obs_step 与特征提取
- WM dataset buffer 管理（6 个 `if reset_env_ids` / `if interval` 分支）
- WM training（`train_world_model()`）
- Depth predictor training（`train_depth_predictor()`）
- AMP reward computation
- 日志与 checkpoint

**如果在这个文件里再塞 Dreamer behavior learning，代码将完全不可维护。**

**修改意见**：设计稿中 `DreamerRunner` 必须新建文件，不与 `WMPRunner` 共用主循环。**额外建议**：`WMPRunner.learn()` 内部也应拆分为多个私有方法（`_rollout()`、`_update_wm_buffer()`、`_train_wm()` 等），以便未来两个 runner 可以共享部分逻辑。

### 2.4 Mode Switch 与 `sys.argv` — 设计稿判断正确但改动面大

**相关文件**：`rsl_rl/runners/wmp_runner.py:164-175`、`legged_gym/scripts/train.py`、`legged_gym/utils/task_registry.py`

当前 WM config 加载路径：

```python
# wmp_runner.py:164-175
configs = yaml.safe_load(
    (pathlib.Path(sys.argv[0]).parent.parent.parent / "dreamer/configs.yaml").read_text()
)
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
# ...为每个 config key 动态注册 CLI 参数...
self.wm_config = parser.parse_args()  # 二次调用 parse_args！
```

问题：
1. **硬编码路径依赖**：假设调用方是 `legged_gym/scripts/train.py`
2. **`sys.argv` 二次解析**：`WMPRunner.__init__` 里再次 `parse_args()`，与入口 `train.py` 的 `get_args()` 可能冲突
3. **不可测试**：无法在单元测试中构造 `WMPRunner` 而不经过 CLI

**修改意见**：

> 第一步实施时，必须将 WM config 加载改为：
> 1. 在 `task_registry` 或 `train.py` 层面加载 YAML
> 2. 通过 `train_cfg` dict 传入 `WMPRunner.__init__`
> 3. 删除 `_build_world_model()` 中对 `sys.argv` 和 `argparse` 的依赖

### 2.5 "新增 DreamerReplay" — 方向正确，需补充具体语义

**相关文件**：`rsl_rl/runners/wmp_runner.py:384-477`

当前 WM dataset 的数据维度（chunk-step 语义）：

| 字段 | 维度 | 说明 |
|------|------|------|
| `prop` | `(num_envs, T, prop_dim)` | chunk-step 时刻的 proprioception |
| `action` | `(num_envs, T, num_actions * interval)` | **拼接的** chunk 内所有 action |
| `reward` | `(num_envs, T)` | chunk 内累计奖励 |
| `image` | `(camera_envs, T, H, W, 1)` | 深度图（仅相机环境） |

Dreamer replay 需要定义的是：**从这个已有数据构造 `(o_t, a_t, r_t, o_{t+1}, is_first)` 序列的采样方式**。

当前 `train_world_model()` 里对 batch 的采样已经是 chunk-step 级别的：

```python
# wmp_runner.py:455-477
batch_idx = np.random.choice(range(self.env.num_envs), self.wm_config.batch_size, ...)
batch_length = min(int(self.wm_dataset_size[batch_idx].min()), self.wm_config.batch_length)
```

**修改意见**：

> `DreamerReplay` 需要显式定义：
> 1. 采样策略（uniform / prioritized / episode-based）
> 2. `is_first` 标记的插入规则（episode 边界）
> 3. batch_length 与 imagined horizon 的关系

### 2.6 "保留视觉链路" — 回避了 DepthPredictor 的归属

**相关文件**：`rsl_rl/modules/depth_predictor.py`、`rsl_rl/runners/wmp_runner.py:310-319`

当前只在 `camera_num_envs=1024` 个环境中使用真实深度，其余 `4096-1024=3072` 个环境由 `DepthPredictor` 生成伪深度：

```python
# wmp_runner.py:310-319
if (self.env.cfg.depth.use_camera):
    forward_heightmap = self.env.get_forward_map().to(self._world_model.device)
    pred_depth_image = self.depth_predictor(forward_heightmap, wm_obs["prop"])
    wm_obs["image"] = pred_depth_image
```

`DepthPredictor` 有自己的 optimizer 和训练循环。设计稿说"保留视觉链路"但没讨论它在 Dreamer Branch 里的位置。

**修改意见**：

> 在设计稿中补充：`DepthPredictor` 在 Dreamer Branch 里继续作为 WM 输入预处理器，其训练循环放在 `DreamerRunner` 的 real-side 逻辑中，不与 behavior learning 耦合。

### 2.7 "AMP 放在 real rollout side" — 边界正确但 optimizer 归属不明

**相关文件**：`rsl_rl/algorithms/amp_ppo.py:172-260`、`rsl_rl/algorithms/amp_discriminator.py`

当前 AMP loss 和 PPO loss 共用同一个 optimizer：

```python
# amp_ppo.py:72-79
params = [
    {'params': self.actor_critic.parameters(), 'name': 'actor_critic'},
    {'params': self.discriminator.trunk.parameters(), ...},
    {'params': self.discriminator.amp_linear.parameters(), ...}]
self.optimizer = optim.Adam(params, lr=learning_rate)
```

在 Dreamer Branch 里：
- `AMPDiscriminator` 是独立模块，有自己的参数量
- Behavior actor / latent critic 有独立的 optimizer
- WM 也有独立的 optimizer

**修改意见**：

> 在设计稿中明确 AMP 的 optimizer 归属：
> - **推荐**：AMP discriminator 使用独立 optimizer，不与 behavior optimizer 合并
> - 更新时机：每个 Dreamer iteration 的 real rollout 之后、behavior update 之前

---

## 3. 实施顺序评估

### 设计稿五步 vs 代码实际需要的修订

| 步骤 | 设计稿 | 评估 | 需要补充 |
|------|--------|------|----------|
| 第一步 | 拆模式分发与参数链路 | ✅ 正确 | **增加**：WM config 加载去 `sys.argv` 化 |
| **新增 1.5** | — | 🔴 缺失 | **验证 reward/cont head 在现有数据上的训练可行性** |
| 第二步 | 补齐 reward/cont head | ⚠️ 被低估 | 从 `loss_scale=0` / 注释状态到可用，需要调参验证 |
| 第三步 | 做 hybrid baseline | ✅ 合理 | — |
| 第四步 | 关键消融 | ✅ 合理 | — |
| 第五步 | motion prior imagination 化 | 远期 | 暂不评估 |

### 第 1.5 步的必要性

如果 reward prediction 在当前数据上无法收敛，后续所有 behavior learning 都不可靠。这个 spike 需要验证：

1. 将 `reward_head.loss_scale` 从 `0.0` 改为 `1.0`
2. 取消注释 `cont_head`，训练 continuation prediction
3. 观察 reward loss 和 cont loss 的收敛曲线
4. 检查 predicted reward 与真实累计 reward 的相关性
5. 如果失败，分析原因（数据分布、chunk-step 噪声、WM 容量等）

---

## 4. 设计稿未覆盖的关键问题

### 4.1 时间语义对齐

当前关键参数：

| 参数 | 值 | 含义 |
|------|-----|------|
| `wm_update_interval` | 5 | 1 chunk-step = 5 env-steps |
| `batch_length` | 64 | WM 训练序列长度（chunk-steps） |
| `num_steps_per_env` | 未硬编码 | PPO rollout 步数（env-steps） |

如果 imagined horizon 设为 16 chunk-steps，对应 80 env-steps = 0.4s（dt=0.005, decimation=4）。需要验证这个 horizon 对腿足控制是否足够。

**修改意见**：

> 在设计稿第 5.1 节中补充各时间参数的数值关系表。

### 4.2 训练调度

DreamerV3 标准做法是：collect → train WM → imagine → update behavior。当前 WMP 是一个 iteration 内交替做 PPO rollout + WM training。Dreamer Branch 需要定义：

- 每个 iteration 做几次 behavior update？
- Behavior update 和 WM update 的频率比？
- 是否积累 replay 多个 rollout 后再更新？

**修改意见**：

> 在设计稿中增加"训练调度"一节，定义 collect / WM update / behavior update 的循环结构。

### 4.3 已有但未使用的 actor/critic 配置

`dreamer/configs.yaml` 中已有完整的 actor/critic 配置块（来自上游 dreamerv3-torch）：

```yaml
actor:
  {layers: 2, dist: 'normal', entropy: 3e-4, ...}
critic:
  {layers: 2, dist: 'symlog_disc', slow_target: True, ...}
```

这些从未被代码引用。新增 `DreamerActorCritic` 时可直接复用这些配置 key 和 schema。

**修改意见**：

> 在设计稿 6.1 节"推荐新增模块"中注明：`DreamerActorCritic` 的配置 schema 复用 `dreamer/configs.yaml` 中已有的 `actor`/`critic` 块。

### 4.4 WMPRunner 的 save/load 逻辑

当前 `WMPRunner` 的 `save()` 和 `load()` 方法在 `wmp_runner.py` 中（需要读取文件末尾确认），但 checkpoint 需保存：WM + DepthPredictor + ActorCriticWMP + AMPDiscriminator + PPO optimizer。Dreamer Branch 还要增加 latent actor/critic + slow target critic + behavior optimizer。

**修改意见**：

> 在设计稿第 9.4 节补充：Dreamer checkpoint schema 建议使用 `training_mode` 字段标记，允许独立扩展，不强求与 WMP 同构。

---

## 5. 修订后的最小实现清单

基于以上评估，对设计稿第 11 节的清单提出以下修订：

### 5.1 新增条目

0. **前置验证（第 1.5 步）**
   - 将 `reward_head.loss_scale` 改为 `1.0`，验证 reward prediction 收敛性
   - 取消注释 `cont_head`，实现 continuation prediction
   - 确认训练后 WM 的 reward/cont 预测误差在可接受范围
   - **如果失败，暂停后续步骤，先分析根因**

### 5.2 修改条目

1. **训练入口分发** — 额外要求：
   - 将 WM config YAML 加载逻辑从 `WMPRunner._build_world_model()` 移到 `task_registry` 层
   - WM config 通过 `train_cfg` dict 的 `wm_config` 字段传入
   - 删除 `sys.argv` 和 `argparse` 的二次依赖

5. **补齐 world model** — 拆分为：
   - 5a. reward_head 启用并验证（从 `loss_scale=0` 到可用）
   - 5b. cont_head 实现并验证（从注释到可用）

### 5.3 新增小节建议

在"需要新增什么"（第 6 节）中补充：

| 小节 | 内容 |
|------|------|
| 6.4 训练调度 | collect / WM update / behavior update 的循环结构定义 |
| 6.5 AMP optimizer 归属 | 独立 optimizer 还是合并 |
| 6.6 时间参数关系表 | `interval`、`batch_length`、`horizon` 的数值关系 |

---

## 6. 一句话结论

**设计稿对边界划分（什么属于 Dreamer Branch、什么不属于 imagined phase）的判断全部正确，但对代码层面 WM 的训练缺口（reward 未训、cont 缺失）和工程耦合度（`sys.argv` 依赖、150 行无分解主循环）偏乐观。建议在启动第一步架构拆分前，先做一个 spike 验证 reward/cont head 在当前数据上的可训练性。如果这一步失败，后续架构工作没有意义。**

---

## 附录：关键代码位置速查

| 内容 | 文件 | 行号 |
|------|------|------|
| WM config 加载（sys.argv 依赖） | `rsl_rl/runners/wmp_runner.py` | 164-175 |
| WM train 主循环 | `rsl_rl/runners/wmp_runner.py` | 228-375 |
| WM dataset 初始化 | `rsl_rl/runners/wmp_runner.py` | 384-420 |
| WM batch 采样与训练 | `rsl_rl/runners/wmp_runner.py` | 455-477 |
| reward_head 定义（loss_scale=0） | `dreamer/configs.yaml` | 34-35 |
| cont_head 注释 | `dreamer/models.py` | 81-92 |
| RSSM imagine_with_action | `dreamer/networks.py` | 162-167 |
| lambda_return 工具函数 | `dreamer/tools.py` | 717-743 |
| ActorCriticWMP evaluate | `rsl_rl/modules/actor_critic_wmp.py` | 205-210 |
| AMPPPO update（PPO+AMP 联合 loss） | `rsl_rl/algorithms/amp_ppo.py` | 172-260 |
| AMPDiscriminator predict_amp_reward | `rsl_rl/algorithms/amp_discriminator.py` | 86-99 |
| train.py 入口 | `legged_gym/scripts/train.py` | 39-54 |
| task_registry make_wmp_runner | `legged_gym/utils/task_registry.py` | 163-215 |
| A1AMPCfg 配置 | `legged_gym/envs/a1/a1_amp_config.py` | 全文 |
| 已有但未用的 actor/critic 配置 | `dreamer/configs.yaml` | 28-32 |
