# WMP 项目 Codespace 环境搭建指南

> 本文档记录了在 GitHub Codespace 中搭建 WMP 项目开发环境的完整步骤。
> 注意：Codespace 通常没有 NVIDIA GPU，因此 **Isaac Gym** 无法在此环境中运行。但可以通过 SSH 隧道等方式连接到外部 GPU 服务器来执行训练。

---

## 1. 创建 Conda 环境

```bash
# 配置 conda 频道（首次需要）
conda config --add channels defaults
conda config --add channels conda-forge

# 创建 Python 3.8 环境
conda create -n wmp python=3.8 -y

# 激活环境
conda init bash
source ~/.bashrc
conda activate wmp
```

## 2. 安装 PyTorch（CUDA 11.7 版本）

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

## 3. 安装系统依赖

```bash
sudo apt-get update
sudo apt-get install -y build-essential ninja-build libgl1 libglib2.0-0
```

## 4. 安装 Python 依赖包

```bash
pip install setuptools==59.5.0
pip install ruamel_yaml==0.17.4
pip install opencv-contrib-python
pip install -r requirements.txt
```

## 7. 安装 Isaac Gym（需要 GPU）

> ⚠️ 此步骤需要 NVIDIA GPU 和驱动，Codespace 默认环境无法完成。

```bash
# 1. 前往 https://developer.nvidia.com/isaac-gym 下载 Isaac Gym Preview 3
# 2. 解压并安装
cd isaacgym/python && pip install -e .
```

## 8. 训练模型（需要 GPU）

```bash
conda activate wmp
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0
```

- 训练需要约 **23GB GPU 内存**
- 建议至少 **10,000 次迭代**
- 推荐使用 A100 或同等规格 GPU

## 9. 可视化（需要预训练模型）

```bash
python legged_gym/scripts/play.py --task=a1_amp --sim_device=cuda:0 --terrain=climb
```

---

## 使用外部 GPU 算力

GitHub Codespace 本身不提供 GPU，但可以通过以下方式连接到外部 GPU 服务器进行训练。

### 方案一：SSH 隧道 + 远程 GPU 服务器

适用于：拥有本地 GPU 工作站或云 GPU 服务器（如阿里云、腾讯云、AWS EC2 G系列等）。

#### 1. 在 GPU 服务器上准备环境

```bash
# 安装 conda（如果尚未安装）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 克隆项目并安装依赖
git clone https://github.com/bytedance/WMP.git
cd WMP
conda create -n wmp python=3.8 -y
conda activate wmp

# 安装 PyTorch（GPU 版本）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 安装 Isaac Gym Preview 3（从 NVIDIA 官网下载）
# https://developer.nvidia.com/isaac-gym
cd isaacgym/python && pip install -e .

# 安装其他依赖
sudo apt-get install -y build-essential ninja-build libgl1 libglib2.0-0
pip install setuptools==59.5.0 ruamel_yaml==0.17.4 opencv-contrib-python
pip install -r requirements.txt
```

#### 2. 通过 SSH 从 Codespace 连接到 GPU 服务器

```bash
# 在 Codespace 终端中
ssh -L 6006:localhost:6006 user@your-gpu-server-ip

# 参数说明：
# -L 6006:localhost:6006  将服务器的 TensorBoard 端口转发到 Codespace
# user@your-gpu-server-ip  替换为你的 GPU 服务器地址
```

#### 3. 在 GPU 服务器上直接运行训练

```bash
# 在 GPU 服务器上
conda activate wmp
cd WMP
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0
```

#### 4. （可选）使用 VS Code Remote SSH 直接开发

直接在本地 VS Code 中通过 Remote SSH 连接到 GPU 服务器进行开发，完全绕过 Codespace：

```bash
# 在本地 VS Code 中安装 Remote - SSH 扩展
# 配置 ~/.ssh/config
Host gpu-server
    HostName your-gpu-server-ip
    User your-username
    Port 22
    IdentityFile ~/.ssh/id_rsa
```

### 方案二：Tailscale VPN 组网

适用于：需要频繁、安全地连接多台设备的场景。

1. 在 [Tailscale](https://tailscale.com) 注册账号（免费版支持最多 3 台设备）。
2. 在 GPU 服务器上安装 Tailscale：
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ```
3. 在 Codespace 中安装 Tailscale：
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ```
4. 连接后，两台设备会获得一个 `100.x.x.x` 的内网 IP，可以直接 SSH 连接：
   ```bash
   ssh user@100.x.x.x
   ```

### 方案三：云 GPU 用于 AI 调试（推荐）

> 🎯 **需求：** 你只需要一个云 GPU 来给 AI 做代码调试和快速验证，不需要跑完整训练。
> 因此核心要求是：① 支持 SSH 连接；② 按需付费（不用不花钱）；③ 能运行 Isaac Gym。

#### 方案对比

| 服务商 | 适合调试？ | 最低价格 | GPU 类型 | SSH 连接 | 特点 |
|--------|-----------|---------|---------|---------|------|
| **vast.ai** ⭐ | ✅ **最推荐** | ~$0.2/小时 | RTX 3090/4090 | ✅ 原生 SSH | 按秒计费，实例模板多，性价比极高 |
| **RunPod** | ✅ 推荐 | ~$0.2/小时 | RTX 3090/4090 | ✅ 原生 SSH | 界面简洁，支持 VS Code |
| **Lambda Labs** | ✅ | ~$0.5/小时 | RTX 4090 | ✅ 原生 SSH | PyTorch 预装，学术折扣 |
| **阿里云/腾讯云** | ✅ | ~¥5/小时 | V100 | ✅ 原生 SSH | 国内速度快，延迟低 |
| **Google Colab** | ⚠️ 有限 | 免费/按量 | T4/V100 | ❌ 不支持 SSH | 适合小实验，不适合 Isaac Gym |

#### 推荐方案：vast.ai（性价比最高）

##### 为什么选 vast.ai

- 💰 **便宜**：RTX 3090 约 $0.2/小时，按秒计费，调试完就关掉，几乎不花钱
- 🔌 **原生 SSH**：就像拥有一台自己的 GPU 服务器
- 📦 **预装镜像**：可以选择 PyTorch + CUDA 已配置好的镜像，省去安装时间
- 🌍 **全球节点**：可以选择离你近的节点，延迟低

##### 使用步骤

**1. 注册并充值**

- 打开 https://vast.ai
- 注册账号，充值少量金额（比如 $10 够用很久）

**2. 租用实例**

```bash
# 在 vast.ai 控制台搜索实例，筛选条件：
# - GPU: RTX 3090 或 RTX 4090（显存24GB，够用）
# - 价格: < $0.3/小时
# - 镜像: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
# - 磁盘: 至少 20GB

# 租用后，你会得到 SSH 连接命令，类似：
ssh -p 12345 root@xxx.vast.ai
```

**3. 安装 Isaac Gym**

```bash
# SSH 连接到 vast.ai 实例
ssh -p 12345 root@xxx.vast.ai

# 安装系统依赖
apt-get update
apt-get install -y build-essential ninja-build libgl1 libglib2.0-0

# 安装 Isaac Gym
# 从 https://developer.nvidia.com/isaac-gym 下载后 scp 上传到实例
# 或者在实例中直接 wget（如果有直链）
# 解压并安装
cd isaacgym/python && pip install -e .

# 安装项目依赖
pip install setuptools==59.5.0 ruamel_yaml==0.17.4 opencv-contrib-python
```

**4. 配置 SSH 连接（在 Codespace 中配置）**

```bash
# 在 Codespace 终端中执行

# 4.1 生成 SSH 密钥（如果还没有）
ssh-keygen -t ed25519 -C "vast-ai"

# 4.2 查看公钥并复制
cat ~/.ssh/id_ed25519.pub
# 输出类似：ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... root@codespace

# 4.3 将公钥添加到 vast.ai 账号
# 打开 https://cloud.vast.ai/account/  → SSH Keys → Add
# 把上面复制的公钥粘贴进去

# 4.4 配置 SSH config 方便连接
cat >> ~/.ssh/config << 'EOF'
Host vast-ai
    HostName xxx.vast.ai    # 替换为实例的 IP
    Port 12345               # 替换为实例的端口
    User root
    ServerAliveInterval 60
    StrictHostKeyChecking no
EOF

# 4.5 测试连接
ssh vast-ai
# 第一次连接会提示确认，输入 yes 即可
```

**5. 安装 WMP 环境（仅需一次）**

```bash
# SSH 连接到 vast.ai
ssh vast-ai

# 安装系统依赖
apt-get update
apt-get install -y build-essential ninja-build libgl1 libglib2.0-0

# 克隆项目
git clone https://github.com/AnthonySung/WMP.git
cd WMP

# 创建 conda 环境
conda create -n wmp python=3.8 -y

# 安装 PyTorch（CUDA 11.7）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 安装 Isaac Gym（从 GitHub Releases 下载）
gh release download v0.1.0-alpha --repo AnthonySung/WMP
tar -xzf IsaacGym_Preview_3_Package.tar.gz
cd isaacgym/python && pip install -e .
cd ../..

或者

cd /home/WMP && \
wget https://github.com/AnthonySung/WMP/releases/download/v0.1.0-alpha/IsaacGym_Preview_3_Package.tar.gz && \
tar -xzf IsaacGym_Preview_3_Package.tar.gz && \
cd isaacgym/python && \
pip install -e . && \
python -c "import isaacgym; print('Isaac Gym 安装成功')"

# 安装其他依赖
pip install setuptools==59.5.0 ruamel_yaml==0.17.4 opencv-contrib-python
pip install -r requirements.txt

# 验证
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 装好后退出 SSH
exit
```

**6. 日常开发流程（VS Code Remote SSH + Codespace 配合）**

这是最推荐的方案——**在 Codespace 中写代码，在 vast.ai 上调试**，两边用 VS Code 无缝切换。

#### 第一步：在 Codespace 中写代码（不花一分钱）

在 Codespace 的 VS Code 中正常使用 AI 编程助手写代码、改代码。改完后推送到 GitHub：

```bash
git add .
git commit -m "修改了 xxx"
git push origin master
```

#### 第二步：启动 vast.ai 实例

打开 https://cloud.vast.ai/instances，找到你的实例，点击 **Start**。

等待状态变为 **"Open"** 或 **"Connect"**（通常几秒到几十秒）。

#### 第三步：用 VS Code Remote SSH 连接到 vast.ai

在 Codespace 的 VS Code 中：

1. 安装 **Remote - SSH** 扩展（如果还没装）
2. 按 `F1` 或 `Ctrl+Shift+P`
3. 输入 `Remote-SSH: Connect to Host` → 选择 `vast-ai`
4. 新窗口打开后，点击 **Open Folder** → `/root/WMP`

现在你就在 vast.ai 的 GPU 环境里了！可以：

- 在 VS Code 终端中运行训练命令
- 直接在 vast.ai 上拉取最新代码
- 调试、验证代码

```bash
# 在 vast.ai 的 VS Code 终端中
cd /root/WMP
git pull origin master
conda activate wmp

# 运行快速调试（比如只跑 100 次迭代验证）
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0 --max_iterations=100
```

#### 第四步：切回 Codespace 继续写代码

调试完后，在 VS Code 中：

1. `F1` → `Remote-SSH: Close Remote Connection`
2. 自动回到 Codespace 的本地环境

#### 第五步：停止 vast.ai 实例（停止计费）

打开 https://cloud.vast.ai/instances，点击 **Stop**。

> **注意：** 停止后环境会保留，下次 Start 后所有文件都在。每月只需付少量存储费（约 $1.4/月）。

### 完整工作流图示

```
┌─────────────────────────────────────────────────────────┐
│  Codespace（免费）                                       │
│  ┌─────────────────────────────────────────────────────┐│
│  │  VS Code + AI 编程助手                              ││
│  │  写代码、改代码 → git push                          ││
│  │  不需要 GPU，不花一分钱                              ││
│  └─────────────────────────────────────────────────────┘│
│                           │                              │
│             需要调试时，Remote SSH 一键连接               │
│                           ▼                              │
│  ┌─────────────────────────────────────────────────────┐│
│  │  vast.ai 云 GPU（按秒计费 $0.2/小时）                ││
│  │  VS Code Remote SSH 直接连接                        ││
│  │  git pull → conda activate wmp → 运行调试           ││
│  │  调完关掉，每次调试约 $0.03（2毛钱）                  ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

**费用总结：**

| 项目 | 费用 |
|------|------|
| Codespace 写代码 | **免费**（在免费额度内） |
| vast.ai 调试（10分钟/次） | **~$0.03（2毛钱）** |
| vast.ai 存储（不开机时） | **~$1.4/月** |
| **每月总花费（假设调试20次）** | **~$2（约15元）** |

**5. 调试完成后释放实例（可选）**

```bash
# 如果长时间不用，可以在 vast.ai 控制台 Destroy 实例
# 注意：Destroy 会删除所有数据，下次需要重新安装环境
# 推荐用 Stop 而不是 Destroy
```

#### 备选方案：阿里云/腾讯云 GPU 服务器

如果国内访问 vast.ai 速度慢，可以选择国内云服务商：

```bash
# 以阿里云为例：
# 1. 创建 GPU 实例（ecs.gn6v-c10g1.20xlarge，V100）
# 2. 选择 Ubuntu 20.04 + CUDA 11.7 镜像
# 3. 安全组开放 22 端口

# SSH 连接
ssh root@<公网IP>

# 安装环境（同上）
git clone https://github.com/bytedance/WMP.git
cd WMP
# ... 安装依赖 ...
```

> **💡 建议：** 先用 vast.ai 试试，$5 就能调试很多次。如果网络不好再换阿里云。两种方式都支持 VS Code Remote SSH，AI 改完代码直接在 GPU 上验证，体验和本地开发一样流畅。

### 方案四：GitHub Actions + 自托管 Runner

适用于：需要自动化训练流水线的场景。

#### 工作原理

```
你 push 代码到 GitHub
        │
        ▼
GitHub Actions 收到通知
        │
        ▼
分配任务给自托管 Runner（公司 GPU 服务器）
        │
        ▼
GPU 服务器执行训练
        │
        ▼
训练完成 → 模型/日志自动上传到 GitHub
        │
        ▼
你在 Codespace 中 pull 即可获取结果
```

#### 第一步：在公司 GPU 服务器上注册自托管 Runner

1. 打开仓库：`https://github.com/bytedance/WMP` → **Settings** → **Actions** → **Runners** → **New self-hosted runner**
2. 选择 Linux 系统，复制注册命令
3. 在 GPU 服务器上执行：

```bash
# SSH 到公司 GPU 服务器
ssh songanyang@10.230.117.139

# 创建 runner 目录
mkdir actions-runner && cd actions-runner

# 下载 GitHub Actions Runner（版本号以 GitHub 页面提示为准）
curl -o actions-runner-linux-x64-2.317.0.tar.gz \
    -L https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-linux-x64-2.317.0.tar.gz

# 解压
tar xzf actions-runner-linux-x64-2.317.0.tar.gz

# 注册 Runner（用 GitHub 页面上获取的 token）
./config.sh --url https://github.com/bytedance/WMP --token YOUR_TOKEN

# 启动 Runner
./run.sh

# 让 Runner 后台运行（作为服务）
sudo ./svc.sh install
sudo ./svc.sh start
```

#### 第二步：创建工作流文件

创建 `.github/workflows/train.yml`：

```yaml
name: WMP Training

on:
  push:
    branches: [ master ]
  workflow_dispatch:
    inputs:
      max_iterations:
        description: '训练迭代次数'
        required: true
        default: '100000'
      task:
        description: '训练任务'
        required: true
        default: 'a1_amp'

jobs:
  train:
    runs-on: self-hosted
    timeout-minutes: 1440  # 24小时超时

    steps:
      - uses: actions/checkout@v4

      - name: Setup conda
        run: |
          source ~/.bashrc
          conda activate wmp
          echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
          nvidia-smi

      - name: Run training
        run: |
          source ~/.bashrc
          conda activate wmp
          python legged_gym/scripts/train.py \
            --task=${{ github.event.inputs.task || 'a1_amp' }} \
            --headless \
            --sim_device=cuda:0 \
            --max_iterations=${{ github.event.inputs.max_iterations || '100000' }}

      - name: Upload logs
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: logs-${{ github.run_id }}
          path: logs/

      - name: Upload model
        if: success()
        uses: actions/upload-artifact@v4
        with:
          name: model-${{ github.run_id }}
          path: |
            **/model_*.pt
```

#### 第三步：使用流程

```bash
# 在 Codespace 中修改代码后
git add .
git commit -m "调整训练参数"
git push origin master
# ✅ 训练自动在 GPU 服务器上开始！
# 去 https://github.com/bytedance/WMP/actions 查看进度
# 完成后在 Actions 页面下载 Artifacts（模型+日志）
```

#### 方案四的优缺点

| 优点 | 缺点 |
|------|------|
| ✅ **不需要公网 IP** — Runner 主动连接 GitHub | ❌ **AI 改代码时无法 debug/验证**（见下方说明） |
| ✅ **自动化** — push 代码即触发训练 | ❌ **初始配置稍复杂** |
| ✅ **结果自动保存** — 模型和日志自动上传 | ❌ **训练日志实时查看不如 SSH 直观** |
| ✅ **版本追踪** — 每次训练的代码版本一一对应 | |
| ✅ **团队协作** — 团队成员都可触发训练 | |

#### ⚠️ 重要限制说明

> **问题：AI 改代码时没有环境进行 debug 和验证**

这个方案中，Codespace 只有 Python 基础环境，**没有 Isaac Gym**（因为需要 GPU），所以 AI 在修改代码后无法立即运行验证。解决方法：

**方案 A：结合方案二（Tailscale）使用**
- 用 **GitHub Actions** 做正式训练（自动化、可追溯）
- 同时在 GPU 服务器上安装 Tailscale，需要快速验证时手动 SSH 上去运行
- 这样既有自动化流水线，又可以随时 debug

**方案 B：在 Codespace 中做单元测试**
- 将不依赖 Isaac Gym 的模块（如策略网络、数据处理）提取出来单独测试
- 创建轻量级测试脚本，在 Codespace 中用 CPU 运行：

```bash
# 例如测试网络前向传播
python -c "
import torch
from rsl_rl.modules.actor_critic_wmp import ActorCriticWMP
model = ActorCriticWMP(42, 42, 12)
dummy_input = torch.randn(1, 51)
output = model.actor(dummy_input)
print('前向传播测试通过:', output.shape)
"
```

**方案 C：本地 GPU 工作站 + VS Code Remote SSH**
- 如果本地有 GPU 工作站，直接用 VS Code 的 Remote SSH 连接到工作站开发
- AI 改代码后可以直接在本地 GPU 上验证
- 完全绕过 Codespace 的限制

---

## 注意事项

1. **Isaac Gym** 需要 NVIDIA GPU 和对应驱动，无法在 CPU-only 环境运行。
2. 训练需要约 **23GB GPU 内存**，建议使用 A100 或类似规格的 GPU。
3. 每次打开新终端时，需要先激活环境：`conda activate wmp`。
4. 如果在有 GPU 的机器上使用，推荐安装 CUDA 11.7 及以上版本的驱动。
5. 使用 SSH 连接时，确保 GPU 服务器的防火墙允许 SSH 端口（默认 22）访问。
