# Isaac Gym Preview 3 安装指南

> 本文档记录了 Isaac Gym Preview 3 的安装步骤。
> Isaac Gym 是 NVIDIA 的物理仿真引擎，WMP 项目的训练依赖此环境。

---

## 前提条件

- **NVIDIA GPU** 和对应驱动（CUDA 11.7+）
- **Python 3.6 / 3.7 / 3.8**（推荐 3.8）
- 已安装 PyTorch（CUDA 版本）

## 获取安装包

### 方式一：从 GitHub Releases 下载（推荐）

安装包已上传到仓库的 Releases 页面：

```bash
# 下载最新 Release 中的安装包
gh release download v0.1.0-alpha --repo AnthonySung/WMP
# 或直接在浏览器中访问：
# https://github.com/AnthonySung/WMP/releases
```

### 方式二：从 NVIDIA 官网下载

访问 https://developer.nvidia.com/isaac-gym 下载 Isaac Gym Preview 3。

## 安装步骤

### 1. 解压安装包

```bash
# 将安装包放在项目根目录，然后解压
tar -xzf IsaacGym_Preview_3_Package.tar.gz

# 解压后会生成 isaacgym/ 目录
```

### 2. 安装 Isaac Gym Python 包

```bash
# 进入 python 目录并安装
cd isaacgym/python
pip install -e .
```

### 3. 验证安装

```bash
# 测试导入
python -c "import isaacgym; print('Isaac Gym 安装成功')"

# 测试 GPU 可用性
python -c "
from isaacgym import gymapi
gym = gymapi.acquire_gym()
print('Isaac Gym API 加载成功')
"
```

## 常见问题

### Q: 安装时提示 `gcc` 或 `build-essential` 缺失

```bash
sudo apt-get update
sudo apt-get install -y build-essential
```

### Q: 提示 `ninja` 缺失

```bash
sudo apt-get install -y ninja-build
```

### Q: 提示 `libgl` 相关错误

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

### Q: 提示 CUDA 版本不兼容

确保 CUDA 版本为 11.7+：

```bash
nvidia-smi
nvcc --version
```

## 完整环境安装（从零开始）

在一台全新的 GPU 服务器上搭建完整环境：

```bash
# 1. 克隆项目
git clone https://github.com/AnthonySung/WMP.git
cd WMP

# 2. 创建 conda 环境
conda create -n wmp python=3.8 -y
conda activate wmp

# 3. 安装 PyTorch（CUDA 11.7）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 4. 安装系统依赖
sudo apt-get update
sudo apt-get install -y build-essential ninja-build libgl1 libglib2.0-0

# 5. 安装 Isaac Gym
# 从 Releases 下载安装包
gh release download v0.1.0-alpha --repo AnthonySung/WMP
tar -xzf IsaacGym_Preview_3_Package.tar.gz
cd isaacgym/python && pip install -e .
cd ../..

# 6. 安装项目 Python 依赖
pip install setuptools==59.5.0 ruamel_yaml==0.17.4 opencv-contrib-python
pip install -r requirements.txt

# 7. 验证安装
python -c "
import torch
import isaacgym
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('Isaac Gym: OK')
"
```

---

## 注意事项

1. Isaac Gym Preview 3 是 NVIDIA 的专有软件，需要注册下载。
2. 训练需要约 **23GB GPU 内存**，建议使用 A100 或同等规格 GPU。
3. 安装包约 187MB，通过 GitHub Releases 管理（不使用 Git LFS）。
