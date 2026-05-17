#!/bin/bash
# =============================================================================
# WMP Docker 镜像构建脚本
# Isaac Gym 从 GitHub Releases 自动下载
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="wmp-training"
IMAGE_TAG="latest"
GITHUB_REPO="AnthonySung/WMP"
GYM_RELEASE_TAG="v0.1.0-alpha"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  WMP Docker 镜像构建脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查 GitHub CLI 是否登录
if ! gh auth status 2>/dev/null; then
    echo -e "${YELLOW}[!] GitHub CLI 未登录，尝试用 Token 登录...${NC}"
    if [ -n "$GH_TOKEN" ] || [ -n "$GITHUB_TOKEN" ]; then
        echo -e "${GREEN}[✓] 已找到 GitHub Token${NC}"
    else
        echo -e "${YELLOW}[!] 未设置 GH_TOKEN 或 GITHUB_TOKEN${NC}"
        echo -e "${YELLOW}构建时可能无法下载 Isaac Gym，请确保 Release 是公开的${NC}"
    fi
fi

# 检查 CUDA 驱动 (宿主机)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}[✓] NVIDIA 驱动已安装${NC}"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
else
    echo -e "${YELLOW}[!] 未检测到 NVIDIA 驱动 (容器运行时需要 nvidia-container-toolkit)${NC}"
fi

# 构建镜像 (Isaac Gym 在 Dockerfile 内通过 gh 从 Release 下载)
echo -e "${GREEN}[...] 开始构建镜像 ${IMAGE_NAME}:${IMAGE_TAG} ...${NC}"
echo -e "${GREEN}[...] Isaac Gym 将从 ${GITHUB_REPO} Release ${GYM_RELEASE_TAG} 自动下载${NC}"

docker build \
    --build-arg GITHUB_REPO="${GITHUB_REPO}" \
    --build-arg GYM_RELEASE_TAG="${GYM_RELEASE_TAG}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "$SCRIPT_DIR/Dockerfile.wmp" \
    "$PROJECT_DIR"

# 验证
echo -e "${GREEN}[...] 验证镜像...${NC}"
docker run --rm --gpus all "${IMAGE_NAME}:${IMAGE_TAG}" python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
import isaacgym
print('Isaac Gym: OK')
"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  构建完成!${NC}"
echo -e "${GREEN}  镜像: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo -e "${GREEN}  运行: docker run --gpus all -it ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo -e "${GREEN}========================================${NC}"
