# ComfyUI远程控制设置

## 提醒日期
**2026年2月23日 (3天后)**

---

## 需要完成的任务

### 1. 公司电脑安装 ComfyUI

**方式A - Docker (推荐)**:
```bash
# 安装Docker Desktop
# https://www.docker.com/products/docker-desktop

# 启动ComfyUI容器
docker run -d -p 8188:8188 \
  -v /Users/你的用户名/comfyui/models:/workspace/ComfyUI/models \
  -v /Users/你的用户名/comfyui/output:/workspace/ComfyUI/output \
  --gpus all \
  --name comfyui \
  comfyanonymous/comfyui:latest
```

**方式B - 直接运行**:
```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install xformers

# 克隆仓库
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 下载模型
# 把模型放到 models/checkpoints/

# 启动
./run_gpu.sh  # 或 ./run_cpu.sh
```

### 2. 安装 cloudflared (建立隧道)

```bash
# macOS
brew install cloudflared

# 测试运行
cloudflared tunnel --url http://localhost:8188
# 会得到: https://abc123.trycloudflare.com
```

### 3. 配置开机自启 (可选)

```bash
# 创建启动脚本
cat > ~/start_comfyui.sh << 'EOF'
#!/bin/bash
cd ~/ComfyUI
./run_gpu.sh &
sleep 5
cloudflared tunnel --url http://localhost:8188
EOF

chmod +x ~/start_comfyui.sh
```

---

## 家里OpenClaw端配置

配置好后会给你一个Python脚本，你只需要发送命令即可生成图片：

```python
# 示例用法
generate_image("a beautiful sunset over mountains")
# 返回图片给你
```

---

## 注意事项

- 确保公司电脑GPU足够 (建议8GB+显存)
- cloudflared免费版有流量限制
- 首次启动需要下载模型 (约10GB)

---

*创建于: 2026-02-20*
