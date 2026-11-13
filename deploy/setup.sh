#!/usr/bin/env bash
# ============================================================
#  Response-G1 云端一键部署脚本（在 Linux GPU 云服务器上运行）
#  用法：上传整个项目后，执行  bash deploy/setup.sh
#  默认适配：AutoDL / 矩池云 / 恒源云 等（root 用户 + PyTorch 镜像）
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---------- 可配置项（按需修改） ----------
# 模型：默认 8B；显存小可换成 4B/2B（如 Qwen/Qwen3-VL-4B-Instruct）
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
# 数据盘路径：恒源云为 /hy-tmp；AutoDL 为 /root/autodl-tmp（按平台修改）
MODEL_DIR="${MODEL_DIR:-/hy-tmp/models/Qwen3-VL-8B-Instruct}"
DATA_DIR="${DATA_DIR:-/hy-tmp/data}"
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "============================================"
echo " 第1步：检查 GPU 环境"
echo "============================================"
nvidia-smi || { echo "[错误] 未检测到 GPU，请确认已租用带 GPU 的实例"; exit 1; }
echo "Python 版本：$(python --version)"

echo ""
echo "============================================"
echo " 第2步：安装系统依赖（ffmpeg）"
echo "============================================"
if command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg 已存在，跳过"
else
  apt-get update -qq && apt-get install -y -qq ffmpeg
fi

echo ""
echo "============================================"
echo " 第3步：安装 Python 依赖（精简核心依赖）"
echo "============================================"
# 只装项目真正用到的库，torch 使用镜像预装版本，避免版本冲突
pip install -r "$SCRIPT_DIR/requirements_core.txt" -i "$PIP_INDEX"

echo ""
echo "============================================"
echo " 第4步：下载 Qwen3-VL 模型（ModelScope 国内源，快）"
echo "============================================"
pip install modelscope -i "$PIP_INDEX"
mkdir -p "$(dirname "$MODEL_DIR")"
python -c "from modelscope import snapshot_download; snapshot_download('$MODEL_ID', local_dir='$MODEL_DIR')"
echo "模型已下载到: $MODEL_DIR"

echo ""
echo "============================================"
echo " 第5步：冒烟测试（加载模型 + 一次最小推理）"
echo "============================================"
python "$SCRIPT_DIR/smoke_test.py" --ckpt_path "$MODEL_DIR"

echo ""
echo "============================================"
echo " ✅ 部署完成！模型已跑通"
echo "============================================"
echo " 模型位置: $MODEL_DIR"
echo " 数据目录: $DATA_DIR"
echo ""
echo " 下一步：下载评测数据并运行评测（见本脚本末尾注释）"
echo ""

# ============================================================
#  以下是「下载数据 + 跑完整评测」的说明，按需取消注释执行
# ============================================================
#
# ---- StreamingBench（主动输出任务）----
# 1) 从 StreamingBench 官网(streamingbench.github.io)下载：
#      - 任务标注: Proactive_Output.csv
#      - 视频:     StreamingBench_PO 目录（每个 sample_*/video.mp4）
#    放到 $DATA_DIR 下。
# 2) 运行：
#    cd $PROJECT_DIR
#    export CUDA_VISIBLE_DEVICES=0
#    python eval_streamingbench_proactive.py \
#      --ckpt_path $MODEL_DIR \
#      --task_csv $DATA_DIR/Proactive_Output.csv \
#      --video_dir $DATA_DIR/StreamingBench_PO \
#      --result_dir ./runs/streamingbench_proactive
#
# ---- OVO-Bench（CRR 主动子任务）----
# 1) 从 OVO-Bench 仓库(github.com/JoeLeelyf/OVO-Bench)下载：
#      - 标注: ovo_bench_new.json
#      - 视频: src_videos/
#    放到 $DATA_DIR 下。
# 2) 运行：
#    cd $PROJECT_DIR
#    export CUDA_VISIBLE_DEVICES=0
#    python eval_ovobench_proactive.py \
#      --ckpt_path $MODEL_DIR \
#      --task_json $DATA_DIR/ovo_bench_new.json \
#      --video_dir $DATA_DIR/src_videos \
#      --result_dir ./runs/ovobench_proactive
#
# ---- 跑完算指标 ----
#    python eval_streaming_results.py --result_file ./runs/.../xxx.jsonl --output_file ./runs/.../metrics.json
#    python eval_ovo_results.py --result_file ./runs/.../xxx.jsonl
# ============================================================