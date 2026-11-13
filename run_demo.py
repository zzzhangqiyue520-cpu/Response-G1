# -*- coding: utf-8 -*-
"""命令行跑一次完整推理，绕开 Gradio 界面和 SSH 隧道，用于诊断和录屏演示。

用法：
    python run_demo.py /root/autodl-tmp/test.mp4 "视频里有没有出现闯红灯的车？"
"""
import subprocess
import sys
import time

import app

CKPT = "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"


def show_gpu():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
         "--format=csv,noheader"],
        capture_output=True, text=True)
    print("GPU 显存(总/已用/空闲):", r.stdout.strip(), flush=True)


def main():
    video = sys.argv[1] if len(sys.argv) >= 2 else "/root/autodl-tmp/test.mp4"
    question = (sys.argv[2] if len(sys.argv) >= 3
                else "视频里有没有出现闯红灯的车？")

    print("=== 加载模型 ===", flush=True)
    print(app.load_model(CKPT), flush=True)
    show_gpu()

    print(f"=== 开始推理：视频={video}，问题={question} ===", flush=True)
    t0 = time.time()
    trigger, answer, logs = app.run_proactive(video, question, "", 0, 15)
    print(f"=== 推理完成，总耗时 {time.time() - t0:.1f} 秒 ===", flush=True)
    print(f"回答时间：{trigger}", flush=True)
    print(f"模型回答：{answer}", flush=True)
    print(f"决策过程：\n{logs}", flush=True)
    show_gpu()


if __name__ == "__main__":
    main()