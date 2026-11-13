# -*- coding: utf-8 -*-
"""冒烟测试：验证 Qwen3-VL 模型能加载并完成一次最小推理，不依赖评测数据。"""
import argparse
import numpy as np
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL smoke test")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="本地 Qwen3-VL 权重目录")
    args = parser.parse_args()

    print("[1/4] 加载处理器 ...")
    processor = AutoProcessor.from_pretrained(args.ckpt_path, trust_remote_code=True)

    print("[2/4] 加载模型（bfloat16，自动分配设备）...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("    模型设备:", model.device)

    # 用 PIL 生成一张纯色图，避免依赖外部图片 URL
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "图中是什么？请用一句话回答。"},
            {"type": "image", "image": img},
        ],
    }]

    print("[3/4] 预处理输入 ...")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    print("[4/4] 推理 ...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_ids = [
        g[len(inp):] for inp, g in zip(inputs.input_ids, generated_ids)
    ]
    out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("模型输出:", out)
    print("\n✅ 冒烟测试通过：环境、依赖、模型加载、推理均正常。")


if __name__ == "__main__":
    main()