# -*- coding: utf-8 -*-
"""图鉴视界——主动流式视频理解 Gradio 演示界面。

启动：
    python app.py --ckpt_path /root/autodl-tmp/models/Qwen3-VL-8B-Instruct --port 7860
"""
import argparse
import os
import sys
import urllib.parse


import cv2
import gradio as gr
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_streamingbench_proactive as ev
from response_graph.Scene_Graph import (
    Query_Graph_generation,
    Scene_Graph_generation_po_frame,
)
from qwen_online.utils import encode_images, whole_current_frame_tokens

# 采样参数（与评测脚本默认一致）
FPS = 1                # 每秒采样 1 帧
FRAME_INTERVAL = 1     # 每 1 帧处理一次
SG_INTERVAL = -1       # 每帧都生成场景图（-1 = 每帧）
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448

MODEL = None
PROCESSOR = None


def load_model(ckpt_path):
    global MODEL, PROCESSOR
    MODEL = ev.Qwen3VLForConditionalGeneration.from_pretrained(
        ckpt_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    PROCESSOR = ev.AutoProcessor.from_pretrained(
        ckpt_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True,
    )
    MODEL.eval()
    return "模型加载完成"


def build_gui_prompt(question, expected_output, context_graphs):
    """构建触发判断 prompt；expected_output 为空时改为通用判断。"""
    if context_graphs is not None:
        sg_text = "The relative scene graph shows:\n"
        for ts, sg in context_graphs:
            sg_text += (
                f"<{ts:.1f} seconds>:"
                f"{ev.format_scene_graphs_for_prompt_offline([sg])}\n"
            )
        if expected_output and expected_output.strip():
            query = (f'{question} Is it the right time to output '
                     f'"{expected_output.strip()}"? {sg_text} '
                     f'You can only answer yes or no.')
        else:
            query = (f'{question} Is it the right time to answer this '
                     f'question now? {sg_text} You can only answer yes or no.')
    else:
        if expected_output and expected_output.strip():
            query = (f'{question} Is it the right time to output '
                     f'"{expected_output.strip()}"? You can only answer yes or no.')
        else:
            query = (f'{question} Is it the right time to answer this '
                     f'question now? You can only answer yes or no.')

    prompt_trigger = ev.prompt.format(question=query)
    prompt_response = ev.prompt.format(question=question)
    return prompt_trigger, prompt_response


def run_proactive(video_path, question, expected_output, start_second, max_watch_seconds):
    if not video_path:
        return "未上传视频", "无", "请先上传一段视频"
    if not question or not question.strip():
        return "未输入问题", "无", "请输入要判断的问题"

    question = question.strip()
    expected_output = (expected_output or "").strip()

    # 1. 生成查询图
    query_graph = Query_Graph_generation(PROCESSOR, MODEL, question)

    # 2. 读取视频时长
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "视频无法打开", "无", "视频文件无法读取，请换一个视频"
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    total_duration = total_frames / video_fps if video_fps > 0 else 0

    # 3. 确定处理范围
    start_frame = int(start_second * FPS)
    if max_watch_seconds and max_watch_seconds > 0:
        end_second = min(start_second + max_watch_seconds, total_duration)
    else:
        end_second = total_duration
    max_frame = int(end_second * FPS)
    if max_frame <= start_frame:
        return "参数错误", "无", "观看时长太小或视频太短"

    memories = ev.Memory_Bank_Naive(memory_size=max_frame - start_frame + 1)

    logs = []
    trigger_time = None
    answer = "未触发回答"

    print(f"[图鉴视界] 开始分析：视频约 {total_duration:.1f} 秒，处理第 {start_frame}~{max_frame} 帧", flush=True)
    with torch.no_grad():
        for frame_idx in range(start_frame, max_frame + 1, FRAME_INTERVAL):
            timestamp = frame_idx / FPS
            print(f"[图鉴视界] 正在处理第 {frame_idx} 帧（{timestamp:.1f} 秒）...", flush=True)

            current_frame = ev.extract_frame(video_path, timestamp)
            if current_frame is None:
                continue

            cur_emb = PROCESSOR.image_processor(current_frame)
            cur_emb["pixel_values"] = cur_emb["pixel_values"].to(MODEL.device)
            cur_emb["image_grid_thw"] = cur_emb["image_grid_thw"].to(MODEL.device)
            cur_tokens, _, _ = encode_images(
                MODEL, cur_emb["pixel_values"], cur_emb["image_grid_thw"])
            cur_tokens = whole_current_frame_tokens(
                PROCESSOR, MODEL, timestamp, cur_tokens)
            cur_tokens = cur_tokens.unsqueeze(0).to(MODEL.device)
            del cur_emb
            memories.update(cur_tokens)

            # 在线场景图生成
            relative_idx = frame_idx - start_frame
            should_sg = (
                (SG_INTERVAL <= 0)
                or (relative_idx == 0)
                or (SG_INTERVAL > 0 and relative_idx % SG_INTERVAL == 0)
            )
            if should_sg:
                clip_tokens = memories.get_context_frames(context_window=1)
                clip_tokens = torch.cat(clip_tokens, dim=1).to(MODEL.device)
                cur_graph = Scene_Graph_generation_po_frame(
                    PROCESSOR, MODEL, clip_tokens, question, query_graph)
                if len(cur_graph.get("scene_graph", [])) != 0:
                    memories.update_graph(timestamp, cur_graph)
                    print(f"[图鉴视界] {timestamp:.1f}s 生成场景图", flush=True)
                    sg_short = ev.format_scene_graphs_for_prompt_offline(
                        [cur_graph])[:180]
                    logs.append(f"[{timestamp:.1f}s] 场景图：{sg_short}")

            # 检索相关场景图
            context_graphs = memories.get_graphs(MODEL, PROCESSOR, query_graph)
            if context_graphs is not None:
                logs.append(
                    f"[{timestamp:.1f}s] 检索到 {len(context_graphs)} 条相关场景图")

            # 构建触发 / 回答 prompt
            prompt_trigger, prompt_response = build_gui_prompt(
                question, expected_output, context_graphs)

            trig_ids = PROCESSOR.tokenizer(
                ["<|im_start|>user\n" + prompt_trigger + "<|im_end|>\n"],
                **ev.text_kwargs)
            trig_tensor = torch.tensor(trig_ids["input_ids"]).to(MODEL.device)
            trig_emb = MODEL.get_input_embeddings()(trig_tensor)

            resp_ids = PROCESSOR.tokenizer(
                ["<|im_start|>user\n" + prompt_response + "<|im_end|>\n"],
                **ev.text_kwargs)
            resp_tensor = torch.tensor(resp_ids["input_ids"]).to(MODEL.device)
            resp_emb = MODEL.get_input_embeddings()(resp_tensor)

            # Stage 1：判断是否该回答（yes/no）
            past_tokens = memories.get_context_frames(
                context_window=min(8, frame_idx - start_frame + 1))
            past_tokens = torch.cat(past_tokens, dim=1).to(MODEL.device)
            input_emb = torch.cat([past_tokens, trig_emb], dim=1).to(MODEL.device)

            gen_ids = PROCESSOR.tokenizer(
                ["<|im_start|>assistant\n"], **ev.text_kwargs)
            gen_tensor = torch.tensor(gen_ids["input_ids"]).to(MODEL.device)
            gen_emb = MODEL.get_input_embeddings()(gen_tensor)
            input_emb = torch.cat([input_emb, gen_emb], dim=1).to(MODEL.device)
            attn = torch.ones(input_emb.shape[:2], dtype=torch.long).to(MODEL.device)

            generated = MODEL.generate(
                inputs_embeds=input_emb, attention_mask=attn,
                **ev.generate_trigger_kwargs)
            out = PROCESSOR.batch_decode(
                generated, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            trigger_response = out[0]
            del input_emb, attn, generated, past_tokens, gen_tensor, gen_emb
            torch.cuda.empty_cache()

            print(f"[图鉴视界] {timestamp:.1f}s 触发判断：{trigger_response.strip()!r}", flush=True)
            if "yes" in trigger_response.strip().lower():
                # Stage 2：生成答案
                past_tokens = memories.get_context_frames(
                    context_window=min(8, frame_idx - start_frame + 1))
                past_tokens = torch.cat(past_tokens, dim=1).to(MODEL.device)
                input_emb = torch.cat([past_tokens, resp_emb], dim=1).to(MODEL.device)

                gen_ids = PROCESSOR.tokenizer(
                    ["<|im_start|>assistant\n"], **ev.text_kwargs)
                gen_tensor = torch.tensor(gen_ids["input_ids"]).to(MODEL.device)
                gen_emb = MODEL.get_input_embeddings()(gen_tensor)
                input_emb = torch.cat([input_emb, gen_emb], dim=1).to(MODEL.device)
                attn = torch.ones(input_emb.shape[:2], dtype=torch.long).to(MODEL.device)

                generated = MODEL.generate(
                    inputs_embeds=input_emb, attention_mask=attn,
                    **ev.generate_trigger_kwargs)
                out = PROCESSOR.batch_decode(
                    generated, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)
                answer = out[0]
                del input_emb, attn, generated, past_tokens, gen_tensor, gen_emb
                torch.cuda.empty_cache()

                trigger_time = timestamp
                logs.append(f"✅ 在第 {timestamp:.1f} 秒触发回答")
                break

    if trigger_time is None:
        trigger_str = "未触发（观看范围内信息不足）"
    else:
        t = int(trigger_time)
        trigger_str = f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}（第 {trigger_time:.1f} 秒）"

    print(f"[图鉴视界] 分析完成：触发时间={trigger_str}，回答={answer[:50]!r}", flush=True)
    return trigger_str, answer, "\n".join(logs) if logs else "（无日志）"


def build_ui():
    def _preview_video(value):
        path = None
        if isinstance(value, str):
            path = value
        elif value is not None:
            path = getattr(value, "path", None) or getattr(value, "name", None)
        if not path:
            return "<p style='color:#888;margin:8px 0'>上传视频后可在此预览</p>"
        src = "/gradio_api/file=" + urllib.parse.quote(str(path))
        return (
            f'<video src="{src}" controls playsinline '
            f'style="width:100%;max-height:320px;border-radius:8px;background:#000">'
            f"</video>"
        )

    with gr.Blocks(title="图鉴视界", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎬 图鉴视界 —— 主动流式视频理解演示\n"
            "模型边看视频边判断「什么时候该回答」，并在合适时机主动给出答案。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.File(label="上传视频")
                gr.Markdown("**视频预览**")
                video_preview = gr.HTML(
                    value="<p style='color:#888;margin:8px 0'>上传视频后可在此预览</p>")
                video.change(
                    fn=_preview_video,
                    inputs=video,
                    outputs=video_preview,
                )
                question = gr.Textbox(
                    label="问题（触发条件）",
                    placeholder="例如：视频中是否出现了闯红灯的车辆？",
                    lines=2,
                )
                expected = gr.Textbox(
                    label="预期输出（可选，留空则自动判断）",
                    placeholder="例如：有车辆闯红灯",
                    lines=1,
                )
                with gr.Accordion("高级设置（一般不用改）", open=False):
                    start_second = gr.Number(
                        label="开始观看时间（秒）", value=0, precision=0)
                    max_watch = gr.Number(
                        label="最多观看时长（秒）", value=30, precision=0)
                run_btn = gr.Button("🚀 开始分析", variant="primary")
            with gr.Column(scale=1):
                trigger_out = gr.Textbox(label="回答时间", interactive=False)
                answer_out = gr.Textbox(
                    label="模型回答", interactive=False, lines=3)
                log_out = gr.Textbox(
                    label="决策过程（场景图 + 触发判断）",
                    interactive=False, lines=18)

        run_btn.click(
            fn=run_proactive,
            inputs=[video, question, expected, start_second, max_watch],
            outputs=[trigger_out, answer_out, log_out],
        )

        gr.Examples(
            examples=[
                ["视频中是否出现了闯红灯的车辆？", "有车辆闯红灯"],
                ["画面里的人在做什么动作？", "有人在挥手"],
                ["视频中出现了什么动物？", "出现了一只狗"],
            ],
            inputs=[question, expected],
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("正在加载模型 ...")
    print(load_model(args.ckpt_path))

    demo = build_ui()
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()