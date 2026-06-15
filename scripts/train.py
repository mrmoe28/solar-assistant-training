#!/usr/bin/env python3
"""Solar Business Assistant Training - Llama 3.2 3B with Unsloth QLoRA."""

import os
import sys
import json
import time
import threading
from datetime import datetime

import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

import unsloth
from unsloth import FastLanguageModel

# Globals
training_logs = []
current_stats = {
    "status": "Idle",
    "epoch": 0,
    "step": 0,
    "loss": 0.0,
    "learning_rate": 0.0,
    "gpu_memory_mb": 0,
    "time_elapsed": "0:00:00",
}
training_thread = None
stop_training = False

# Config
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit"
DATASET_PATH = "data/solar_training_data.jsonl"
OUTPUT_DIR = "models/solar-assistant-lora"
MAX_SEQ_LENGTH = 2048

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10
SAVE_STEPS = 50
LOG_STEPS = 5


class UILogCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()

    def on_train_begin(self, args, state, control, **kwargs):
        global current_stats
        current_stats["status"] = "Training"
        current_stats["epoch"] = 0
        current_stats["step"] = 0
        current_stats["loss"] = 0.0
        log_msg("[START] Training began")

    def on_log(self, args, state, control, logs=None, **kwargs):
        global current_stats, training_logs
        if logs is None:
            return
        current_stats["step"] = state.global_step
        current_stats["epoch"] = round(state.epoch, 2) if state.epoch else 0
        if "loss" in logs:
            current_stats["loss"] = round(logs["loss"], 4)
        if "learning_rate" in logs:
            current_stats["learning_rate"] = round(logs["learning_rate"], 8)
        if torch.cuda.is_available():
            mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
            current_stats["gpu_memory_mb"] = round(mem_mb, 1)
        elapsed = time.time() - self.start_time
        current_stats["time_elapsed"] = format_time(elapsed)
        log_str = json.dumps(logs, indent=None)
        log_msg(f"[STEP {state.global_step}] {log_str}")

    def on_train_end(self, args, state, control, **kwargs):
        global current_stats
        current_stats["status"] = "Completed"
        log_msg("[DONE] Training finished")

    def on_save(self, args, state, control, **kwargs):
        log_msg(f"[SAVE] Checkpoint at step {state.global_step}")


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def log_msg(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    training_logs.append(entry)
    while len(training_logs) > 200:
        training_logs.pop(0)
    print(entry)


# Chat template for Llama 3.2
USER_START = "<|start_header_id|>user<|end_header_id|>"
ASSISTANT_START = "<|start_header_id|>assistant<|end_header_id|>"
EOT_TOKEN="<|eot_id|>"
NEWLINE = chr(10)


def run_training():
    global stop_training

    log_msg("Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    log_msg("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    log_msg("Loading dataset...")
    raw_data = load_dataset("json", data_files=DATASET_PATH, split="train")
    formatted_texts = []
    for i in range(len(raw_data)):
        instruction = raw_data["instruction"][i]
        input_text = raw_data["input"][i] if "input" in raw_data.column_names else ""
        output = raw_data["output"][i]
        if input_text and input_text.strip():
            user_msg = f"{instruction}\n\nInput:\n{input_text}"
        else:
            user_msg = instruction
        text = (
            USER_START + NEWLINE + NEWLINE + user_msg + EOT_TOKEN + NEWLINE
            + ASSISTANT_START + NEWLINE + NEWLINE + output + EOT_TOKEN + NEWLINE
        )
        formatted_texts.append(text)

    dataset = Dataset.from_dict({"text": formatted_texts})
    log_msg(f"Dataset loaded: {len(dataset)} examples")

    log_msg("Starting SFTTrainer...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        eos_token="<|eot_id|>",  # Llama 3.2 end-of-turn token
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        callbacks=[UILogCallback()],
    )

    trainer.train()

    log_msg("Saving LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    config_path = os.path.join(OUTPUT_DIR, "training_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRAD_ACCUM,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "dataset_size": len(dataset),
            "trained_at": datetime.now().isoformat(),
        }, f, indent=2)

    log_msg(f"[OK] Training complete. Adapter saved to: {OUTPUT_DIR}")


def build_ui():
    import gradio as gr

    def start_training_fn():
        global training_thread, stop_training
        if training_thread is not None and training_thread.is_alive():
            return "Training already running!"
        stop_training = False
        training_logs.clear()
        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()
        return "Training started! Watch logs below."

    def get_logs():
        return "\n".join(training_logs[-50:]) if training_logs else "No logs yet..."

    def get_stats():
        return json.dumps(current_stats, indent=2)

    def test_model_fn(instruction, input_text):
        import gc
        if not os.path.exists(OUTPUT_DIR + "/adapter_config.json"):
            return "Model not trained yet. Train first!"
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                adapter_name=OUTPUT_DIR,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            if input_text.strip():
                user_msg = f"{instruction}\n\nInput:\n{input_text}"
            else:
                user_msg = instruction
            messages = [{"role": "user", "content": user_msg}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if user_msg in response:
                response = response.split(user_msg)[-1].strip()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return response
        except Exception as e:
            return f"Error: {str(e)}"

    with gr.Blocks(title="Solar Assistant Training") as demo:
        gr.Markdown("# [JOB] Solar Business Assistant - Model Training")
        gr.Markdown("Train a local LLM for CRM, quotes, invoices, and scheduling.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## [1] Configuration")
                gr.Markdown(f"**Model:** `{MODEL_NAME}`")
                gr.Markdown(f"**LoRA Rank:** `{LORA_R}` | **Alpha:** `{LORA_ALPHA}`")
                gr.Markdown(f"**Epochs:** `{EPOCHS}` | **LR:** `{LEARNING_RATE}`")
                gr.Markdown(f"**Max Length:** `{MAX_SEQ_LENGTH}` tokens")
                start_btn = gr.Button("[START] Begin Training", variant="primary")
                status_text = gr.Textbox(label="Status", value="Idle", interactive=False)
                refresh_btn = gr.Button("[REFRESH] Update Stats")
                gr.Markdown("## [2] Live Stats")
                stats_box = gr.JSON(label="Training Stats", value=current_stats)
            with gr.Column(scale=2):
                gr.Markdown("## [3] Training Logs")
                logs_box = gr.Textbox(
                    label="Logs (last 50)",
                    lines=25,
                    value="Click [START] to begin training...",
                    interactive=False,
                )
        with gr.Row():
            with gr.Column():
                gr.Markdown("## [4] Test Your Model")
                gr.Markdown("After training completes, test the model here:")
                test_instruction = gr.Textbox(
                    label="Instruction",
                    value="Create a CRM entry for a new solar lead",
                    lines=2,
                )
                test_input = gr.Textbox(
                    label="Input Context",
                    value="Name: Tom Anderson, Phone: 404-555-0147",
                    lines=3,
                )
                test_btn = gr.Button("[TEST] Generate Response")
                test_output = gr.Textbox(label="Model Output", lines=10, interactive=False)

        start_btn.click(start_training_fn, outputs=status_text)
        refresh_btn.click(get_logs, outputs=logs_box)
        refresh_btn.click(get_stats, outputs=stats_box)
        test_btn.click(test_model_fn, inputs=[test_instruction, test_input], outputs=test_output)

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print(" Solar Business Assistant - Training & Monitoring UI ")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("-" * 60)
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        print("Running in CLI training mode...")
        run_training()
    else:
        print("Launching Gradio dashboard...")
        print("Open: http://localhost:7860")
        print("Press Ctrl+C to stop")
        app = build_ui()
        app.launch(server_name="0.0.0.0", server_port=7860, share=False)
