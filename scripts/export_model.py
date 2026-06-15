#!/usr/bin/env python3
"""
Export trained model to Ollama-compatible GGUF format.
Merges LoRA adapter into base model, quantizes, and creates Modelfile.
"""

import os
import sys
import torch
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit"
ADAPTER_PATH = "models/solar-assistant-lora"
OUTPUT_DIR = "outputs"

def export_to_gguf():
    print("=" * 60)
    print(" Solar Assistant - Export to Ollama")
    print("=" * 60)

    print(f"[1/5] Loading base model + adapter from: {ADAPTER_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        adapter_name=ADAPTER_PATH,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    print("[2/5] Merging LoRA adapter into base model...")
    model = model.merge_and_unload()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[3/5] Saving merged model in HuggingFace format...")
    merged_path = os.path.join(OUTPUT_DIR, "solar-assistant-merged")
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"  -> Saved to: {merged_path}")

    print("[4/5] Exporting to GGUF (Q4_K_M quantization)...")
    gguf_path = os.path.join(OUTPUT_DIR, "solar-assistant-q4.gguf")
    model.save_pretrained_gguf(
        OUTPUT_DIR,
        tokenizer,
        quantization_method=["q4_k_m"],
    )
    # Unsloth saves as unsloth.Q4_K_M.gguf in the dir, rename it
    src = os.path.join(OUTPUT_DIR, "unsloth.Q4_K_M.gguf")
    if os.path.exists(src):
        os.rename(src, gguf_path)
        print(f"  -> Saved GGUF to: {gguf_path}")
    else:
        # Try alternative name
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".gguf"):
                src = os.path.join(OUTPUT_DIR, f)
                os.rename(src, gguf_path)
                print(f"  -> Saved GGUF to: {gguf_path}")
                break

    print("[5/5] Creating Ollama Modelfile...")
    modelfile_content = f"""FROM {gguf_path}

SYSTEM """ + '"""' + """You are SolarBot, an AI assistant specialized for EkoSolar LLC — a residential and commercial solar installation company based in Atlanta, GA.

Your capabilities:
- Create and manage CRM entries for solar leads and customers
- Generate accurate solar quotes with equipment, labor, and financing breakdowns
- Draft professional emails and follow-up messages
- Calculate system sizing based on energy usage and roof characteristics
- Create invoices and track payments
- Schedule appointments and send reminders
- Generate maintenance checklists and onboarding sequences
- Calculate financing options (cash, loan, lease)
- Summarize sales reports and pipeline status

Rules:
- Always respond with structured JSON when the user requests data entry (CRM, quotes, invoices)
- For emails and messages, use professional but friendly tone
- Include solar-specific details: panel types (REC, Q.Cells, LG), inverters (SolarEdge, Enphase), batteries (Tesla Powerwall, Enphase IQ), tax credits (30% federal ITC), local incentives (Georgia)
- Ask clarifying questions if information is missing
- Never make up customer data — if unsure, ask the user

You help small solar businesses work smarter, not harder.""" + '"""' + f"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""

    modelfile_path = os.path.join(OUTPUT_DIR, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    print(f"  -> Saved Modelfile to: {modelfile_path}")

    print("-" * 60)
    print("[OK] Export complete!")
    print(f"  GGUF model: {gguf_path}")
    print(f"  Modelfile:  {modelfile_path}")
    print("")
    print("To use with Ollama:")
    print(f"  cd {OUTPUT_DIR}")
    print("  ollama create solar-assistant -f Modelfile")
    print("  ollama run solar-assistant")
    print("=" * 60)


if __name__ == "__main__":
    if not os.path.exists(ADAPTER_PATH + "/adapter_config.json"):
        print(f"[X] No trained adapter found at: {ADAPTER_PATH}")
        print("    Train the model first with: python scripts/train.py --train")
        sys.exit(1)
    export_to_gguf()
