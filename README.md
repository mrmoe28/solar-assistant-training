# Solar Business Assistant Training Project

This project trains a local LLM to handle solar business office tasks: CRM management, scheduling, invoicing, quotes, and customer communications.

## Quick Start

```bash
cd ~/solar-assistant-training
source venv/bin/activate

# Launch training + monitoring UI
python scripts/train.py
```

Then open http://localhost:7860 in your browser to watch training live.

## Project Structure

```
solar-assistant-training/
├── data/                    # Training datasets (JSONL)
├── models/                  # Saved LoRA adapters + merged models
├── logs/                    # Training logs & metrics
├── scripts/                 # Python training & inference scripts
├── templates/               # Invoice/quote templates
├── outputs/                 # Exported GGUF models for Ollama
└── README.md
```

## Hardware

- Local RTX 3070 (8GB VRAM)
- Training: Unsloth QLoRA on Qwen2.5-7B-Instruct
- Inference: Ollama GGUF export for daily use

## Adding Your Own Data

Edit `data/solar_training_data.jsonl` with your real customer interactions, quote formats, and workflow examples. More data = better model.

## Using the Trained Model

After training completes:

```bash
# Merge LoRA into base model
python scripts/export_model.py

# Run in Ollama
ollama create solar-assistant -f outputs/Modelfile
ollama run solar-assistant
```

