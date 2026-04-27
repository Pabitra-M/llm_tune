# =========================================================
# GPU MEMORY WAIT (MUST BE FIRST — BEFORE TORCH IMPORT)
# =========================================================

import os
import subprocess
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MIN_FREE_MIB  = 8000
POLL_INTERVAL = 60
MAX_WAIT      = 3600

def wait_for_gpu(min_free_mib=MIN_FREE_MIB, poll_interval=POLL_INTERVAL):
    start_time = time.time()

    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free",
                 "--format=csv,noheader,nounits"],
                text=True
            )

            rows = [
                (int(r.split(",")[0]), int(r.split(",")[1].strip()))
                for r in out.strip().splitlines()
            ]

            best_idx, best_free = max(rows, key=lambda x: x[1])

            if best_free >= min_free_mib:
                print(f"[GPU] Using GPU {best_idx} ({best_free} MiB free)")
                return str(best_idx)

            print(f"[GPU] Waiting... GPU {best_idx} has {best_free} MiB")

        except Exception as e:
            print(f"[GPU] Error: {e}")

        if time.time() - start_time > MAX_WAIT:
            print("[GPU] Timeout → using CPU")
            return ""

        time.sleep(poll_interval)

gpu_id = wait_for_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# =========================================================
# IMPORTS
# =========================================================

import json
import re
import torch
import numpy as np
import csv

# 🔥 IMPORTANT (no GUI plotting)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """ You are an expert evaluator (judge model) for factual question-answering systems. You will be given: 1. A question 2. A ground truth answer (fact-based, domain-specific) 3. Four generated answers: - Base Answer (original model output) - Empathy Tone Answer - Creative Tone Answer - Logical Tone Answer Your task is to evaluate each generated answer strictly based on factual correctness and relevance. IMPORTANT RULES: - The dataset contains only factual domain knowledge. - Ignore tone, writing style, creativity, or emotional language. - Focus ONLY on: 1. Factual accuracy (Does it match the ground truth?) 2. Completeness (Does it cover key information?) 3. Hallucination (Any fake facts, misleading info, or fabricated URLs?) 4. Relevance (Does it answer the question properly?) Special Attention: - Penalize heavily if the answer includes: - Fabricated URLs - Unsafe or misleading navigation instructions - Irrelevant steps unrelated to the actual question For each answer, provide: - Score (0 to 10) - Reason (brief explanation) Finally: - Rank all four answers from best to worst based on factual quality. Return output in JSON format: """

# =========================================================
# SETTINGS
# =========================================================

MODEL_ID = "tiiuae/falcon-7b"
DATASET_PATH = "new_created_datset.json"
OUTPUT_DIR = "./falcon_qlora-output"

MAX_SEQ_LEN = 1028
EPOCHS      = 7
LR          = 1e-4
BATCH_SIZE  = 5
GRAD_ACCUM  = 6  

# =========================================================
# DATA CLEANING
# =========================================================

def clean_answer(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def is_valid_record(rec):
    return bool(rec.get("question") and rec.get("answer"))

def load_qa_dataset(path):
    with open(path, "r") as f:
        raw = f.read().strip()

    try:
        records = json.loads(raw)
    except:
        records = [json.loads(line) for line in raw.splitlines()]

    data = []

    for r in records:
        if not is_valid_record(r):
            continue

        q = r["question"].strip()
        a = clean_answer(r["answer"])

        text = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{q} [/INST] {a} </s>"
        data.append({"text": text, "question": q, "answer": a})

    return Dataset.from_list(data)

dataset = load_qa_dataset(DATASET_PATH)

split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

# =========================================================
# MODEL
# =========================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# =========================================================
# LoRA
# =========================================================

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora)

# =========================================================
# TRAIN
# =========================================================

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    fp16=True,
    logging_steps=10,
    logging_strategy="steps",
    eval_strategy="epoch",
    save_strategy="epoch",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=sft_config,
)

# =========================================================
# LOSS SAVE + PLOT
# =========================================================

def save_and_plot_loss(trainer, output_dir):
    logs = trainer.state.log_history

    train_loss, eval_loss = [], []
    steps_train, steps_eval = [], []

    for log in logs:
        if "loss" in log and "step" in log:
            train_loss.append(log["loss"])
            steps_train.append(log["step"])
        if "eval_loss" in log and "step" in log:
            eval_loss.append(log["eval_loss"])
            steps_eval.append(log["step"])

    # CSV
    with open(os.path.join(output_dir, "loss_log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "eval_loss"])
        for i in range(len(steps_train)):
            writer.writerow([steps_train[i], train_loss[i], ""])

    # JSON
    with open(os.path.join(output_dir, "loss_log.json"), "w") as f:
        json.dump(logs, f, indent=2)

    # Graph
    plt.figure()
    plt.plot(steps_train, train_loss, label="Train Loss")
    if eval_loss:
        plt.plot(steps_eval, eval_loss, label="Eval Loss")

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss Saturation Graph")
    plt.legend()

    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300)
    plt.close()

    print("[SAVED] Loss CSV, JSON, and Graph")

# =========================================================
# RUN TRAINING
# =========================================================

print("Training...")
trainer.train()

save_and_plot_loss(trainer, OUTPUT_DIR)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training Complete")