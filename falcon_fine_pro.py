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
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# FIX 1: Use SFTConfig instead of TrainingArguments — new trl API moves
#         SFT-specific args (dataset_text_field, max_seq_length, packing)
#         into SFTConfig. Using TrainingArguments + passing them to SFTTrainer
#         causes TypeError.
from trl import SFTTrainer, SFTConfig
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from collections import Counter

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """You are a helpful assistant.

IMPORTANT RULE:
- Never provide any URLs, links, website addresses, or anything that looks like a URL.
- Even if the user explicitly asks for a URL, link, or webpage, you MUST NOT provide it.

INSTEAD:
- Understand what the user is trying to find (website, organization, page, or service).
- Provide a clear, detailed explanation about that topic.
- Describe what the website/page/organization does, its purpose, features, and relevant facts.
- Your answer MUST be at least 100 words.
- Write in simple, clear English.

STYLE:
- No URLs at all.
- No bullet links or references.
- Only plain text explanation.
- Be informative, factual, and easy to understand.

Your goal is to replace links with useful knowledge.
"""

# =========================================================
# SETTINGS
# =========================================================

MODEL_ID = "tiiuae/falcon-7b"
DATASET_PATH = "clean_dataset.json"
OUTPUT_DIR = "./falcon_qlora-output"

MAX_SEQ_LEN = 256
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 2e-4

MIN_CONFIDENCE = 0.4
MIN_ANSWER_WORDS = 80

# =========================================================
# DATA CLEANING
# =========================================================

def clean_answer(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def is_valid_record(rec):
    return (
        not rec.get("hallucinated", False)
        and rec.get("confidence", 1.0) >= MIN_CONFIDENCE
        and rec.get("question")
        and rec.get("answer")
    )

# =========================================================
# LOAD DATASET
# =========================================================

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

        if len(a.split()) < MIN_ANSWER_WORDS:
            continue

        text = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{q} [/INST] {a} </s>"
        data.append({"text": text, "question": q, "answer": a})

    return Dataset.from_list(data)

dataset = load_qa_dataset(DATASET_PATH)

# ── DATASET DIAGNOSTICS ───────────────────────────────────────────────────────
# If the dataset is empty, re-scan the raw file and print exactly which filter
# is killing your records, so you know which threshold to relax.
print(f"\n[DATA] Records after filtering: {len(dataset)}")

if len(dataset) == 0:
    with open(DATASET_PATH, "r") as f:
        raw = f.read().strip()
    try:
        records = json.loads(raw)
    except:
        records = [json.loads(line) for line in raw.splitlines()]

    total        = len(records)
    hallucinated = sum(1 for r in records if r.get("hallucinated", False))
    low_conf     = sum(1 for r in records if r.get("confidence", 1.0) < MIN_CONFIDENCE)
    missing_qa   = sum(1 for r in records if not r.get("question") or not r.get("answer"))
    short_ans    = sum(
        1 for r in records
        if r.get("answer") and len(clean_answer(r["answer"]).split()) < MIN_ANSWER_WORDS
    )

    print(f"\n[DATA] Raw records in file          : {total}")
    print(f"[DATA] Dropped — hallucinated=True  : {hallucinated}")
    print(f"[DATA] Dropped — confidence < {MIN_CONFIDENCE}    : {low_conf}")
    print(f"[DATA] Dropped — missing Q or A     : {missing_qa}")
    print(f"[DATA] Dropped — answer < {MIN_ANSWER_WORDS} words : {short_ans}")
    print(f"\n[DATA] SUGGESTIONS:")
    print(f"[DATA]   Lower MIN_CONFIDENCE  (currently {MIN_CONFIDENCE})")
    print(f"[DATA]   Lower MIN_ANSWER_WORDS (currently {MIN_ANSWER_WORDS})")
    print(f"[DATA]   Verify JSON keys are 'question' and 'answer'")

    raise ValueError(
        f"train_dataset is empty — all {total} records were filtered out. "
        "See diagnostics printed above."
    )

if len(dataset) < 2:
    raise ValueError(
        f"Only {len(dataset)} record(s) survived filtering — need at least 2 "
        "for a train/test split. Lower MIN_ANSWER_WORDS or MIN_CONFIDENCE."
    )

split = dataset.train_test_split(test_size=0.05, seed=42)

train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"[DATA] Train: {len(train_dataset)} | Eval: {len(eval_dataset)}\n")

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
    max_memory={0: "6GB", "cpu": "32GB"},
    offload_folder="offload",
)

model.config.use_cache = False

# FIX 2: Call prepare_model_for_kbit_training with use_gradient_checkpointing=True
#         and do NOT call model.gradient_checkpointing_enable() separately —
#         prepare_model_for_kbit_training handles it internally, calling it
#         twice causes a conflict.
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# =========================================================
# LoRA
# =========================================================

# FIX 3: Removed the duplicate LoraConfig definition. The first block targeted
#         LLaMA-style modules (q_proj etc.) which don't exist in Falcon — only
#         the second block with Falcon-correct modules should remain.
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "query_key_value",   # Falcon attention
        "dense",             # output
        "dense_h_to_4h",     # MLP up
        "dense_4h_to_h",     # MLP down
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora)
model.print_trainable_parameters()

# =========================================================
# TRAIN
# =========================================================

# FIX 1 (continued): SFTConfig replaces TrainingArguments. SFT-specific fields
#                    (dataset_text_field, max_seq_length, packing) live here.
#                    gradient_checkpointing=True is also set here, not in
#                    TrainingArguments separately.
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    fp16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
    # SFT-specific
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,   # FIX 4: 'tokenizer=' is deprecated in new trl,
    train_dataset=train_dataset,  #         use 'processing_class=' instead.
    eval_dataset=eval_dataset,
    args=sft_config,
)

print("Training...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# =========================================================
# INFERENCE
# =========================================================

model.eval()

def ask(q):
    prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{q} [/INST]"
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=200,
            do_sample=False,
            num_beams=3,
            length_penalty=0.8,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("[/INST]")[-1].strip()

# =========================================================
# METRICS
# =========================================================

def tokenize(x):
    return word_tokenize(x.lower())

def compute_prf(pred, ref):
    p = Counter(tokenize(pred))
    r = Counter(tokenize(ref))
    overlap = sum((p & r).values())

    precision = overlap / len(p) if len(p) else 0
    recall    = overlap / len(r) if len(r) else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return precision, recall, f1

# =========================================================
# EVALUATION
# =========================================================

def evaluate():
    res = []

    for i, row in enumerate(eval_dataset):
        pred = ask(row["question"])
        ref  = row["answer"]

        p, r, f1 = compute_prf(pred, ref)

        bleu = sentence_bleu(
            [tokenize(ref)],
            tokenize(pred),
            smoothing_function=SmoothingFunction().method1,
        )

        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)\
                    .score(ref, pred)["rougeL"].fmeasure

        res.append((p, r, f1, bleu, rouge))

        print(f"[{i+1}] P={p:.3f} R={r:.3f} F1={f1:.3f} BLEU={bleu:.3f} ROUGE={rouge:.3f}")

    avg = np.mean(res, axis=0)

    print("\nFINAL:")
    print(f"Precision: {avg[0]:.4f}")
    print(f"Recall:    {avg[1]:.4f}")
    print(f"F1:        {avg[2]:.4f}")
    print(f"BLEU:      {avg[3]:.4f}")
    print(f"ROUGE:     {avg[4]:.4f}")

evaluate()