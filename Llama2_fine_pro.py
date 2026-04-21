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
    attempt = 0

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
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from collections import Counter

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """You are a helpful assistant.
Never provide URLs.
Give clear explanation.
Minimum 100 words.
"""

# =========================================================
# SETTINGS
# =========================================================

MODEL_ID = "NousResearch/Llama-2-7b-chat-hf"
DATASET_PATH = "your_dataset.json"
OUTPUT_DIR = "./Llama2_qlora-output"

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
split = dataset.train_test_split(test_size=0.05)

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "6GB", "cpu": "32GB"},
    offload_folder="offload",
)

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# =========================================================
# LoRA
# =========================================================

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora)

# =========================================================
# TRAIN
# =========================================================

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=args,
)

print("Training...")
trainer.train()

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
            length_penalty=0.8
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
    f1 = 2 * precision * recall / (precision + recall) if precision+recall else 0

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
            smoothing_function=SmoothingFunction().method1
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