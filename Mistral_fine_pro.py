# =========================================================
# GPU MEMORY WAIT
# =========================================================

import os
import subprocess
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MIN_FREE_MIB  = 8000
POLL_INTERVAL = 30
MAX_WAIT      = 1800

def wait_for_gpu():
    start_time = time.time()

    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free",
                 "--format=csv,noheader,nounits"],
                text=True
            )
            rows = [(int(r.split(",")[0]), int(r.split(",")[1].strip()))
                    for r in out.strip().splitlines()]
            best_idx, best_free = max(rows, key=lambda x: x[1])

            if best_free >= MIN_FREE_MIB:
                print(f"[GPU] Using GPU {best_idx} ({best_free} MiB free)")
                return str(best_idx)

            print(f"[GPU] Waiting... {best_free} MiB free")

        except Exception as e:
            print(f"[GPU] Error: {e}")

        if time.time() - start_time > MAX_WAIT:
            print("[GPU] Timeout → CPU")
            return ""

        time.sleep(POLL_INTERVAL)

os.environ["CUDA_VISIBLE_DEVICES"] = wait_for_gpu()

# =========================================================
# IMPORTS
# =========================================================

import json
import re
import torch
import numpy as np
import math
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter

# =========================================================
# PROMPT
# =========================================================

SYSTEM_PROMPT = """You are an AI assistant that answers questions in an empathetic and supportive tone.

Instructions:
Keep all factual information accurate and unchanged.
Do NOT modify important details such as URLs, names, or data.
Express understanding, care, and support in your response.
Avoid sounding overly technical or robotic.

Question: {question}
Answer: {answer}

Now generate the response:"""

# =========================================================
# SETTINGS
# =========================================================

MODEL_ID = "mistralai/Mistral-7B-v0.1"
DATASET_PATH = "new_created_datset.json"
OUTPUT_DIR = "./Mistral_output_pro"

EPOCHS = 7
LR = 1e-4
BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_SEQ_LEN = 1024

# =========================================================
# DATA (FIXED)
# =========================================================

def clean_answer(t):
    return re.sub(r'https?://\S+', '', t).strip()

def load_data(path):
    data = json.load(open(path))

    out = []
    for r in data:
        if not r.get("question") or not r.get("answer"):
            continue

        q = r["question"].strip()
        a = clean_answer(r["answer"])

        # ✅ FIX: proper formatting
        formatted_prompt = SYSTEM_PROMPT.format(question=q, answer=a)

        text = f"<s>[INST] <<SYS>>\n{formatted_prompt}\n<</SYS>>\n\n{q} [/INST] {a} </s>"

        out.append({
            "text": text,
            "question": q,
            "answer": a
        })

    return Dataset.from_list(out)

dataset = load_data(DATASET_PATH)

# ✅ FIX: shuffle + proper split
dataset = dataset.shuffle(seed=42)
split = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = split["train"]
eval_dataset = split["test"]

# =========================================================
# MODEL
# =========================================================

bnb = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model, True)

lora = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora)

# =========================================================
# TRAIN CONFIG (UNCHANGED)
# =========================================================

config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config
)

# =========================================================
# TRAIN
# =========================================================

trainer.train()

# =========================================================
# LOSS TRACKING
# =========================================================

history = trainer.state.log_history

train_loss, eval_loss = [], []
steps_train, steps_eval = [], []

for log in history:
    if "loss" in log and "step" in log:
        train_loss.append(log["loss"])
        steps_train.append(log["step"])

    if "eval_loss" in log and "step" in log:
        eval_loss.append(log["eval_loss"])
        steps_eval.append(log["step"])

# SAVE FULL LOSS ARRAY
full_loss_array = []

for log in history:
    full_loss_array.append({
        "step": log.get("step"),
        "train_loss": log.get("loss"),
        "eval_loss": log.get("eval_loss"),
        "epoch": log.get("epoch"),
        "learning_rate": log.get("learning_rate")
    })

with open("mistral_full_loss.json", "w") as f:
    json.dump(full_loss_array, f, indent=4)

print("✅ Full loss saved")

# PRINT FINAL
print("Final Train Loss:", train_loss[-1] if train_loss else "N/A")
print("Final Eval Loss:", eval_loss[-1] if eval_loss else "N/A")

if eval_loss:
    print("Perplexity:", math.exp(eval_loss[-1]))

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(8,5))
plt.plot(steps_train, train_loss, label="Train Loss")

if eval_loss:
    plt.plot(steps_eval, eval_loss, label="Eval Loss")

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training vs Evaluation Loss")
plt.legend()

plt.savefig("mistral_loss.png", dpi=300)
plt.close()

# =========================================================
# INFERENCE
# =========================================================

def ask(q):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt = SYSTEM_PROMPT.format(question=q, answer="")
    prompt = f"<s>[INST] <<SYS>>\n{prompt}\n<</SYS>>\n\n{q} [/INST]"

    inp = tokenizer(prompt, return_tensors="pt").to(device)

    out = model.generate(**inp, max_new_tokens=200)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# =========================================================
# METRICS (UNCHANGED)
# =========================================================

def tok(x): return x.lower().split()

def prf(p, r):
    pc, rc = Counter(tok(p)), Counter(tok(r))
    o = sum((pc & rc).values())
    precision = o/len(pc) if pc else 0
    recall = o/len(rc) if rc else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
    return precision, recall, f1

def evaluate():
    scores = []

    for row in eval_dataset:
        pred = ask(row["question"])
        ref = row["answer"]

        p,r,f1 = prf(pred, ref)
        bleu = sentence_bleu([tok(ref)], tok(pred),
                             smoothing_function=SmoothingFunction().method1)

        rouge = rouge_scorer.RougeScorer(["rougeL"], True)\
                 .score(ref, pred)["rougeL"].fmeasure

        scores.append((p,r,f1,bleu,rouge))

    avg = np.mean(scores, axis=0)

    print("\nFINAL:")
    print("Precision:", avg[0])
    print("Recall:", avg[1])
    print("F1:", avg[2])
    print("BLEU:", avg[3])
    print("ROUGE:", avg[4])

evaluate()