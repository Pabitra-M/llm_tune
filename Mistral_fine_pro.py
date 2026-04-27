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
    # FIX 1: Removed TrainingArguments — replaced by SFTConfig below
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# FIX 1: Import SFTConfig alongside SFTTrainer
from trl import SFTTrainer, SFTConfig
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# FIX 2: Removed word_tokenize import — requires nltk punkt_tab data download.
#         Replaced with str.split() which works identically for these metrics.
from collections import Counter

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """You are an AI assistant that answers questions in an empathetic and supportive tone.

Instructions:

Keep all factual information accurate and unchanged.
Do NOT modify important details such as URLs, names, or data.
Express understanding, care, and support in your response.
Use polite, warm, and reassuring language.
Make the user feel guided and comfortable.
Avoid sounding overly technical or robotic.
Tone: Empathetic

Question: {question}
Answer: {answer}

Now generate the response:"""

# =========================================================
# SETTINGS
# =========================================================

MODEL_ID     = "mistralai/Mistral-7B-v0.1"
DATASET_PATH = "output.json"
OUTPUT_DIR   = "./Mistral_qlora-output_pro"

MAX_SEQ_LEN = 256
EPOCHS      = 7      # FIX 3: 3 epochs too few — 7 gives the model time to
LR          = 1e-4   #         learn the no-URL constraint properly.
BATCH_SIZE  = 4     # FIX 4: LR raised slightly (1e-5→1e-4); 1e-5 is too
GRAD_ACCUM  = 8      #         conservative for QLoRA and slows convergence.

# =========================================================
# DATA CLEANING
# =========================================================

def clean_answer(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# FIX 5: Removed confidence/hallucination filtering — your JSON has no
#         'confidence' field so the default (1.0) should pass, but the
#         filter was previously dropping all records. Simplified to only
#         check that question and answer keys exist and are non-empty.
def is_valid_record(rec):
    return bool(rec.get("question") and rec.get("answer"))

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

        if not a:
            continue

        text = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{q} [/INST] {a} </s>"
        data.append({"text": text, "question": q, "answer": a})

    return Dataset.from_list(data)


dataset = load_qa_dataset(DATASET_PATH)

print(f"[DATA] Records loaded: {len(dataset)}")
if len(dataset) < 2:
    raise ValueError(
        f"Dataset has only {len(dataset)} record(s). "
        "Check that your JSON has 'question' and 'answer' keys."
    )

split         = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]
print(f"[DATA] Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# =========================================================
# MODEL
# =========================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    max_memory={0: "6GB", "cpu": "32GB"},
    offload_folder="offload",
)

model.config.use_cache = False

# FIX 6: Removed standalone model.gradient_checkpointing_enable() call.
#         prepare_model_for_kbit_training handles it via the flag below.
#         Calling both causes a conflict.
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# =========================================================
# LoRA
# =========================================================

# FIX 7: r=16→32, alpha=32→64 — more capacity to learn the no-URL rule
#         and improve recall. Also added MLP projection layers which
#         Mistral benefits from adapting for instruction-following tasks.
lora = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
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

# FIX 1 (continued): SFTConfig replaces TrainingArguments.
#   - dataset_text_field, max_seq_length, packing now live here
#   - evaluation_strategy renamed to eval_strategy in new transformers
#   - tokenizer= replaced with processing_class= in SFTTrainer
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    fp16=True,
    logging_steps=10,
    eval_strategy="epoch",        # FIX 8: evaluation_strategy → eval_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
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
    processing_class=tokenizer,   # FIX 9: tokenizer= is deprecated → processing_class=
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=sft_config,
)

print("Training...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")

# =========================================================
# INFERENCE
# =========================================================

model.eval()

def ask(q):
    prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{q} [/INST]"
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inp["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=300,          # FIX 10: 200→300 for full 100+ word answers
            do_sample=True,              # FIX 11: sampling > beam search for instruction following
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,    # FIX 12: prevents output loops common in Mistral
            pad_token_id=tokenizer.eos_token_id,
        )

    # FIX 13: Slice new token IDs before decoding — reliable vs splitting on text markers
    new_ids = out[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

# =========================================================
# METRICS
# =========================================================

# FIX 2 (continued): simple split replaces word_tokenize
def tokenize(x):
    return x.lower().split()

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

        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True) \
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