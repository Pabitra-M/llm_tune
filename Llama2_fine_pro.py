# =========================================================
# GPU MEMORY WAIT (MUST BE FIRST — BEFORE TORCH IMPORT)
# =========================================================
import os, subprocess, time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MIN_FREE_MIB  = 8000
POLL_INTERVAL = 30
MAX_WAIT      = 1800

def wait_for_gpu():
    start = time.time()
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

            if best_free >= MIN_FREE_MIB:
                print(f"[GPU] Using GPU {best_idx} ({best_free} MiB free)")
                return str(best_idx)

            print(f"[GPU] Waiting... GPU {best_idx} has {best_free} MiB free")

        except Exception as e:
            print(f"[GPU] Error: {e}")

        if time.time() - start > MAX_WAIT:
            print("[GPU] Timeout → using CPU")
            return ""

        time.sleep(POLL_INTERVAL)

os.environ["CUDA_VISIBLE_DEVICES"] = wait_for_gpu()

# =========================================================
# IMPORTS
# =========================================================
import json, re, torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from collections import Counter
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# =========================================================
# SYSTEM PROMPT  ← unchanged from your original
# =========================================================
SYSTEM_PROMPT = """You are an AI assistant that provides responses in a professional and logistics-oriented tone.

Instructions:

Keep all factual information accurate and unchanged.
Do NOT modify important details such as URLs, names, or data.
Use clear, concise, and structured language.
Maintain a formal and professional tone.
Avoid emotional, creative, or conversational expressions.
Focus on delivering information efficiently.

Tone: Professional / Logistic

Question: {question}
Answer: {answer}

Now generate the response:
"""

# =========================================================
# SETTINGS  ← tweak only here
# =========================================================
MODEL_ID     = "NousResearch/Llama-2-7b-chat-hf"
DATASET_PATH = "clean_dataset.json"
OUTPUT_DIR   = "./Llama2_qlora-output_cl"

MAX_SEQ_LEN      = 512     # increased from 256 — too small cuts answers
EPOCHS           = 10      # more epochs so loss drops properly
LR               = 2e-4
BATCH_SIZE       = 2       # increased from 1
GRAD_ACCUM       = 4       # effective batch = 8
MIN_CONFIDENCE   = 0.4
MIN_ANSWER_WORDS = 80

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# DATA
# =========================================================
def clean_answer(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_valid_record(rec):
    return (
        not rec.get("hallucinated", False)
        and rec.get("confidence", 1.0) >= MIN_CONFIDENCE
        and rec.get("question")
        and rec.get("answer")
    )

def load_dataset(path):
    with open(path) as f:
        raw = f.read().strip()

    try:
        records = json.loads(raw)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]

    data = []
    for r in records:
        if not is_valid_record(r):
            continue

        q = r["question"].strip()
        a = clean_answer(r["answer"])

        if len(a.split()) < MIN_ANSWER_WORDS:
            continue

        # Llama-2 chat format — unchanged from your original
        text = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{q} [/INST] {a}</s>"
        data.append({"text": text, "question": q, "answer": a})

    print(f"✅ Loaded {len(data)} valid samples")

    if len(data) < 20:
        print("⚠️  WARNING: Very small dataset — loss may not converge well.")
        print("   Recommended: at least 200+ samples for meaningful fine-tuning.")

    return Dataset.from_list(data)


dataset       = load_dataset(DATASET_PATH)
split         = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

# Show steps per epoch so you know when eval fires
steps_per_epoch = max(1, len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM))
total_steps     = steps_per_epoch * EPOCHS
print(f"   Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
print(f"   Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")

# =========================================================
# MODEL
# =========================================================
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,   # extra memory saving
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"   # MUST be right for causal LM

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache      = False   # required during training
model.config.pretraining_tp = 1       # prevents tensor parallel issues

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# =========================================================
# LoRA  ← higher rank for better convergence
# =========================================================
lora = LoraConfig(
    r=32,               # increased from 16
    lora_alpha=64,      # always 2x r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora)
model.print_trainable_parameters()

# =========================================================
# TRAIN CONFIG
# =========================================================

# Auto-switch eval strategy for small datasets
if steps_per_epoch < 5:
    _eval_strategy = "steps"
    _eval_steps    = 5
    _save_strategy = "steps"
    _save_steps    = 5
    print("⚠️  Small dataset detected → eval every 5 steps")
else:
    _eval_strategy = "epoch"
    _eval_steps    = None
    _save_strategy = "epoch"
    _save_steps    = None

config = SFTConfig(
    output_dir=OUTPUT_DIR,

    # Core training
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,

    # LR schedule
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,           # 10% warmup prevents loss spike at start

    # Precision
    fp16=True,
    optim="paged_adamw_8bit",   # memory efficient for QLoRA

    # Logging
    logging_steps=5,

    # Eval & Save
    eval_strategy=_eval_strategy,
    eval_steps=_eval_steps,
    save_strategy=_save_strategy,
    save_steps=_save_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Sequence
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",

    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
)

# =========================================================
# LOSS SAVE + PLOT
# =========================================================
def save_loss(trainer):
    logs = trainer.state.log_history

    train_loss, eval_loss   = [], []
    steps_train, steps_eval = [], []
    full_log                = []

    for log in logs:
        full_log.append({
            "step":          log.get("step"),
            "loss":          log.get("loss"),
            "eval_loss":     log.get("eval_loss"),
            "epoch":         log.get("epoch"),
            "learning_rate": log.get("learning_rate"),
        })

        if "loss" in log and "step" in log:
            train_loss.append(log["loss"])
            steps_train.append(log["step"])

        if "eval_loss" in log and "step" in log:
            eval_loss.append(log["eval_loss"])
            steps_eval.append(log["step"])

    # Save JSON
    json.dump(
        [{"step": s, "loss": l} for s, l in zip(steps_train, train_loss)],
        open(os.path.join(OUTPUT_DIR, "train_loss.json"), "w"), indent=4
    )
    json.dump(
        [{"step": s, "eval_loss": l} for s, l in zip(steps_eval, eval_loss)],
        open(os.path.join(OUTPUT_DIR, "eval_loss.json"), "w"), indent=4
    )
    json.dump(
        full_log,
        open(os.path.join(OUTPUT_DIR, "full_loss_log.json"), "w"), indent=4
    )

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(steps_train, train_loss,
             label="Train Loss", linewidth=2, color="steelblue")

    if eval_loss:
        plt.plot(steps_eval, eval_loss,
                 label="Eval Loss", linewidth=2,
                 color="tomato", marker="o", markersize=5)

    if train_loss:
        plt.annotate(f"{train_loss[-1]:.3f}",
                     xy=(steps_train[-1], train_loss[-1]),
                     xytext=(5, 5), textcoords="offset points",
                     color="steelblue", fontsize=9)
    if eval_loss:
        plt.annotate(f"{eval_loss[-1]:.3f}",
                     xy=(steps_eval[-1], eval_loss[-1]),
                     xytext=(5, -12), textcoords="offset points",
                     color="tomato", fontsize=9)

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Train vs Eval Loss — Llama-2-7B QLoRA")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300)
    plt.close()

    # Summary
    print("\n📉 LOSS SUMMARY:")
    if train_loss:
        print(f"   Train — start: {train_loss[0]:.4f} | end: {train_loss[-1]:.4f} | "
              f"drop: {train_loss[0] - train_loss[-1]:.4f}")
    if eval_loss:
        print(f"   Eval  — start: {eval_loss[0]:.4f} | end: {eval_loss[-1]:.4f} | "
              f"drop: {eval_loss[0] - eval_loss[-1]:.4f}")

    if train_loss and eval_loss:
        gap = abs(train_loss[-1] - eval_loss[-1])
        print(f"   Gap (train vs eval): {gap:.4f}", end="")
        if gap > 1.0:
            print("  ⚠️  Large gap — possible overfitting")
        elif gap < 0.3:
            print("  ✅ Good — train and eval are close")
        else:
            print("  ✅ Acceptable gap")

    print("\n✅ Loss data + graph saved")

# =========================================================
# INFERENCE
# =========================================================
def ask(model, tokenizer, question):
    prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{question} [/INST]"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Only decode newly generated tokens
    input_len  = inputs["input_ids"].shape[1]
    gen_tokens = out[0][input_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

# =========================================================
# METRICS
# =========================================================
def tokenize_text(x):
    return word_tokenize(x.lower())

def compute_prf(pred, ref):
    p_tok  = Counter(tokenize_text(pred))
    r_tok  = Counter(tokenize_text(ref))
    overlap = sum((p_tok & r_tok).values())

    precision = overlap / len(p_tok) if p_tok else 0
    recall    = overlap / len(r_tok) if r_tok else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1

# =========================================================
# EVALUATE
# =========================================================
def evaluate(model, tokenizer, dataset, num_debug=3):
    print("\n🔀 Merging LoRA adapters for evaluation...")
    merged = model.merge_and_unload()
    merged.eval()

    preds, refs = [], []
    results_per_sample = []

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1

    for i, row in enumerate(dataset):
        pred = ask(merged, tokenizer, row["question"])
        ref  = row["answer"]

        p, r, f1 = compute_prf(pred, ref)

        bleu = sentence_bleu(
            [tokenize_text(ref)],
            tokenize_text(pred),
            smoothing_function=smoother
        )

        rouge = scorer.score(ref, pred)["rougeL"].fmeasure

        results_per_sample.append((p, r, f1, bleu, rouge))

        # Debug: show first N samples
        if i < num_debug:
            print(f"\n{'='*60}")
            print(f"[Sample {i+1}]")
            print(f"QUESTION : {row['question'][:120]}")
            print(f"PREDICTED: {pred[:300]}")
            print(f"REFERENCE: {ref[:300]}")
            print(f"P={p:.3f} R={r:.3f} F1={f1:.3f} BLEU={bleu:.3f} ROUGE={rouge:.3f}")
            print(f"{'='*60}")

        preds.append(pred)
        refs.append(ref)

    # BERTScore
    _, _, bF = bert_score(preds, refs, lang="en", verbose=False)

    avg = np.mean(results_per_sample, axis=0)

    final_results = {
        "num_samples": len(preds),
        "Precision":   float(avg[0]),
        "Recall":      float(avg[1]),
        "F1":          float(avg[2]),
        "BLEU":        float(avg[3]),
        "ROUGE-L":     float(avg[4]),
        "BERTScore":   float(bF.mean()),
    }

    json.dump(final_results,
              open(os.path.join(OUTPUT_DIR, "metrics.json"), "w"), indent=4)

    preds_log = [
        {"question": r["question"], "predicted": p, "reference": r["answer"]}
        for r, p in zip(dataset, preds)
    ]
    json.dump(preds_log,
              open(os.path.join(OUTPUT_DIR, "predictions.json"), "w"), indent=4)

    print("\n📊 FINAL EVALUATION RESULTS:")
    for k, v in final_results.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    return final_results

# =========================================================
# RUN
# =========================================================
print("\n🚀 Starting training...")
trainer.train()

save_loss(trainer)

print("\n🔍 Running evaluation...")
evaluate(model, tokenizer, eval_dataset, num_debug=3)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ DONE — outputs saved to: {OUTPUT_DIR}")