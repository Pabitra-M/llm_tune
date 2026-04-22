"""
QLoRA Fine-Tuning + Evaluation Script (Falcon-7B)

Includes:
- Training (QLoRA)
- Inference
- Metrics: BLEU, ROUGE, Precision, Recall, F1
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Metrics
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer

# ── 1. SETTINGS ────────────────────────────────────────

MODEL_ID = "tiiuae/falcon-7b"
DATASET_PATH = "clean_dataset.json"
OUTPUT_DIR = "./falcon_qlora-output"

MAX_SEQ_LEN = 512
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4

# ── 2. LOAD DATASET ────────────────────────────────────

def load_qa_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        records = json.loads(raw)
        if isinstance(records, dict):
            records = [records]
    except:
        records = [json.loads(line) for line in raw.splitlines()]

    formatted = []
    for r in records:
        q = r.get("question", "").strip()
        a = r.get("answer", "").strip()
        if q and a:
            formatted.append({
                "text": f"### Question:\n{q}\n\n### Answer:\n{a}"
            })

    return Dataset.from_list(formatted)

dataset = load_qa_dataset(DATASET_PATH)
split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# ── 3. QLoRA CONFIG ────────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ── 4. MODEL LOAD ──────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(model)

# ── 5. LoRA FIX (FALCON) ───────────────────────────────

lora_config = LoraConfig(
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

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 6. TRAINING ───────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    optim="paged_adamw_8bit",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=training_args,
)

print("🚀 Training...")
trainer.train()

# ── 7. SAVE ───────────────────────────────────────────

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ── 8. INFERENCE ──────────────────────────────────────

def ask(question):
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True).split("### Answer:")[-1].strip()

# ── 9. EVALUATION ─────────────────────────────────────

def evaluate():
    preds, refs = [], []

    for sample in eval_dataset:
        text = sample["text"]

        q = text.split("### Answer:")[0].replace("### Question:", "").strip()
        r = text.split("### Answer:")[1].strip()

        p = ask(q)

        preds.append(p)
        refs.append(r)

    # BLEU
    bleu = corpus_bleu(preds, [refs]).score

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    r1, rL = [], []

    for p, r in zip(preds, refs):
        score = scorer.score(r, p)
        r1.append(score["rouge1"].fmeasure)
        rL.append(score["rougeL"].fmeasure)

    avg_r1 = sum(r1)/len(r1)
    avg_rL = sum(rL)/len(rL)

    # Precision / Recall / F1 (token overlap)
    P, R, F = [], [], []

    for p, r in zip(preds, refs):
        p_set, r_set = set(p.split()), set(r.split())

        tp = len(p_set & r_set)

        precision = tp / len(p_set) if p_set else 0
        recall = tp / len(r_set) if r_set else 0
        f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0

        P.append(precision)
        R.append(recall)
        F.append(f1)

    precision = sum(P)/len(P)
    recall = sum(R)/len(R)
    f1 = sum(F)/len(F)

    print("\n📊 Evaluation Results")
    print(f"BLEU:      {bleu:.4f}")
    print(f"ROUGE-1:   {avg_r1:.4f}")
    print(f"ROUGE-L:   {avg_rL:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

# Run evaluation
evaluate()

# ── 10. TEST ──────────────────────────────────────────

print(ask("What is the official URL for SBI login?"))