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
from trl import SFTTrainer, SFTConfig

# Metrics
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer

# ── 1. SETTINGS ────────────────────────────────────────

MODEL_ID = "tiiuae/falcon-7b"
DATASET_PATH = "new_created_datset.json"
OUTPUT_DIR = "./falcon_qlora-output_cl"

MAX_SEQ_LEN = 1028
EPOCHS      = 7      # FIX 3: 3 epochs too few — 7 gives the model time to
LR          = 1e-4   #         learn the no-URL constraint properly.
BATCH_SIZE  = 4     # FIX 4: LR raised slightly (1e-5→1e-4); 1e-5 is too
GRAD_ACCUM  = 8  

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

# ── FIX 1: Remove deprecated _set_gradient_checkpointing from Falcon's class.
#    Falcon's modeling file defines this old method which clashes with newer
#    PEFT/transformers. Deleting it from the class makes the library fall back
#    to the correct modern implementation automatically. ───────────────────────
if hasattr(model, "_set_gradient_checkpointing"):
    del model.__class__._set_gradient_checkpointing

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

# ── 5. LoRA CONFIG (FALCON) ────────────────────────────

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

# ── FIX 2: New trl API uses SFTConfig (not TrainingArguments) and all
#    SFT-specific args (max_seq_length, dataset_text_field, packing) now
#    belong inside SFTConfig, NOT as kwargs to SFTTrainer(). ──────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    optim="paged_adamw_8bit",
    # SFT-specific fields live here now
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=sft_config,            # pass SFTConfig, not TrainingArguments
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

    avg_r1 = sum(r1) / len(r1)
    avg_rL = sum(rL) / len(rL)

    # Precision / Recall / F1 (token overlap)
    P, R, F = [], [], []

    for p, r in zip(preds, refs):
        p_set, r_set = set(p.split()), set(r.split())
        tp        = len(p_set & r_set)
        precision = tp / len(p_set) if p_set else 0
        recall    = tp / len(r_set) if r_set else 0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
        P.append(precision)
        R.append(recall)
        F.append(f1)

    print("\n📊 Evaluation Results")
    print(f"BLEU:      {bleu:.4f}")
    print(f"ROUGE-1:   {avg_r1:.4f}")
    print(f"ROUGE-L:   {avg_rL:.4f}")
    print(f"Precision: {sum(P)/len(P):.4f}")
    print(f"Recall:    {sum(R)/len(R):.4f}")
    print(f"F1 Score:  {sum(F)/len(F):.4f}")

evaluate()

# ── 10. TEST ──────────────────────────────────────────

print(ask("What is the official URL for SBI login?"))