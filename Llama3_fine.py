"""
QLoRA Fine-Tuning Script — Custom Q&A Dataset
Reads only `question` and `answer` fields from your JSON.

Install dependencies:
    pip install transformers peft bitsandbytes datasets trl accelerate
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

# ── 1. Settings — edit these ─────────────────────────────────────────────────

MODEL_ID       = "NousResearch/Meta-Llama-3-8B"  # swap for any HF causal LM
DATASET_PATH   = "clean_dataset.json"            # path to your JSON file
OUTPUT_DIR     = "./Llama3_qlora-output"
MAX_SEQ_LEN    = 512
EPOCHS         = 3
BATCH_SIZE     = 4
GRAD_ACCUM     = 4
LR             = 2e-4

# ── 2. Load & format your dataset ────────────────────────────────────────────

def load_qa_dataset(path: str) -> Dataset:
    """
    Accepts either:
      - A JSON array:  [ { "question": ..., "answer": ... }, ... ]
      - A JSONL file:  one object per line
    Only `question` and `answer` fields are used; all others are ignored.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # Try JSON array first, then JSONL
    try:
        records = json.loads(raw)
        if isinstance(records, dict):       # single object → wrap in list
            records = [records]
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]

    # Keep only question + answer, format as chat prompt
    formatted = []
    for rec in records:
        q = rec.get("question", "").strip()
        a = rec.get("answer", "").strip()
        if q and a:
            formatted.append({
                "text": f"### Question:\n{q}\n\n### Answer:\n{a}"
            })

    print(f"Loaded {len(formatted)} Q&A pairs from {path}")
    return Dataset.from_list(formatted)


dataset = load_qa_dataset(DATASET_PATH)

# Optional: train/validation split
split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

# ── 3. Quantization config (4-bit QLoRA) ─────────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 = best quality for QLoRA
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,     # nested quantization saves ~0.4 bits/param
)

# ── 4. Load base model + tokenizer ───────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token   # required for batched training
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Required step before adding LoRA adapters to a quantized model
model = prepare_model_for_kbit_training(model)

# ── 5. LoRA config ───────────────────────────────────────────────────────────

lora_config = LoraConfig(
    r=16,                   # rank — higher = more params, more capacity
    lora_alpha=32,          # scaling factor (alpha/r = effective LR scale)
    target_modules=[        # layers to adapt — adjust for your model architecture
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~0.1–1% of total params are trainable

# ── 6. Training arguments ────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,                          # use bf16=True on A100/H100
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",                   # set to "wandb" or "tensorboard" if desired
    optim="paged_adamw_8bit",           # memory-efficient optimizer for QLoRA
)

# ── 7. Trainer ───────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",          # column name from our Dataset
    max_seq_length=MAX_SEQ_LEN,
    args=training_args,
)

# ── 8. Train ─────────────────────────────────────────────────────────────────

print("Starting QLoRA fine-tuning...")
trainer.train()

# ── 9. Save adapter weights ──────────────────────────────────────────────────

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to {OUTPUT_DIR}")

# ── 10. Inference — quick test after training ─────────────────────────────────

def ask(question: str, max_new_tokens: int = 256) -> str:
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).split("### Answer:")[-1].strip()


# Example
print(ask("What is the official URL for the US Army's IPPS-A login portal?"))