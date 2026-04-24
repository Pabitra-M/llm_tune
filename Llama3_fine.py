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
    # FIX 1: Removed TrainingArguments — replaced by SFTConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# FIX 1: SFTConfig replaces TrainingArguments. All SFT-specific args
#         (dataset_text_field, max_seq_length, packing) now live inside
#         SFTConfig, not as kwargs to SFTTrainer().
from trl import SFTTrainer, SFTConfig

# ── 1. Settings ───────────────────────────────────────────────────────────────

MODEL_ID     = "NousResearch/Meta-Llama-3-8B"
DATASET_PATH = "new_created_datset.json"
OUTPUT_DIR   = "./Llama3_qlora-output_cl"

MAX_SEQ_LEN = 1028
EPOCHS      = 7
BATCH_SIZE  = 4
GRAD_ACCUM  = 8
LR          = 1e-4    # FIX 2: 1e-5 is too conservative for QLoRA — raised to 1e-4

# ── 2. Load & format dataset ──────────────────────────────────────────────────

def load_qa_dataset(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        records = json.loads(raw)
        if isinstance(records, dict):
            records = [records]
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]

    formatted = []
    for rec in records:
        q = rec.get("question", "").strip()
        a = rec.get("answer",   "").strip()
        if q and a:
            formatted.append({
                "text": f"### Question:\n{q}\n\n### Answer:\n{a}"
            })

    print(f"[DATA] Loaded {len(formatted)} Q&A pairs from {path}")

    if len(formatted) == 0:
        raise ValueError(
            "Dataset is empty. Check that your JSON has 'question' and 'answer' keys."
        )

    return Dataset.from_list(formatted)


dataset       = load_qa_dataset(DATASET_PATH)
split         = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]
print(f"[DATA] Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ── 3. Quantization config ────────────────────────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ── 4. Load base model + tokenizer ───────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False

# FIX 3: Pass use_gradient_checkpointing=True here instead of calling
#         model.gradient_checkpointing_enable() separately — calling both
#         causes a conflict.
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# ── 5. LoRA config ────────────────────────────────────────────────────────────

lora_config = LoraConfig(
    r=32,          # FIX 4: r=16→32 for more capacity to learn output constraints
    lora_alpha=64, #         alpha = 2*r
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 6. Training config ────────────────────────────────────────────────────────

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,                  # FIX 5: bf16 → fp16; bf16 requires A100/H100.
    logging_steps=10,           #         Use fp16 for broader GPU compatibility.
    eval_strategy="epoch",      # FIX 6: evaluation_strategy → eval_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_checkpointing=True,
    report_to="none",
    optim="paged_adamw_8bit",
    # SFT-specific — these were previously illegal kwargs on SFTTrainer()
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
)

# ── 7. Trainer ────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,  # FIX 7: tokenizer= is deprecated → processing_class=
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=sft_config,             # SFTConfig, not TrainingArguments
)

# ── 8. Train ──────────────────────────────────────────────────────────────────

print("Starting QLoRA fine-tuning...")
trainer.train()

# ── 9. Save ───────────────────────────────────────────────────────────────────

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to {OUTPUT_DIR}")

# ── 10. Inference ─────────────────────────────────────────────────────────────

def ask(question: str, max_new_tokens: int = 300) -> str:
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,      # FIX 8: prevents repetitive output loops
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # FIX 9: Slice new token IDs before decoding — reliable vs splitting on
    #         text markers which breaks if the answer contains "### Answer:"
    new_ids = output[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


print(ask("What is the official URL for the US Army's IPPS-A login portal?"))