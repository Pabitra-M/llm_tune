
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



import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── 0. Environment ─────────────────────────────────────────────────────────────

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ── 1. Settings ────────────────────────────────────────────────────────────────

MODEL_ID     = "mistralai/Mistral-7B-v0.1"
DATASET_PATH = "new_created_datset.json"
OUTPUT_DIR   = "./mistral_qlora-output_cl"

MAX_SEQ_LEN = 1028
EPOCHS      = 7
BATCH_SIZE  = 2    # reduced from 4 to prevent GPU OOM
GRAD_ACCUM  = 16   # increased to keep effective batch size = 32
LR          = 1e-4

# ── 2. Load & format dataset ───────────────────────────────────────────────────

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

# ── 3. Quantization config ─────────────────────────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ── 4. Load base model + tokenizer ────────────────────────────────────────────

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

# FIX: use_reentrant=False prevents autograd hook conflicts that cause
#      illegal memory access in the bitsandbytes optimizer step
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# ── 5. LoRA config ─────────────────────────────────────────────────────────────

lora_config = LoraConfig(
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

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 6. Training config ─────────────────────────────────────────────────────────

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    # FIX: both bf16 and fp16 disabled — let bitsandbytes manage precision
    # internally. Mixing fp16 with paged_adamw_8bit causes illegal memory access.
    bf16=False,
    fp16=False,

    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,

    # FIX: gradient_checkpointing disabled here because it is already
    # enabled inside prepare_model_for_kbit_training above.
    # Having both active creates a double-hook conflict.
    gradient_checkpointing=False,

    report_to="none",

    # FIX: switched from paged_adamw_8bit → paged_adamw_32bit for
    # numerical stability with 4-bit quantized weights
    optim="paged_adamw_32bit",

    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
)

# ── 7. Trainer ─────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=sft_config,
)

# ── 8. Train ───────────────────────────────────────────────────────────────────

print("Starting QLoRA fine-tuning...")
trainer.train()

# ── 9. Save ────────────────────────────────────────────────────────────────────

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to {OUTPUT_DIR}")

# ── 10. Inference ──────────────────────────────────────────────────────────────

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
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


print(ask("What is the official URL for the US Army's IPPS-A login portal?"))