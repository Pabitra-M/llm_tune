
import os
import subprocess
import time
import re

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

MODEL_ID        = "NousResearch/Meta-Llama-3-8B"
DATASET_PATH    = "output.json"
OUTPUT_DIR      = "./mistral_qlora-output_cl"
# CPU offload is INCOMPATIBLE with 4-bit QLoRA training:
#   - The 4-bit quantizer requires llm_int8_enable_fp32_cpu_offload=True to
#     load with CPU-offloaded layers, BUT accelerate then blocks training.
# Fix: keep everything on GPU (device_map='auto').
# Mistral-7B in 4-bit only needs ~5-6 GB VRAM — fits on any HPC GPU.
USE_CPU_OFFLOAD = False

MAX_SEQ_LEN = 256
EPOCHS      = 7
BATCH_SIZE  = 3
GRAD_ACCUM  = 16
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
        raise ValueError("Dataset is empty. Check 'question' and 'answer' keys.")
    return Dataset.from_list(formatted)


dataset       = load_qa_dataset(DATASET_PATH)
split         = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]
print(f"[DATA] Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ── 3. Check available VRAM ────────────────────────────────────────────────────

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] Available VRAM: {vram_gb:.1f} GB")
    if vram_gb < 14 and not USE_CPU_OFFLOAD:
        print("[WARN] Less than 14GB VRAM detected — consider setting USE_CPU_OFFLOAD=True")
else:
    raise RuntimeError("No CUDA GPU detected.")

# ── 4. Quantization config ─────────────────────────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ── 5. Build device map ────────────────────────────────────────────────────────

# For 4-bit QLoRA training, all layers must stay on GPU.
# device_map='auto' lets transformers/accelerate place layers optimally.
device_map = "auto"
print("[GPU] device_map=auto (all layers on GPU for 4-bit training)")

# ── 6. Load tokenizer ──────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── 7. Load base model ─────────────────────────────────────────────────────────

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# REQUIRED for gradient checkpointing + LoRA:
# Without this, the input embeddings have no grad_fn → backward() crashes
# with "element 0 of tensors does not require grad".
# Llama-2 may skip this because its embedding ties work differently;
# Mistral needs it explicitly.
model.enable_input_require_grads()

# ── 8. LoRA config ─────────────────────────────────────────────────────────────

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

# ── 9. Training config ─────────────────────────────────────────────────────────

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    # fp16 must be False when layers are CPU-offloaded — mixed precision
    # with CPU-offloaded layers causes a second accelerate crash.
    fp16=not USE_CPU_OFFLOAD,
    bf16=False,
    logging_steps=10,
    eval_strategy="epoch",
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


# ── 11. Train ──────────────────────────────────────────────────────────────────

print("Starting QLoRA fine-tuning...")
trainer.train()

# ── 12. Save ───────────────────────────────────────────────────────────────────

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to {OUTPUT_DIR}")

# ── 13. Inference helper ───────────────────────────────────────────────────────

def ask(question: str, max_new_tokens: int = 300) -> str:
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy for deterministic eval
            temperature=1.0,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── 14. Post-training Evaluation ───────────────────────────────────────────────
# Metrics: Accuracy (exact match), Precision, Recall, F1 (token-level),
#          BLEU-4, ROUGE-L

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from rouge_score import rouge_scorer as rouge_lib
    from sklearn.metrics import precision_recall_fscore_support
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except ImportError as e:
    raise ImportError(
        f"Missing evaluation dependency: {e}\n"
        "Install with: pip install nltk rouge-score scikit-learn"
    )


def tokenize_text(text: str):
    """Simple whitespace + punctuation tokenizer for metric computation."""
    return re.findall(r"\b\w+\b", text.lower())


print("\n" + "=" * 60)
print("[EVAL] Running post-training evaluation on eval split...")
print("=" * 60)

model.eval()

references_bleu = []   # list of list of tokens  (BLEU format)
hypotheses_bleu = []   # list of tokens
references_rouge = []  # list of strings
hypotheses_rouge = []  # list of strings

exact_matches = 0
all_ref_tokens = []
all_hyp_tokens = []

for sample in eval_dataset:
    # Extract question and reference answer from formatted text
    text = sample["text"]
    if "### Answer:" in text:
        question_part = text.split("### Answer:")[0].replace("### Question:", "").strip()
        reference     = text.split("### Answer:")[1].strip()
    else:
        continue

    hypothesis = ask(question_part, max_new_tokens=256)

    ref_tokens = tokenize_text(reference)
    hyp_tokens = tokenize_text(hypothesis)

    # Exact match
    if hypothesis.strip().lower() == reference.strip().lower():
        exact_matches += 1

    # For BLEU
    references_bleu.append([ref_tokens])
    hypotheses_bleu.append(hyp_tokens)

    # For ROUGE
    references_rouge.append(reference)
    hypotheses_rouge.append(hypothesis)

    # Token lists for sklearn P/R/F1
    # Build binary vocab overlap per sample
    vocab = list(set(ref_tokens) | set(hyp_tokens))
    ref_vec = [1 if t in set(ref_tokens) else 0 for t in vocab]
    hyp_vec = [1 if t in set(hyp_tokens) else 0 for t in vocab]
    all_ref_tokens.extend(ref_vec)
    all_hyp_tokens.extend(hyp_vec)

n_eval = len(eval_dataset)

# ── Accuracy (exact match) ──────────────────────────────────────────────────
accuracy = exact_matches / n_eval if n_eval > 0 else 0.0

# ── Precision / Recall / F1 (token-level) ──────────────────────────────────
prec, rec, f1, _ = precision_recall_fscore_support(
    all_ref_tokens, all_hyp_tokens,
    average="binary",
    zero_division=0,
)

# ── BLEU-4 ─────────────────────────────────────────────────────────────────
smoothing = SmoothingFunction().method1
bleu4 = corpus_bleu(
    references_bleu, hypotheses_bleu,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=smoothing,
)

# ── ROUGE-L ─────────────────────────────────────────────────────────────────
scorer  = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
rougeL_scores = [
    scorer.score(ref, hyp)["rougeL"].fmeasure
    for ref, hyp in zip(references_rouge, hypotheses_rouge)
]
rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0

# ── Print summary ───────────────────────────────────────────────────────────
print(f"\n{'Metric':<20} {'Score':>10}")
print("-" * 32)
print(f"{'Accuracy (EM)':<20} {accuracy * 100:>9.2f}%")
print(f"{'Precision':<20} {prec * 100:>9.2f}%")
print(f"{'Recall':<20} {rec * 100:>9.2f}%")
print(f"{'F1 Score':<20} {f1 * 100:>9.2f}%")
print(f"{'BLEU-4':<20} {bleu4 * 100:>9.2f}%")
print(f"{'ROUGE-L':<20} {rougeL * 100:>9.2f}%")
print("=" * 32)
print(f"Evaluated on {n_eval} samples from eval split.")

# Save metrics to file
metrics_path = os.path.join(OUTPUT_DIR, "eval_metrics.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(metrics_path, "w") as mf:
    mf.write(f"Accuracy (EM): {accuracy * 100:.2f}%\n")
    mf.write(f"Precision:     {prec * 100:.2f}%\n")
    mf.write(f"Recall:        {rec * 100:.2f}%\n")
    mf.write(f"F1 Score:      {f1 * 100:.2f}%\n")
    mf.write(f"BLEU-4:        {bleu4 * 100:.2f}%\n")
    mf.write(f"ROUGE-L:       {rougeL * 100:.2f}%\n")
    mf.write(f"Eval samples:  {n_eval}\n")
print(f"[EVAL] Metrics saved to {metrics_path}")

