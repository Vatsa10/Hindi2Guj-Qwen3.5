# train.py
# Hindi → Gujarati Translation
# Dataset : ai4bharat/IN22-Conv
# Model   : Qwen/Qwen3.5-0.8B
# Training: Accelerate + DeepSpeed ZeRO-2 or ZeRO-3
#
# Run with ZeRO-2:
#   accelerate launch --config_file accelerate_zero2.yaml train.py
#
# Run with ZeRO-3:
#   accelerate launch --config_file accelerate_zero3.yaml train.py

import os
import math
import random
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
# ─────────────────────────────────────────────────────────────
# Config — change these as needed
# ─────────────────────────────────────────────────────────────

login(token=os.getenv("HF_TOKEN"))

MODEL_NAME    = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR    = "outputs/hin-guj"
MAX_SEQ_LEN   = 256
BATCH_SIZE    = 2          # per GPU
GRAD_ACCUM    = 4          # effective batch = 8 * 4 * num_gpus
NUM_EPOCHS    = 20         # small dataset (1503 samples) needs more epochs
LR            = 2e-5
WARMUP_RATIO  = 0.1
LOGGING_STEPS = 10


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "<|im_start|>user\n"
    "Translate the following Hindi sentence to Gujarati.\n\n"
    "Hindi: {src}\n"
    "Gujarati:<|im_end|>\n"
    "<|im_start|>assistant\n"
    "{tgt}<|im_end|>"
)

class HindiGujaratiDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.samples = []
        for p in pairs:
            text = PROMPT_TEMPLATE.format(src=p["src"], tgt=p["tgt"])
            # Tokenize prompt-only to find where target starts (for loss masking)
            prompt_only = PROMPT_TEMPLATE.format(src=p["src"], tgt="").rsplit("<|im_start|>assistant\n", 1)[0] + "<|im_start|>assistant\n"
            enc      = tokenizer(text,         max_length=MAX_SEQ_LEN, truncation=True, padding="max_length", return_tensors="pt")
            enc_prompt = tokenizer(prompt_only, max_length=MAX_SEQ_LEN, truncation=True, return_tensors="pt")

            input_ids      = enc["input_ids"].squeeze()
            attention_mask = enc["attention_mask"].squeeze()
            labels         = input_ids.clone()

            # Mask prompt tokens — only train on the Gujarati output
            prompt_len = enc_prompt["input_ids"].shape[1]
            labels[:prompt_len] = -100
            labels[labels == tokenizer.pad_token_id] = -100

            self.samples.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_pairs():
    # IN22-Conv is n-way parallel — load "all" and extract hin_Deva + guj_Gujr
    raw = load_dataset("ai4bharat/IN22-Conv", "default", split="test")
    pairs = [
        {"src": row["hin_Deva"], "tgt": row["guj_Gujr"]}
        for row in raw
        if row.get("hin_Deva") and row.get("guj_Gujr")
    ]
    random.seed(42)
    random.shuffle(pairs)
    # 80 / 10 / 10 split
    n       = len(pairs)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    return pairs[:n_train], pairs[n_train:n_train + n_val]


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train():
    # Accelerator handles all distributed / ZeRO logic automatically
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM)

    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Using {accelerator.num_processes} GPUs")
        print(f"Mixed precision: {accelerator.mixed_precision}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Data ---
    if accelerator.is_main_process:
        print("Loading dataset...")
    train_pairs, val_pairs = load_pairs()
    if accelerator.is_main_process:
        print(f"  Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    train_ds = HindiGujaratiDataset(train_pairs, tokenizer) 
    val_ds   = HindiGujaratiDataset(val_pairs, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model ---
    # With ZeRO-3, Accelerate/DeepSpeed shards the model across GPUs during init
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False   # required for gradient checkpointing

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # --- Scheduler ---
    total_steps   = math.ceil(len(train_loader) / GRAD_ACCUM) * NUM_EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Accelerate prepare — this is where ZeRO kicks in ---
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    if accelerator.is_main_process:
        print(f"Total steps: {total_steps}  |  Warmup: {warmup_steps}")

    # --- Training loop ---
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss  = 0.0
        global_step = 0

        for step, batch in enumerate(train_loader):
            # Accelerate context manager handles gradient accumulation cleanly
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss  += loss.item()
            global_step += 1

            if accelerator.is_main_process and global_step % LOGGING_STEPS == 0:
                print(f"  Epoch {epoch+1} | Step {global_step} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)

        # Gather val loss across all GPUs and average
        val_loss_tensor = torch.tensor(val_loss, device=accelerator.device)
        print("Device:", accelerator.device)
        val_loss_tensor = accelerator.reduce(val_loss_tensor, reduction="mean")
        val_loss = val_loss_tensor.item()

        if accelerator.is_main_process:
            avg_train = total_loss / global_step
            print(f"\n  ── Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {val_loss:.4f}\n")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                #accelerator.save_state(f"{OUTPUT_DIR}/best_checkpoint")
                #print(f"  ✓ Best checkpoint saved (val_loss={val_loss:.4f})")

    # --- Save final model ---
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        f"{OUTPUT_DIR}/final",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
        print(f"\n✅ Training complete! Model saved to {OUTPUT_DIR}/final")


# ─────────────────────────────────────────────────────────────
# Inference — run separately after training
# ─────────────────────────────────────────────────────────────

def translate(hindi_text, model_path=f"{OUTPUT_DIR}/final"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    model.eval()

    prompt = (
        f"<|im_start|>user\n"
        f"Translate the following Hindi sentence to Gujarati.\n\n"
        f"Hindi: {hindi_text}\n"
        f"Gujarati:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=200,
            do_sample=False, num_beams=4,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    train()