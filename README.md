# Hindi → Gujarati Translation
### Fine-tuned `Qwen/Qwen3.5-0.8B` with Accelerate + DeepSpeed ZeRO-2/3

---

## Overview

This project fine-tunes **Qwen3.5-0.8B** on the `ai4bharat/IN22-Conv` dataset to translate conversational Hindi sentences into Gujarati. Training uses **Accelerate + DeepSpeed ZeRO-2 or ZeRO-3** across multiple GPUs — no complex boilerplate required.

| Item | Detail |
|---|---|
| **Model** | `Qwen/Qwen3.5-0.8B` |
| **Dataset** | `ai4bharat/IN22-Conv` (1503 Hindi–Gujarati pairs) |
| **Task** | Hindi (Devanagari) → Gujarati (Gujarati script) |
| **Training** | Accelerate + DeepSpeed ZeRO-2 or ZeRO-3 |
| **Target GPU** | L40 / A100 (bf16) |

---

## Project Structure

```
.
├── train.py                  # Training + inference code
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

**ZeRO-2** — recommended for Qwen3.5-0.8B. Shards optimizer states + gradients.
```bash
accelerate launch --config_file accelerate_zero2.yaml train.py
```

**ZeRO-3** — maximum memory savings. Shards params + gradients + optimizer. Good for 7B+.
```bash
accelerate launch --config_file accelerate_zero3.yaml train.py
```

**Key training settings** (edit at the top of `train.py`):

| Parameter | Default | Notes |
|---|---|---|
| `BATCH_SIZE` | 8 | Per GPU |
| `GRAD_ACCUM` | 4 | Effective batch = 8 × 4 × num_gpus |
| `NUM_EPOCHS` | 10 | Small dataset needs more epochs |
| `LR` | 2e-5 | Cosine decay with warmup |
| `MAX_SEQ_LEN` | 256 | Conversational sentences are short |

---

## Inference

### Option 1 — Python (direct)

```python
from train import translate

result = translate("माँ, चलो कल एक फिल्म देखने चलते हैं।")
print(result)
```

### Option 2 — HuggingFace Transformers Server

Install the latest transformers with serving support:
```bash
pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
```

Launch the server:
```bash
transformers serve --force-model outputs/hin-guj/final --port 8000 --continuous-batching
```

Call it via OpenAI SDK:
```bash
pip install -U openai
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"
```

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="outputs/hin-guj/final",
    messages=[
        {
            "role": "user",
            "content": (
                "Translate the following Hindi sentence to Gujarati.\n\n"
                "Hindi: माँ, चलो कल एक फिल्म देखने चलते हैं।\n"
                "Gujarati:"
            )
        }
    ],
    max_tokens=200,
    temperature=0.0,
)
print(response.choices[0].message.content)
```

### Option 3 — Direct HuggingFace Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "outputs/hin-guj/final"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")
model.eval()

hindi = "क्या आप मुझे बता सकते हैं कि नजदीकी बस स्टॉप कहाँ है?"

prompt = (
    f"<|im_start|>user\n"
    f"Translate the following Hindi sentence to Gujarati.\n\n"
    f"Hindi: {hindi}\nGujarati:<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        num_beams=4,
        eos_token_id=tokenizer.eos_token_id,
    )

result = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(result)
```

---

## ZeRO-2 vs ZeRO-3 — Quick Guide

| | ZeRO-2 | ZeRO-3 |
|---|---|---|
| **Shards** | Optimizer + Gradients | Optimizer + Gradients + Parameters |
| **Memory saving** | Medium | Maximum |
| **Speed** | Faster | ~10–15% slower |
| **Best for** | Models up to ~3B | Models 7B and above |
| **For this project** | ✅ Use this | Overkill for 0.8B |

---

## Dataset Details

- **Source:** `ai4bharat/IN22-Conv` — conversational evaluation benchmark across 22 Indic languages
- **Split used:** `test` split (the only available split), repurposed for fine-tuning
- **Pairs extracted:** `sentence_hin_Deva` (Hindi) + `sentence_guj_Gujr` (Gujarati)
- **Split:** 80% train / 10% val / 10% test (~1200 / 150 / 150)

---