import os
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

# 0. Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 1. Environment / performance settings
# Helps reduce CUDA memory fragmentation in some long-seq workloads
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Enable TF32 on Ampere+ GPUs for faster matmul with minimal accuracy impact
torch.backends.cuda.matmul.allow_tf32 = True

# 2. Load model & tokenizer
model_name = "../Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Use EOS as PAD to avoid padding-related errors for causal LM
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",            # Automatically dispatch layers across available GPUs
    torch_dtype=torch.bfloat16,   # Recommended on A100/A800/H100, etc.
    trust_remote_code=True
)


# 3. Load dataset (CNN/DailyMail)
TRAIN_FRACTION = 0.20
dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"train[:{int(TRAIN_FRACTION*100)}%]")

# 4. Build ChatML-style inputs and labels
def format_sample(example):
    article = example["article"]
    summary = example["highlights"]

    # ChatML prompt format: system + user + assistant (assistant content is supervised)
    prompt = (
        "<|im_start|>system\n"
        "You are a precise summarization assistant.\n"
        "Summarize the following article in 2–3 full sentences.\n"
        "Guidelines:\n"
        "1. Use only facts stated in the article. Do not add or omit anything.\n"
        "2. Preserve all names, dates, locations, and key actions.\n"
        "3. Write clearly and factually, without rephrasing or interpreting.\n"
        "4. Avoid generic summaries or creativity.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{article}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # Tokenize prompt and target separately (so we can mask prompt in labels)
    prompt_ids = tokenizer(prompt, truncation=True, max_length=1024)["input_ids"]
    summary_ids = tokenizer(summary, truncation=True, max_length=256)["input_ids"]

    # Concatenate prompt + summary as model input
    input_ids = prompt_ids + summary_ids

    # Mask prompt tokens in labels with -100 so loss is computed only on summary tokens
    labels = [-100] * len(prompt_ids) + summary_ids

    return {"input_ids": input_ids, "labels": labels}

tokenized_dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

# 5. Custom collate_fn (important for padding labels with -100)
def collate_fn(batch):
    # Separate inputs and labels
    input_features = [{k: ex[k] for k in ex if k != "labels"} for ex in batch]
    labels_list = [ex["labels"] for ex in batch]

    # Pad input_ids (+ attention_mask) to the same length
    input_batch = tokenizer.pad(
        input_features,
        padding=True,
        return_tensors="pt",
    )

    # Pad labels to the same sequence length as input_ids; use -100 as ignore_index
    seq_len = input_batch["input_ids"].shape[1]
    padded_labels = []
    for l in labels_list:
        if len(l) < seq_len:
            l = l + [-100] * (seq_len - len(l))
        else:
            l = l[:seq_len]
        padded_labels.append(l)

    input_batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    return input_batch

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./Qwen2.5-CNN",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    logging_steps=10,
    save_strategy="epoch",        # Save once per epoch
    save_total_limit=None,        # Keep all epoch checkpoints
    learning_rate=2e-5,
    fp16=False,
    bf16=True,
    report_to="none",
    seed=SEED,
)

# 7. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=collate_fn,
)

trainer.train()
