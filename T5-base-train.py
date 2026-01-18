import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate

# =====================
# Device
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Model & Tokenizer
# =====================
tokenizer = T5Tokenizer.from_pretrained("epoch0")
model = T5ForConditionalGeneration.from_pretrained("epoch0").to(device)

# =====================
# Dataset
# =====================
raw_datasets = load_dataset("cnn_dailymail", "3.0.0")

# =====================
# Preprocessing
# =====================
def preprocess_function(example):
    input_text = "summarize: " + example["article"]
    target_text = example["highlights"]

    input_ids = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).input_ids.squeeze(0)

    labels = tokenizer(
        target_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).input_ids.squeeze(0)

    labels[labels == tokenizer.pad_token_id] = -100
    return {"input_ids": input_ids, "labels": labels}


train_data = [preprocess_function(ex) for ex in tqdm(raw_datasets["train"])]
val_data = [preprocess_function(ex) for ex in tqdm(raw_datasets["validation"])]

# =====================
# DataLoader
# =====================
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

# =====================
# Optimizer
# =====================
optimizer = AdamW(model.parameters(), lr=2e-5)

# =====================
# Evaluation
# =====================
def evaluate_bleu_chrf(model, tokenizer, val_data, max_samples=100):
    model.eval()
    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")

    preds = []
    refs = []

    with torch.no_grad():
        for example in tqdm(val_data[:max_samples]):
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            label_ids = example["labels"].tolist()

            ref = tokenizer.decode(
                [i for i in label_ids if i != -100 and i != tokenizer.pad_token_id],
                skip_special_tokens=True,
            )

            output_ids = model.generate(input_ids, max_length=128)[0]
            pred = tokenizer.decode(output_ids, skip_special_tokens=True)

            preds.append(pred)
            refs.append([ref])

    bleu_score = bleu.compute(predictions=preds, references=refs)["bleu"]
    chrf_score = chrf.compute(predictions=preds, references=[r[0] for r in refs])["score"]

    return bleu_score, chrf_score

# =====================
# Training Loop
# =====================
num_epochs = 6

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} | avg loss: {avg_loss:.4f}")

    # bleu, chrf = evaluate_bleu_chrf(model, tokenizer, val_data)

    save_dir = f"epoch{epoch}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
