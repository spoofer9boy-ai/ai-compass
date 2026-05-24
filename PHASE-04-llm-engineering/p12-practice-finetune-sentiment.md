# Practice: Fine-Tune Sentiment Classifier

**Phase:** PHASE-04-llm-engineering  
**Subjects Required:** 75 Tokenization: BPE, 80 Fine-Tuning: Full  
**Estimated Time:** 240 minutes  
**Difficulty:** Intermediate

## Industry Context

You are an ML engineer at a customer-support SaaS startup. Support tickets arrive at 2,000 per hour, and the ops team needs real-time sentiment tagging (positive, neutral, negative) to prioritize escalations. Off-the-shelf APIs cost \$0.002 per request—prohibitively expensive at scale. The product manager wants an in-house classifier that runs on a single GPU and beats a 90% accuracy threshold. You have 5,000 labeled tickets and a pretrained BERT-base model.

## The Problem

Build a complete sentiment-classification fine-tuning pipeline using the Hugging Face ecosystem. Your pipeline must:

1. Load a pretrained `bert-base-uncased` model and its BPE tokenizer.
2. Tokenize the IMDB sentiment dataset (or an equivalent CSV with `text` and `label` columns) with proper padding and truncation.
3. Fine-tune the entire model (full fine-tuning) for 3 epochs with appropriate hyperparameters.
4. Evaluate on a held-out test set and report accuracy and F1.
5. Save the model and tokenizer to disk for inference.

You may use `transformers`, `datasets`, and `torch`. Do not use `scikit-learn` for the model itself, but you may use it for metrics if desired.

## Constraints

- Must use full fine-tuning (update all parameters, not just heads or adapters).
- Max sequence length: 256 tokens.
- Batch size must fit on a single GPU with 8 GB VRAM (use gradient accumulation if needed).
- Training must complete in under 30 minutes on a single GPU.
- Do not use automatic mixed precision (keep it simple for debugging).

## Starter Code

```python
# starter.py
import torch
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# TODO: Load tokenizer and model
# TODO: Load dataset and create train/test split
# TODO: Tokenize dataset
# TODO: Define compute_metrics function
# TODO: Configure TrainingArguments
# TODO: Instantiate Trainer and train
# TODO: Evaluate and save

if __name__ == "__main__":
    pass
```

## Evaluation Criteria

1. **Correctness:** Model trains without errors and achieves ≥ 85% accuracy on the test set.
2. **Tokenization:** Uses the BPE tokenizer correctly with padding, truncation, and attention masks.
3. **Full Fine-Tuning:** All model parameters have `requires_grad=True` (no frozen layers).
4. **Reproducibility:** Sets random seeds and reports final metrics consistently.
5. **Inference Ready:** Saved model can be reloaded with `from_pretrained` for prediction.

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
# solution.py
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 1. Load BPE tokenizer and pretrained model
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

# Verify full fine-tuning: all parameters trainable
for param in model.parameters():
    param.requires_grad = True

# 2. Load dataset (IMDB is a standard proxy for binary sentiment)
dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# 3. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# 4. Training arguments (fit 8 GB VRAM)
training_args = TrainingArguments(
    output_dir="./sentiment-bert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    seed=SEED,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=SEED).select(range(5000)),
    eval_dataset=tokenized_datasets["test"].shuffle(seed=SEED).select(range(1000)),
    compute_metrics=compute_metrics,
)

# 5. Train and evaluate
trainer.train()
results = trainer.evaluate()
print("Evaluation results:", results)

# 6. Save for inference
trainer.save_model("./sentiment-bert-final")
tokenizer.save_pretrained("./sentiment-bert-final")

# Quick inference check
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="./sentiment-bert-final", tokenizer="./sentiment-bert-final")
print(classifier("This movie was absolutely fantastic!"))
```

</details>

## What You Actually Learned

- **Tokenization: BPE** — You loaded a BPE tokenizer, applied padding and truncation to a fixed length, and understood why attention masks are necessary for variable-length sequences.
- **Fine-Tuning: Full** — You unfroze all parameters and ran end-to-end backpropagation through the entire BERT stack, observing how pretrained representations adapt to a downstream classification task.
- **Practical Engineering** — You managed VRAM constraints via batch sizing, set reproducibility seeds, and exported a production-ready model directory compatible with Hugging Face inference pipelines.
