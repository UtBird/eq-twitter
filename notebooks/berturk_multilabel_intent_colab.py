"""
Colab-ready training script for:
https://huggingface.co/datasets/deprem-private/intent_test_v13_anonymized

This script trains BERTurk directly on the dataset's original multi-label
intent annotations without converting them to binary help / not_help labels.
"""

# If you run this in Colab, uncomment the next line first:
# !pip install -q transformers datasets accelerate scikit-learn pandas numpy

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "dbmdz/bert-base-turkish-cased"
DATASET_NAME = "deprem-private/intent_test_v13_anonymized"
MAX_LENGTH = 256
TEST_SIZE = 0.15
RANDOM_SEED = 42
THRESHOLD = 0.5


def build_label_space(train_split):
    all_labels = set()
    for row in train_split:
        for label in row["label"]:
            all_labels.add(label)

    label_list = sorted(all_labels)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label_list, label2id, id2label


def encode_example(example, label2id, label_count):
    multi_hot = [0.0] * label_count
    for label in example["label"]:
        multi_hot[label2id[label]] = 1.0

    example["labels"] = multi_hot
    # In this dataset, `image_url` contains the tweet text.
    example["text"] = example["image_url"]
    return example


def tokenize_batch(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def build_compute_metrics(threshold=THRESHOLD):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)

        return {
            "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "samples_f1": f1_score(labels, preds, average="samples", zero_division=0),
            "subset_accuracy": accuracy_score(labels, preds),
        }

    return compute_metrics


def main():
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    label_list, label2id, id2label = build_label_space(dataset["train"])
    print("Labels:", label_list)

    dataset = dataset.map(
        lambda example: encode_example(example, label2id, len(label_list))
    )

    split_dataset = dataset["train"].train_test_split(
        test_size=TEST_SIZE,
        seed=RANDOM_SEED,
    )
    train_ds = split_dataset["train"]
    val_ds = split_dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = train_ds.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True)
    val_ds = val_ds.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True)

    keep_columns = ["input_ids", "attention_mask", "labels"]
    train_ds = train_ds.remove_columns(
        [column for column in train_ds.column_names if column not in keep_columns]
    )
    val_ds = val_ds.remove_columns(
        [column for column in val_ds.column_names if column not in keep_columns]
    )

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir="./berturk-deprem-intent",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)

    trainer.save_model("./berturk-deprem-intent-final")
    tokenizer.save_pretrained("./berturk-deprem-intent-final")

    sample_text = (
        "Acil çadır ve battaniye ihtiyacı var. Hatay Antakya Odabaşı Mahallesi."
    )
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    predicted = [label_list[idx] for idx, prob in enumerate(probs) if prob >= THRESHOLD]
    print("Sample text:", sample_text)
    print("Predicted labels:", predicted)


if __name__ == "__main__":
    main()
