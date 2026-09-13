from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# =========================================================
# 1. Configuration
# =========================================================

MODEL_NAME = "yiyanghkust/finbert-pretrain"

DATA_DIR = Path("data") / "processed"
RESULTS_DIR = Path("results") / "tables"
MODEL_DIR = Path("models") / "finbert_best"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_LENGTH = 128
id2label = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

label2id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
# =========================================================
# 2. Reproducibility
# =========================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)
print("Model:", MODEL_NAME)


# =========================================================
# 3. Load frozen PhraseBank splits
# =========================================================

train_df = pd.read_csv(
    DATA_DIR / "phrasebank_train.csv"
)

validation_df = pd.read_csv(
    DATA_DIR / "phrasebank_validation.csv"
)

test_df = pd.read_csv(
    DATA_DIR / "phrasebank_test.csv"
)

print("\nDataset sizes:")
print("Train:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))


# =========================================================
# 4. Load tokenizer
# =========================================================

print("\nLoading tokenizer...")

tokenizer = BertTokenizer.from_pretrained(
    MODEL_NAME,
    do_lower_case=True,
)



# =========================================================
# 5. Dataset class
# =========================================================

class PhraseBankDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=128,
    ):
        self.texts = dataframe["sentence"].tolist()
        self.labels = (
            dataframe["label"]
            .astype(int)
            .tolist()
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        encoding["labels"] = label

        return encoding


train_dataset = PhraseBankDataset(
    train_df,
    tokenizer,
    MAX_LENGTH,
)

validation_dataset = PhraseBankDataset(
    validation_df,
    tokenizer,
    MAX_LENGTH,
)

test_dataset = PhraseBankDataset(
    test_df,
    tokenizer,
    MAX_LENGTH,
)


# =========================================================
# 6. Dynamic padding
# =========================================================
#
# Instead of padding every sentence to 128 tokens,
# we only pad each batch to its longest sentence.
#
# This is useful for CPU training.
# =========================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt",
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
)


# =========================================================
# 7. Load finance-domain BERT
# =========================================================

print("\nLoading finance-domain BERT...")

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

model.to(DEVICE)


# =========================================================
# 8. Optimizer + learning-rate scheduler
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

total_training_steps = (
    len(train_loader) * EPOCHS
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_training_steps,
)


# =========================================================
# 9. Evaluation function
# =========================================================

def evaluate(model, dataloader):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0

    with torch.no_grad():

        for batch in dataloader:

            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )

            all_labels.extend(
                batch["labels"]
                .cpu()
                .numpy()
            )

    mean_loss = (
        total_loss / len(dataloader)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
    )

    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": np.array(all_labels),
        "predictions": np.array(all_predictions),
    }


# =========================================================
# 10. Training loop
# =========================================================

best_validation_f1 = -1

training_history = []

print("\n====================================")
print("STARTING FINBERT TRAINING")
print("====================================")


for epoch in range(EPOCHS):

    epoch_number = epoch + 1

    print(
        f"\nEpoch {epoch_number}/{EPOCHS}"
    )

    start_time = time.time()

    model.train()

    total_train_loss = 0

    for step, batch in enumerate(
        train_loader,
        start=1,
    ):

        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        # Prevent unstable very large gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

        if step % 50 == 0:

            print(
                f"  Step {step}/{len(train_loader)}"
            )


    mean_train_loss = (
        total_train_loss
        / len(train_loader)
    )


    # -----------------------------------------------------
    # Validation after each epoch
    # -----------------------------------------------------

    validation_results = evaluate(
        model,
        validation_loader,
    )

    elapsed = time.time() - start_time


    print("\nTraining loss:")
    print(
        round(mean_train_loss, 4)
    )

    print("Validation loss:")
    print(
        round(
            validation_results["loss"],
            4,
        )
    )

    print("Validation accuracy:")
    print(
        round(
            validation_results["accuracy"],
            4,
        )
    )

    print("Validation macro F1:")
    print(
        round(
            validation_results["macro_f1"],
            4,
        )
    )

    print(
        "Epoch time:",
        round(elapsed / 60, 2),
        "minutes",
    )


    training_history.append(
        {
            "epoch": epoch_number,
            "train_loss":
                mean_train_loss,
            "validation_loss":
                validation_results["loss"],
            "validation_accuracy":
                validation_results["accuracy"],
            "validation_macro_f1":
                validation_results["macro_f1"],
            "epoch_minutes":
                elapsed / 60,
        }
    )


    # -----------------------------------------------------
    # Save best checkpoint using validation Macro-F1
    # -----------------------------------------------------

    if (
        validation_results["macro_f1"]
        > best_validation_f1
    ):

        best_validation_f1 = (
            validation_results["macro_f1"]
        )

        print(
            "New best validation Macro-F1."
        )

        print(
            "Saving checkpoint..."
        )

        model.save_pretrained(
            MODEL_DIR
        )

        tokenizer.save_pretrained(
            MODEL_DIR
        )


# =========================================================
# 11. Save training history
# =========================================================

history_df = pd.DataFrame(
    training_history
)

history_df.to_csv(
    RESULTS_DIR
    / "finbert_training_history.csv",
    index=False,
)


# =========================================================
# 12. Reload best validation checkpoint
# =========================================================

print("\n====================================")
print("LOADING BEST CHECKPOINT")
print("====================================")

best_model = (
    BertForSequenceClassification
    .from_pretrained(MODEL_DIR)
)
best_model.to(DEVICE)


# =========================================================


# 13. Final test evaluation
# =========================================================
#
# This is where we finally evaluate on the frozen
# 518-sentence test set.
# =========================================================

print("\n====================================")
print("FINAL FINBERT TEST")
print("====================================")

test_results = evaluate(
    best_model,
    test_loader,
)

y_true = test_results["labels"]
y_pred = test_results["predictions"]


print("\nAccuracy:")
print(
    round(
        test_results["accuracy"],
        4,
    )
)

print("\nMacro F1:")
print(
    round(
        test_results["macro_f1"],
        4,
    )
)


print("\nClassification report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=[
            "negative",
            "neutral",
            "positive",
        ],
        digits=4,
        zero_division=0,
    )
)


# =========================================================
# 14. Confusion matrix
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2],
)

print("\nConfusion matrix:")
print(cm)


# =========================================================
# 15. Save predictions
# =========================================================

prediction_df = test_df.copy()

prediction_df[
    "finbert_prediction"
] = y_pred

prediction_df.to_csv(
    RESULTS_DIR
    / "finbert_predictions.csv",
    index=False,
)


# =========================================================
# 16. Save confusion matrix
# =========================================================

pd.DataFrame(
    cm,
    index=[
        "true_negative",
        "true_neutral",
        "true_positive",
    ],
    columns=[
        "pred_negative",
        "pred_neutral",
        "pred_positive",
    ],
).to_csv(
    RESULTS_DIR
    / "finbert_confusion_matrix.csv"
)


# =========================================================
# 17. Save classification report
# =========================================================

report_dict = classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    target_names=[
        "negative",
        "neutral",
        "positive",
    ],
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(
    report_dict
).T.to_csv(
    RESULTS_DIR
    / "finbert_classification_report.csv"
)


# =========================================================
# 18. Save summary
# =========================================================

summary = pd.DataFrame(
    [
        {
            "model":
                MODEL_NAME,
            "accuracy":
                test_results["accuracy"],
            "macro_f1":
                test_results["macro_f1"],
            "best_validation_macro_f1":
                best_validation_f1,
            "epochs":
                EPOCHS,
            "batch_size":
                BATCH_SIZE,
            "learning_rate":
                LEARNING_RATE,
            "weight_decay":
                WEIGHT_DECAY,
        }
    ]
)

summary.to_csv(
    RESULTS_DIR
    / "finbert_summary.csv",
    index=False,
)


print("\nFinBERT experiment complete.")