from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

# ---------------------------------------------------------
# 1. Load Financial PhraseBank
# ---------------------------------------------------------

print("Loading Financial PhraseBank...")

dataset = load_dataset(
    "takala/financial_phrasebank",
    "sentences_75agree",
)

df = dataset["train"].to_pandas()

print(f"\nOriginal rows: {len(df)}")
print(df.head())


# ---------------------------------------------------------
# 2. Basic cleaning
# ---------------------------------------------------------

df["sentence"] = df["sentence"].astype(str).str.strip()
df["label"] = df["label"].astype(int)
df["label_name"] = df["label"].map(LABEL_NAMES)

# Empty sentences
empty_count = (df["sentence"] == "").sum()
print(f"\nEmpty sentences: {empty_count}")

df = df[df["sentence"] != ""].copy()


# ---------------------------------------------------------
# 3. Check duplicate sentences
# ---------------------------------------------------------

duplicate_count = df.duplicated(subset=["sentence"]).sum()

print(f"Exact duplicate sentences: {duplicate_count}")

df = (
    df
    .drop_duplicates(subset=["sentence"])
    .reset_index(drop=True)
)

print(f"Rows after removing duplicates: {len(df)}")


# ---------------------------------------------------------
# 4. Inspect label distribution
# ---------------------------------------------------------

print("\nLabel counts:")
print(df["label_name"].value_counts())

print("\nLabel proportions:")
print(df["label_name"].value_counts(normalize=True))


# ---------------------------------------------------------
# 5. Stratified 70 / 15 / 15 split
# ---------------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=RANDOM_STATE,
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=RANDOM_STATE,
)

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ---------------------------------------------------------
# 6. Print split sizes
# ---------------------------------------------------------

print("\n--- DATASET SPLITS ---")

print(f"Train:      {len(train_df)}")
print(f"Validation: {len(validation_df)}")
print(f"Test:       {len(test_df)}")

print("\nTrain distribution:")
print(train_df["label_name"].value_counts(normalize=True))

print("\nValidation distribution:")
print(validation_df["label_name"].value_counts(normalize=True))

print("\nTest distribution:")
print(test_df["label_name"].value_counts(normalize=True))


# ---------------------------------------------------------
# 7. Save processed data
# ---------------------------------------------------------

output_dir = Path("data") / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "phrasebank_train.csv", index=False)
validation_df.to_csv(
    output_dir / "phrasebank_validation.csv",
    index=False
)
test_df.to_csv(output_dir / "phrasebank_test.csv", index=False)

print("\nSaved:")
print(output_dir / "phrasebank_train.csv")
print(output_dir / "phrasebank_validation.csv")
print(output_dir / "phrasebank_test.csv")

print("\nDATA PREPARATION COMPLETE")