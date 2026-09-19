from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score


# =========================================================
# 1. Paths
# =========================================================

RESULTS_DIR = Path("results") / "tables"

RULE_FILE = RESULTS_DIR / "rule_based_predictions.csv"
TFIDF_FILE = RESULTS_DIR / "tfidf_predictions.csv"
FINBERT_FILE = RESULTS_DIR / "finbert_predictions.csv"


# =========================================================
# 2. Load prediction files
# =========================================================

rule_df = pd.read_csv(RULE_FILE)
tfidf_df = pd.read_csv(TFIDF_FILE)
finbert_df = pd.read_csv(FINBERT_FILE)

print("Files loaded:")
print("Rule:", rule_df.shape)
print("TF-IDF:", tfidf_df.shape)
print("FinBERT:", finbert_df.shape)


# =========================================================
# 3. Find prediction columns automatically
# =========================================================

def find_prediction_column(df, model_name):
    candidates = [
        f"{model_name}_prediction",
        "prediction",
        "predicted_label",
        "pred_label",
        "y_pred",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    prediction_like = [
        col for col in df.columns
        if "prediction" in col.lower()
    ]

    if len(prediction_like) == 1:
        return prediction_like[0]

    raise ValueError(
        f"Could not identify prediction column for {model_name}. "
        f"Columns are: {list(df.columns)}"
    )


rule_pred_col = find_prediction_column(
    rule_df,
    "rule_based"
)

tfidf_pred_col = find_prediction_column(
    tfidf_df,
    "tfidf"
)

finbert_pred_col = find_prediction_column(
    finbert_df,
    "finbert"
)

print("\nPrediction columns:")
print("Rule:", rule_pred_col)
print("TF-IDF:", tfidf_pred_col)
print("FinBERT:", finbert_pred_col)


# =========================================================
# 4. Build master comparison table
# =========================================================

rule_small = rule_df[
    ["sentence", rule_pred_col]
].rename(
    columns={
        rule_pred_col: "rule_prediction"
    }
)

tfidf_small = tfidf_df[
    ["sentence", tfidf_pred_col]
].rename(
    columns={
        tfidf_pred_col: "tfidf_prediction"
    }
)

# FinBERT file contains the true label
finbert_small = finbert_df[
    [
        "sentence",
        "label",
        "label_name",
        finbert_pred_col,
    ]
].rename(
    columns={
        finbert_pred_col:
            "finbert_prediction"
    }
)

master = (
    finbert_small
    .merge(
        tfidf_small,
        on="sentence",
        how="inner",
    )
    .merge(
        rule_small,
        on="sentence",
        how="inner",
    )
)

print("\nMerged observations:", len(master))

if len(master) != len(finbert_df):
    print(
        "WARNING: merged row count differs "
        "from FinBERT test set."
    )


# =========================================================
# 5. Correct / incorrect flags
# =========================================================

master["rule_correct"] = (
    master["rule_prediction"]
    == master["label"]
)

master["tfidf_correct"] = (
    master["tfidf_prediction"]
    == master["label"]
)

master["finbert_correct"] = (
    master["finbert_prediction"]
    == master["label"]
)


# =========================================================
# 6. Linguistic phenomenon detection
# =========================================================

patterns = {

    "negation": (
        r"\b("
        r"not|no|never|without|neither|nor|"
        r"cannot|can't|won't|didn't|doesn't|"
        r"isn't|aren't|wasn't|weren't|n't"
        r")\b"
    ),

    "contrast": (
        r"\b("
        r"but|however|although|though|"
        r"despite|whereas|while|nevertheless|"
        r"yet"
        r")\b"
    ),

    "expectation": (
        r"\b("
        r"expect|expects|expected|expecting|"
        r"forecast|forecasts|forecasted|"
        r"guidance|outlook|"
        r"anticipate|anticipates|anticipated|"
        r"target|targets|"
        r"beat|beats|miss|missed"
        r")\b"
    ),

    "uncertainty": (
        r"\b("
        r"may|might|could|possibly|possible|"
        r"uncertain|uncertainty|"
        r"likely|unlikely|perhaps|potential|"
        r"potentially"
        r")\b"
    ),
}


for phenomenon, pattern in patterns.items():

    master[phenomenon] = (
        master["sentence"]
        .str.lower()
        .str.contains(
            pattern,
            regex=True,
            na=False,
        )
    )


# =========================================================
# 7. Overall model comparison
# =========================================================

models = {
    "Rule-based": "rule_prediction",
    "TF-IDF + Logistic Regression":
        "tfidf_prediction",
    "Finance BERT": "finbert_prediction",
}

overall_rows = []

for model_name, prediction_col in models.items():

    accuracy = accuracy_score(
        master["label"],
        master[prediction_col],
    )

    macro_f1 = f1_score(
        master["label"],
        master[prediction_col],
        average="macro",
        zero_division=0,
    )

    overall_rows.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "n": len(master),
        }
    )

overall_results = pd.DataFrame(
    overall_rows
)

print("\n===================================")
print("OVERALL MODEL COMPARISON")
print("===================================")

print(
    overall_results.to_string(
        index=False
    )
)


# =========================================================
# 8. Performance by linguistic phenomenon
# =========================================================

linguistic_rows = []

for phenomenon in patterns.keys():

    subset = master[
        master[phenomenon]
    ]

    print(
        f"\n{phenomenon.upper()}: "
        f"{len(subset)} sentences"
    )

    if len(subset) == 0:
        continue

    for model_name, prediction_col in models.items():

        accuracy = accuracy_score(
            subset["label"],
            subset[prediction_col],
        )

        macro_f1 = f1_score(
            subset["label"],
            subset[prediction_col],
            average="macro",
            zero_division=0,
        )

        linguistic_rows.append(
            {
                "phenomenon": phenomenon,
                "model": model_name,
                "n": len(subset),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )


linguistic_results = pd.DataFrame(
    linguistic_rows
)


print("\n===================================")
print("LINGUISTIC SUBSET RESULTS")
print("===================================")

print(
    linguistic_results.to_string(
        index=False
    )
)


# =========================================================
# 9. Disagreement categories
# =========================================================

def disagreement_category(row):

    r = row["rule_correct"]
    t = row["tfidf_correct"]
    b = row["finbert_correct"]

    if r and t and b:
        return "all_correct"

    if (not r) and (not t) and (not b):
        return "all_wrong"

    if b and not t and not r:
        return "finbert_only_correct"

    if t and not b and not r:
        return "tfidf_only_correct"

    if r and not t and not b:
        return "rule_only_correct"

    if b and t and not r:
        return "finbert_tfidf_correct"

    if b and r and not t:
        return "finbert_rule_correct"

    if t and r and not b:
        return "tfidf_rule_correct"

    return "other"


master[
    "disagreement_category"
] = master.apply(
    disagreement_category,
    axis=1,
)


disagreement_summary = (
    master[
        "disagreement_category"
    ]
    .value_counts()
    .rename_axis(
        "category"
    )
    .reset_index(
        name="count"
    )
)

disagreement_summary[
    "percentage"
] = (
    disagreement_summary["count"]
    / len(master)
)


print("\n===================================")
print("DISAGREEMENT SUMMARY")
print("===================================")

print(
    disagreement_summary.to_string(
        index=False
    )
)


# =========================================================
# 10. Cases where models predict different labels
# =========================================================

prediction_disagreement = master[
    (
        master["rule_prediction"]
        != master["tfidf_prediction"]
    )
    |
    (
        master["rule_prediction"]
        != master["finbert_prediction"]
    )
    |
    (
        master["tfidf_prediction"]
        != master["finbert_prediction"]
    )
].copy()


print(
    "\nSentences where at least "
    "two models disagree:",
    len(prediction_disagreement),
)


# =========================================================
# 11. Convert numeric labels to readable names
# =========================================================

label_names = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

for col in [
    "rule_prediction",
    "tfidf_prediction",
    "finbert_prediction",
]:

    master[
        col.replace(
            "_prediction",
            "_prediction_name"
        )
    ] = master[col].map(
        label_names
    )


# Re-create disagreement table with names
prediction_disagreement = master[
    (
        master["rule_prediction"]
        != master["tfidf_prediction"]
    )
    |
    (
        master["rule_prediction"]
        != master["finbert_prediction"]
    )
    |
    (
        master["tfidf_prediction"]
        != master["finbert_prediction"]
    )
].copy()


# =========================================================
# 12. Save outputs
# =========================================================

master.to_csv(
    RESULTS_DIR
    / "model_comparison_master.csv",
    index=False,
)

overall_results.to_csv(
    RESULTS_DIR
    / "model_comparison_summary.csv",
    index=False,
)

linguistic_results.to_csv(
    RESULTS_DIR
    / "linguistic_subset_results.csv",
    index=False,
)

disagreement_summary.to_csv(
    RESULTS_DIR
    / "disagreement_summary.csv",
    index=False,
)

prediction_disagreement.to_csv(
    RESULTS_DIR
    / "model_disagreements.csv",
    index=False,
)


print("\n===================================")
print("ANALYSIS COMPLETE")
print("===================================")

print("\nSaved:")
print("model_comparison_master.csv")
print("model_comparison_summary.csv")
print("linguistic_subset_results.csv")
print("disagreement_summary.csv")
print("model_disagreements.csv")