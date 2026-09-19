from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results") / "tables"

input_file = RESULTS_DIR / "model_comparison_master.csv"

df = pd.read_csv(input_file)

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

MODELS = {
    "Rule-based": "rule_prediction",
    "TF-IDF + Logistic Regression": "tfidf_prediction",
    "Finance BERT": "finbert_prediction",
}


# =========================================================
# 1. Performance conditional on the TRUE class
# =========================================================

class_rows = []

for true_label, true_name in LABEL_NAMES.items():

    subset = df[df["label"] == true_label]

    for model_name, pred_col in MODELS.items():

        correct = (subset[pred_col] == true_label).sum()
        errors = len(subset) - correct

        class_rows.append({
            "true_class": true_name,
            "model": model_name,
            "support": len(subset),
            "correct": correct,
            "errors": errors,
            "class_accuracy_recall": correct / len(subset),
            "error_rate": errors / len(subset),
        })


class_summary = pd.DataFrame(class_rows)


# =========================================================
# 2. Exact error directions
# =========================================================
#
# Example:
# negative -> neutral
# positive -> neutral
# neutral  -> positive
# =========================================================

error_rows = []

for model_name, pred_col in MODELS.items():

    for true_label, true_name in LABEL_NAMES.items():

        for pred_label, pred_name in LABEL_NAMES.items():

            if true_label == pred_label:
                continue

            count = (
                (df["label"] == true_label)
                & (df[pred_col] == pred_label)
            ).sum()

            true_class_total = (
                df["label"] == true_label
            ).sum()

            error_rows.append({
                "model": model_name,
                "true_class": true_name,
                "predicted_as": pred_name,
                "count": count,
                "share_of_true_class": (
                    count / true_class_total
                ),
            })


error_directions = pd.DataFrame(error_rows)


# =========================================================
# 3. Errors only
# =========================================================

all_errors = []

for model_name, pred_col in MODELS.items():

    errors = df[
        df[pred_col] != df["label"]
    ].copy()

    errors["model"] = model_name

    errors["true_class"] = (
        errors["label"].map(LABEL_NAMES)
    )

    errors["predicted_class"] = (
        errors[pred_col].map(LABEL_NAMES)
    )

    all_errors.append(
        errors[
            [
                "sentence",
                "model",
                "true_class",
                "predicted_class",
            ]
        ]
    )


error_examples = pd.concat(
    all_errors,
    ignore_index=True,
)


# =========================================================
# 4. Save outputs
# =========================================================

class_summary.to_csv(
    RESULTS_DIR / "error_analysis_by_true_class.csv",
    index=False,
)

error_directions.to_csv(
    RESULTS_DIR / "error_direction_summary.csv",
    index=False,
)

error_examples.to_csv(
    RESULTS_DIR / "all_model_errors.csv",
    index=False,
)


# =========================================================
# 5. Print useful summaries
# =========================================================

print("\n====================================")
print("PERFORMANCE BY TRUE CLASS")
print("====================================")

print(
    class_summary.to_string(index=False)
)


print("\n====================================")
print("ERROR DIRECTIONS")
print("====================================")

print(
    error_directions[
        error_directions["count"] > 0
    ]
    .sort_values(
        ["model", "count"],
        ascending=[True, False],
    )
    .to_string(index=False)
)


print("\n====================================")
print("TOTAL ERRORS BY MODEL")
print("====================================")

for model_name, pred_col in MODELS.items():

    n_errors = (
        df[pred_col] != df["label"]
    ).sum()

    print(
        f"{model_name}: "
        f"{n_errors}/{len(df)} errors"
    )


print("\nSaved:")
print("error_analysis_by_true_class.csv")
print("error_direction_summary.csv")
print("all_model_errors.csv")