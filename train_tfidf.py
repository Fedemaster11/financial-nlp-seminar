from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_DIR = Path("data") / "processed"
RESULTS_DIR = Path("results") / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load frozen splits
# ---------------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "phrasebank_train.csv")
validation_df = pd.read_csv(DATA_DIR / "phrasebank_validation.csv")
test_df = pd.read_csv(DATA_DIR / "phrasebank_test.csv")

print("Train:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))


# ---------------------------------------------------------
# TF-IDF representation
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(train_df["sentence"])
X_validation = vectorizer.transform(validation_df["sentence"])
X_test = vectorizer.transform(test_df["sentence"])

y_train = train_df["label"]
y_validation = validation_df["label"]
y_test = test_df["label"]

print("\nTF-IDF matrix:")
print("Train shape:", X_train.shape)


# ---------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------

model = LogisticRegression(
    max_iter=3000,
    solver="lbfgs",
)

model.fit(X_train, y_train)


# ---------------------------------------------------------
# Validation results
# ---------------------------------------------------------

val_pred = model.predict(X_validation)

print("\n--- VALIDATION ---")
print("Accuracy:", accuracy_score(y_validation, val_pred))
print("Macro F1:", f1_score(y_validation, val_pred, average="macro"))

print(
    classification_report(
        y_validation,
        val_pred,
        target_names=["negative", "neutral", "positive"],
        digits=4,
    )
)


# ---------------------------------------------------------
# Test results
# ---------------------------------------------------------

test_pred = model.predict(X_test)

print("\n--- TEST ---")
print("Accuracy:", accuracy_score(y_test, test_pred))
print("Macro F1:", f1_score(y_test, test_pred, average="macro"))

report = classification_report(
    y_test,
    test_pred,
    target_names=["negative", "neutral", "positive"],
    digits=4,
    output_dict=True,
)

print(
    classification_report(
        y_test,
        test_pred,
        target_names=["negative", "neutral", "positive"],
        digits=4,
    )
)

cm = confusion_matrix(y_test, test_pred)

print("\nConfusion matrix:")
print(cm)


# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------

predictions = test_df.copy()
predictions["tfidf_prediction"] = test_pred

predictions.to_csv(
    RESULTS_DIR / "tfidf_predictions.csv",
    index=False,
)

pd.DataFrame(report).T.to_csv(
    RESULTS_DIR / "tfidf_classification_report.csv"
)

pd.DataFrame(
    cm,
    index=["true_negative", "true_neutral", "true_positive"],
    columns=["pred_negative", "pred_neutral", "pred_positive"],
).to_csv(
    RESULTS_DIR / "tfidf_confusion_matrix.csv"
)


# ---------------------------------------------------------
# Interpretability: strongest learned features
# ---------------------------------------------------------

feature_names = vectorizer.get_feature_names_out()

label_names = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

rows = []

for class_index, class_id in enumerate(model.classes_):

    coefficients = model.coef_[class_index]

    strongest = coefficients.argsort()[-20:][::-1]

    print(f"\nTop features for {label_names[class_id]}:")

    for rank, feature_idx in enumerate(strongest, start=1):

        word = feature_names[feature_idx]
        coefficient = coefficients[feature_idx]

        print(
            f"{rank:2d}. "
            f"{word:<25} "
            f"{coefficient:.4f}"
        )

        rows.append(
            {
                "class_id": class_id,
                "class_name": label_names[class_id],
                "rank": rank,
                "feature": word,
                "coefficient": coefficient,
            }
        )

pd.DataFrame(rows).to_csv(
    RESULTS_DIR / "tfidf_top_features.csv",
    index=False,
)

print("\nTF-IDF experiment complete.")