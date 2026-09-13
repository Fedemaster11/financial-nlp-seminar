from pathlib import Path
import re

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =========================================================
# 1. Paths
# =========================================================

DATA_DIR = Path("data") / "processed"
EXTERNAL_DIR = Path("data") / "external"
RESULTS_DIR = Path("results") / "tables"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_PATH = DATA_DIR / "phrasebank_test.csv"

LM_PATH = (
    EXTERNAL_DIR
    / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
)


# =========================================================
# 2. Load frozen PhraseBank test set
# =========================================================

test_df = pd.read_csv(TEST_PATH)

print("Test rows:", len(test_df))


# =========================================================
# 3. Load Loughran-McDonald dictionary
# =========================================================

lm_df = pd.read_csv(LM_PATH)

print("\nLM dictionary rows:", len(lm_df))
print("LM dictionary columns:", len(lm_df.columns))

# The uploaded dictionary contains one missing Word entry.
lm_df = lm_df.dropna(subset=["Word"]).copy()

lm_df["Word"] = (
    lm_df["Word"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print("Dictionary rows after removing missing words:", len(lm_df))


# =========================================================
# 4. Build category lexicons
# =========================================================
#
# IMPORTANT:
# LM category values are not sentiment weights.
#
# A non-zero value means the word belongs to that category.
# The value itself commonly represents the year the category
# membership was introduced/recorded.
#
# Therefore we use > 0 rather than treating the values as scores.
# =========================================================

def build_lexicon(column_name):
    values = pd.to_numeric(
        lm_df[column_name],
        errors="coerce"
    ).fillna(0)

    return set(
        lm_df.loc[
            values > 0,
            "Word"
        ]
    )


positive_words = build_lexicon("Positive")
negative_words = build_lexicon("Negative")
uncertainty_words = build_lexicon("Uncertainty")
constraining_words = build_lexicon("Constraining")


print("\nLexicon sizes:")
print("Positive:", len(positive_words))
print("Negative:", len(negative_words))
print("Uncertainty:", len(uncertainty_words))
print("Constraining:", len(constraining_words))


# =========================================================
# 5. Tokenization
# =========================================================

TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?"
)


def tokenize(text):
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(str(text))
    ]


# =========================================================
# 6. Negation handling
# =========================================================

NEGATORS = {
    "no",
    "not",
    "never",
    "neither",
    "nor",
    "without",
    "cannot",
    "can't",
    "isn't",
    "wasn't",
    "weren't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
}


def is_negated(tokens, index, window=3):
    """
    Return True if a negator appears within the previous
    `window` tokens.

    Example:
        "not expect losses"

    If 'losses' is a negative dictionary term and 'not'
    occurs within the preceding three tokens, its polarity
    is flipped.
    """

    start = max(0, index - window)

    context = tokens[start:index]

    return any(
        token in NEGATORS
        for token in context
    )


# =========================================================
# 7. Rule-based prediction
# =========================================================

def analyze_sentence(text):

    tokens = tokenize(text)

    positive_hits = 0
    negative_hits = 0
    uncertainty_hits = 0
    constraining_hits = 0

    matched_positive = []
    matched_negative = []
    matched_uncertainty = []
    matched_constraining = []

    for i, token in enumerate(tokens):

        negated = is_negated(tokens, i)

        # ---------------------------------------------
        # Positive sentiment words
        # ---------------------------------------------

        if token in positive_words:

            if negated:
                negative_hits += 1
                matched_negative.append(
                    f"{token}[NEGATED_POSITIVE]"
                )

            else:
                positive_hits += 1
                matched_positive.append(token)

        # ---------------------------------------------
        # Negative sentiment words
        # ---------------------------------------------

        if token in negative_words:

            if negated:
                positive_hits += 1
                matched_positive.append(
                    f"{token}[NEGATED_NEGATIVE]"
                )

            else:
                negative_hits += 1
                matched_negative.append(token)

        # ---------------------------------------------
        # Linguistic diagnostic categories
        # ---------------------------------------------

        if token in uncertainty_words:
            uncertainty_hits += 1
            matched_uncertainty.append(token)

        if token in constraining_words:
            constraining_hits += 1
            matched_constraining.append(token)

    # ---------------------------------------------
    # Sentiment score
    # ---------------------------------------------

    sentiment_score = (
        positive_hits
        -
        negative_hits
    )

    # PhraseBank:
    # 0 = negative
    # 1 = neutral
    # 2 = positive

    if sentiment_score > 0:
        prediction = 2

    elif sentiment_score < 0:
        prediction = 0

    else:
        prediction = 1

    has_sentiment_coverage = (
        positive_hits + negative_hits
    ) > 0

    return {
        "prediction": prediction,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "uncertainty_hits": uncertainty_hits,
        "constraining_hits": constraining_hits,
        "sentiment_score": sentiment_score,
        "has_sentiment_coverage": has_sentiment_coverage,
        "matched_positive": "; ".join(matched_positive),
        "matched_negative": "; ".join(matched_negative),
        "matched_uncertainty": "; ".join(
            matched_uncertainty
        ),
        "matched_constraining": "; ".join(
            matched_constraining
        ),
    }


# =========================================================
# 8. Run model on frozen test set
# =========================================================

analysis = test_df["sentence"].apply(
    analyze_sentence
)

rule_results = pd.DataFrame(
    analysis.tolist()
)

test_results = pd.concat(
    [
        test_df.reset_index(drop=True),
        rule_results,
    ],
    axis=1,
)

y_true = test_results["label"]
y_pred = test_results["prediction"]


# =========================================================
# 9. Overall evaluation
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)

print("\n====================================")
print("RULE-BASED TEST RESULTS")
print("====================================")

print("\nAccuracy:")
print(round(accuracy, 4))

print("\nMacro F1:")
print(round(macro_f1, 4))

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
# 10. Confusion matrix
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2],
)

print("\nConfusion matrix:")
print(cm)


# =========================================================
# 11. Coverage analysis
# =========================================================

covered = test_results[
    "has_sentiment_coverage"
]

coverage_rate = covered.mean()

no_coverage_count = (
    ~covered
).sum()

print("\n====================================")
print("DICTIONARY COVERAGE")
print("====================================")

print(
    "Coverage rate:",
    round(coverage_rate, 4)
)

print(
    "Sentences with sentiment words:",
    covered.sum()
)

print(
    "Sentences with no sentiment words:",
    no_coverage_count
)


# =========================================================
# 12. Prediction distribution
# =========================================================
#
# Important because the LM dictionary contains many more
# negative than positive terms. We want to see whether that
# causes systematic prediction bias.
# =========================================================

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

prediction_distribution = (
    test_results["prediction"]
    .map(LABEL_NAMES)
    .value_counts()
)

print("\n====================================")
print("PREDICTION DISTRIBUTION")
print("====================================")

print(prediction_distribution)


# =========================================================
# 13. Performance conditional on dictionary coverage
# =========================================================

covered_df = test_results[
    test_results["has_sentiment_coverage"]
].copy()

if len(covered_df) > 0:

    covered_f1 = f1_score(
        covered_df["label"],
        covered_df["prediction"],
        average="macro",
        zero_division=0,
    )

    covered_accuracy = accuracy_score(
        covered_df["label"],
        covered_df["prediction"],
    )

    print("\n====================================")
    print("PERFORMANCE WHEN DICTIONARY MATCHES")
    print("====================================")

    print(
        "Covered observations:",
        len(covered_df)
    )

    print(
        "Covered accuracy:",
        round(covered_accuracy, 4)
    )

    print(
        "Covered macro F1:",
        round(covered_f1, 4)
    )


# =========================================================
# 14. Save outputs
# =========================================================

test_results.to_csv(
    RESULTS_DIR
    / "rule_based_predictions.csv",
    index=False,
)

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
    / "rule_based_confusion_matrix.csv"
)


summary = pd.DataFrame(
    [
        {
            "model": "LM rule-based",
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "coverage": coverage_rate,
            "test_size": len(test_results),
        }
    ]
)

summary.to_csv(
    RESULTS_DIR
    / "rule_based_summary.csv",
    index=False,
)


print("\nRule-based experiment complete.")