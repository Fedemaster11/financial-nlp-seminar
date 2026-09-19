from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)


# =========================================================
# Configuration
# =========================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results") / "tables"
MODEL_DIR = Path("models") / "finbert_best"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = (
    DATA_DIR / "processed" / "phrasebank_train.csv"
)

PROBE_FILE = (
    DATA_DIR / "controlled" / "minimal_pairs.csv"
)

LM_FILE = (
    DATA_DIR
    / "external"
    / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
)

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# =========================================================
# Load data
# =========================================================

train_df = pd.read_csv(TRAIN_FILE)
probes = pd.read_csv(PROBE_FILE)

print("Training observations:", len(train_df))
print("Controlled probes:", len(probes))


# =========================================================
# 1. RULE-BASED MODEL
# =========================================================

lm = pd.read_csv(LM_FILE)

lm = lm.dropna(
    subset=["Word"]
).copy()

positive_words = set(
    lm.loc[
        lm["Positive"].fillna(0) > 0,
        "Word"
    ]
    .astype(str)
    .str.lower()
)

negative_words = set(
    lm.loc[
        lm["Negative"].fillna(0) > 0,
        "Word"
    ]
    .astype(str)
    .str.lower()
)

NEGATORS = {
    "not",
    "no",
    "never",
    "without",
    "neither",
    "nor",
    "cannot",
    "can't",
    "won't",
    "didn't",
    "doesn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
}


def tokenize(text):
    return re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?",
        str(text).lower(),
    )


def rule_predict(text):

    tokens = tokenize(text)

    positive_hits = 0
    negative_hits = 0

    for i, token in enumerate(tokens):

        is_positive = token in positive_words
        is_negative = token in negative_words

        if not is_positive and not is_negative:
            continue

        start = max(0, i - 3)

        preceding = tokens[start:i]

        negated = any(
            word in NEGATORS
            for word in preceding
        )

        if is_positive:

            if negated:
                negative_hits += 1
            else:
                positive_hits += 1

        if is_negative:

            if negated:
                positive_hits += 1
            else:
                negative_hits += 1

    score = positive_hits - negative_hits

    if score > 0:
        prediction = 2
    elif score < 0:
        prediction = 0
    else:
        prediction = 1

    return (
        prediction,
        score,
        positive_hits,
        negative_hits,
    )


rule_outputs = probes["sentence"].apply(
    rule_predict
)

probes["rule_prediction"] = [
    x[0] for x in rule_outputs
]

probes["rule_score"] = [
    x[1] for x in rule_outputs
]

probes["rule_positive_hits"] = [
    x[2] for x in rule_outputs
]

probes["rule_negative_hits"] = [
    x[3] for x in rule_outputs
]


# =========================================================
# 2. TF-IDF + LOGISTIC REGRESSION
# =========================================================
#
# We refit the SAME baseline on the same frozen training set.
# No test data or controlled probes are used for training.
# =========================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(
    train_df["sentence"]
)

y_train = train_df["label"].astype(int)

tfidf_model = LogisticRegression(
    max_iter=3000,
    solver="lbfgs",
    random_state=42,
)

tfidf_model.fit(
    X_train,
    y_train,
)

X_probe = vectorizer.transform(
    probes["sentence"]
)

tfidf_predictions = (
    tfidf_model.predict(X_probe)
)

tfidf_probabilities = (
    tfidf_model.predict_proba(X_probe)
)

probes["tfidf_prediction"] = (
    tfidf_predictions
)

probes["tfidf_confidence"] = (
    tfidf_probabilities.max(axis=1)
)

for label, name in LABEL_NAMES.items():

    class_index = list(
        tfidf_model.classes_
    ).index(label)

    probes[
        f"tfidf_prob_{name}"
    ] = tfidf_probabilities[
        :, class_index
    ]


# =========================================================
# 3. FINBERT
# =========================================================
#
# Load the already fine-tuned best checkpoint.
# NO new training occurs here.
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("FinBERT device:", device)

tokenizer = BertTokenizer.from_pretrained(
    MODEL_DIR
)

finbert = (
    BertForSequenceClassification
    .from_pretrained(MODEL_DIR)
)

finbert.to(device)
finbert.eval()

encoded = tokenizer(
    probes["sentence"].tolist(),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

encoded = {
    key: value.to(device)
    for key, value in encoded.items()
}

with torch.no_grad():

    outputs = finbert(**encoded)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1,
    )

finbert_probabilities = (
    probabilities
    .cpu()
    .numpy()
)

finbert_predictions = (
    finbert_probabilities
    .argmax(axis=1)
)

probes["finbert_prediction"] = (
    finbert_predictions
)

probes["finbert_confidence"] = (
    finbert_probabilities.max(axis=1)
)

for label, name in LABEL_NAMES.items():

    probes[
        f"finbert_prob_{name}"
    ] = finbert_probabilities[
        :, label
    ]


# =========================================================
# 4. Readable prediction names
# =========================================================

for model in [
    "rule",
    "tfidf",
    "finbert",
]:

    probes[
        f"{model}_prediction_name"
    ] = (
        probes[
            f"{model}_prediction"
        ]
        .map(LABEL_NAMES)
    )


# =========================================================
# 5. Alignment with researcher-defined target
# =========================================================
#
# IMPORTANT:
# These are NOT PhraseBank gold labels.
#
# They are semantic targets defined BEFORE predictions
# for this controlled diagnostic experiment.
# =========================================================

for model in [
    "rule",
    "tfidf",
    "finbert",
]:

    probes[
        f"{model}_aligned"
    ] = (
        probes[
            f"{model}_prediction"
        ]
        == probes["intended_label"]
    )


# Probability assigned to intended class
def intended_probability(
    row,
    model,
):

    label_name = (
        row["intended_label_name"]
    )

    return row[
        f"{model}_prob_{label_name}"
    ]


probes[
    "tfidf_intended_probability"
] = probes.apply(
    lambda row:
        intended_probability(
            row,
            "tfidf",
        ),
    axis=1,
)

probes[
    "finbert_intended_probability"
] = probes.apply(
    lambda row:
        intended_probability(
            row,
            "finbert",
        ),
    axis=1,
)


# =========================================================
# 6. Summary by phenomenon
# =========================================================

summary_rows = []

for phenomenon, group in probes.groupby(
    "phenomenon"
):

    for model in [
        "rule",
        "tfidf",
        "finbert",
    ]:

        summary_rows.append({
            "phenomenon":
                phenomenon,

            "model":
                model,

            "n":
                len(group),

            "aligned":
                group[
                    f"{model}_aligned"
                ].sum(),

            "alignment_rate":
                group[
                    f"{model}_aligned"
                ].mean(),
        })


summary = pd.DataFrame(
    summary_rows
)


# =========================================================
# 7. Save
# =========================================================

probes.to_csv(
    RESULTS_DIR
    / "minimal_pair_results.csv",
    index=False,
)

summary.to_csv(
    RESULTS_DIR
    / "minimal_pair_summary.csv",
    index=False,
)


# =========================================================
# 8. Print clean output
# =========================================================

display_columns = [
    "id",
    "phenomenon",
    "sentence",
    "intended_label_name",
    "rule_prediction_name",
    "tfidf_prediction_name",
    "finbert_prediction_name",
]

print("\n====================================")
print("CONTROLLED MINIMAL-PAIR RESULTS")
print("====================================")

print(
    probes[
        display_columns
    ].to_string(index=False)
)


print("\n====================================")
print("ALIGNMENT BY PHENOMENON")
print("====================================")

print(
    summary.to_string(index=False)
)


print("\n====================================")
print("UNCERTAINTY PROBES")
print("====================================")

uncertainty = probes[
    probes["phenomenon"]
    == "uncertainty"
]

print(
    uncertainty[
        [
            "id",
            "sentence",
            "intended_label_name",
            "tfidf_prediction_name",
            "tfidf_intended_probability",
            "finbert_prediction_name",
            "finbert_intended_probability",
        ]
    ].to_string(index=False)
)


print("\nSaved:")
print("minimal_pair_results.csv")
print("minimal_pair_summary.csv")