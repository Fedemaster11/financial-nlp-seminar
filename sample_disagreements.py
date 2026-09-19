from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("results") / "tables"

master = pd.read_csv(
    RESULTS_DIR / "model_comparison_master.csv"
)

RANDOM_STATE = 42


def sample_category(category, n):

    subset = master[
        master["disagreement_category"] == category
    ]

    if len(subset) <= n:
        return subset.copy()

    return subset.sample(
        n=n,
        random_state=RANDOM_STATE,
    )


# Fixed sampling plan.
#
# Rare categories are kept completely.
# Larger categories are randomly sampled with seed 42.
samples = [

    sample_category(
        "finbert_only_correct",
        6,
    ),

    sample_category(
        "tfidf_only_correct",
        6,
    ),

    sample_category(
        "rule_only_correct",
        2,
    ),

    sample_category(
        "all_wrong",
        6,
    ),

    sample_category(
        "finbert_tfidf_correct",
        4,
    ),

    sample_category(
        "finbert_rule_correct",
        3,
    ),

    sample_category(
        "tfidf_rule_correct",
        3,
    ),
]


qualitative_sample = pd.concat(
    samples,
    ignore_index=True,
)


columns = [
    "sentence",
    "label_name",

    "rule_prediction_name",
    "tfidf_prediction_name",
    "finbert_prediction_name",

    "disagreement_category",

    "negation",
    "contrast",
    "expectation",
    "uncertainty",
]


qualitative_sample = qualitative_sample[
    columns
]


qualitative_sample.to_csv(
    RESULTS_DIR
    / "qualitative_disagreement_sample.csv",
    index=False,
)


print(
    "\nQualitative sample size:",
    len(qualitative_sample)
)

print("\nCategory counts:")

print(
    qualitative_sample[
        "disagreement_category"
    ].value_counts()
)

print(
    "\nSaved to:",
    RESULTS_DIR
    / "qualitative_disagreement_sample.csv"
)
