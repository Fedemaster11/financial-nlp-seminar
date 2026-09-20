"""
Recreate the six figures used in the seminar paper:

    From Words to Market Moves:
    Comparing Rule-Based, Statistical, and Transformer Models
    for Financial Language

The numerical values below are the frozen results reported in the paper.
Running this script regenerates both PNG and PDF versions in:

    results/figures/

Usage:
    python make_figures.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


OUT = Path("results") / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, stem):
    """Save a figure in both PNG and vector PDF form."""
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 1: Overall held-out performance
# ---------------------------------------------------------------------
models = ["Rule-based", "TF-IDF + LR", "Finance BERT"]
accuracy = np.array([64.8649, 81.8533, 93.2432])
macro_f1 = np.array([48.4181, 73.4440, 91.2753])

x = np.arange(len(models))
width = 0.36

fig, ax = plt.subplots(figsize=(9, 6))
bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy")
bars2 = ax.bar(x + width / 2, macro_f1, width, label="Macro-F1")

ax.set_title("Held-out Financial PhraseBank Performance")
ax.set_ylabel("Score (%)")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 100)
ax.legend()

for bars in (bars1, bars2):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.2,
            f"{h:.1f}",
            ha="center",
            va="bottom",
        )

save(fig, "01_overall_model_performance")


# ---------------------------------------------------------------------
# Figure 2: Recall by true sentiment class
# ---------------------------------------------------------------------
classes = ["Negative", "Neutral", "Positive"]

rule_recall = np.array([33.33, 89.75, 19.55])
tfidf_recall = np.array([47.62, 98.45, 57.89])
bert_recall = np.array([90.48, 94.41, 91.73])

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(x - width, rule_recall, width, label="Rule-based")
ax.bar(x, tfidf_recall, width, label="TF-IDF + LR")
ax.bar(x + width, bert_recall, width, label="Finance BERT")

ax.set_title("Recall by True Sentiment Class")
ax.set_ylabel("Recall (%)")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 100)
ax.legend()

save(fig, "02_class_recall")


# ---------------------------------------------------------------------
# Figure 3: Controlled diagnostic probe target alignment
# ---------------------------------------------------------------------
phenomena = [
    "Metric + direction",
    "Negation",
    "Intention vs realization",
    "Contrast",
    "Uncertainty",
]

rule_alignment = np.array([12.5, 50.0, 50.0, 50.0, 0.0])
tfidf_alignment = np.array([50.0, 25.0, 50.0, 75.0, 50.0])
bert_alignment = np.array([50.0, 75.0, 75.0, 50.0, 100.0])

x = np.arange(len(phenomena))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, rule_alignment, width, label="Rule-based")
ax.bar(x, tfidf_alignment, width, label="TF-IDF + LR")
ax.bar(x + width, bert_alignment, width, label="Finance BERT")

ax.set_title("Controlled Diagnostic Probe Results")
ax.set_ylabel("Target alignment (%)")
ax.set_xticks(x)
ax.set_xticklabels(phenomena, rotation=18, ha="right")
ax.set_ylim(0, 100)
ax.legend()

save(fig, "03_minimal_pair_alignment")


# ---------------------------------------------------------------------
# Figure 4: Taxonomy of all 35 Finance BERT test errors
# ---------------------------------------------------------------------
error_categories = [
    "Neutral-boundary / lexical overinterpretation",
    "Implicit business-event consequence",
    "Future / expectation / realization",
    "Comparative financial reasoning",
    "Contrast / composition",
    "Recommendation / benchmark semantics",
]
error_counts = np.array([11, 8, 7, 7, 1, 1])

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(error_categories, error_counts)

ax.set_title("Taxonomy of All 35 Finance BERT Test Errors")
ax.set_xlabel("Number of Finance BERT errors")
ax.invert_yaxis()

for bar, value in zip(bars, error_counts):
    ax.text(
        value + 0.15,
        bar.get_y() + bar.get_height() / 2,
        str(value),
        va="center",
    )

save(fig, "04_finbert_error_taxonomy")


# ---------------------------------------------------------------------
# Figure 5: Model agreement/disagreement patterns
# ---------------------------------------------------------------------
patterns = [
    "All correct",
    "FinBERT + TF-IDF correct",
    "FinBERT only correct",
    "FinBERT + Rule correct",
    "TF-IDF + Rule correct",
    "All wrong",
    "TF-IDF only correct",
    "Rule only correct",
]
pattern_counts = np.array([285, 118, 46, 34, 15, 12, 6, 2])

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(patterns, pattern_counts)

ax.set_title("Model Agreement and Disagreement Patterns (n = 518)")
ax.set_xlabel("Number of test sentences")
ax.invert_yaxis()

for bar, value in zip(bars, pattern_counts):
    ax.text(
        value + 3,
        bar.get_y() + bar.get_height() / 2,
        str(value),
        va="center",
    )

save(fig, "05_disagreement_patterns")


# ---------------------------------------------------------------------
# Figure 6: Exploratory linguistic subset accuracy
# ---------------------------------------------------------------------
subsets = ["Negation", "Contrast", "Expectation", "Uncertainty"]

rule_subset = np.array([58.82, 58.82, 62.50, 84.62])
tfidf_subset = np.array([88.24, 82.35, 70.83, 84.62])
bert_subset = np.array([100.0, 94.12, 75.0, 84.62])

x = np.arange(len(subsets))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, rule_subset, width, label="Rule-based")
ax.bar(x, tfidf_subset, width, label="TF-IDF + LR")
ax.bar(x + width, bert_subset, width, label="Finance BERT")

ax.set_title("Exploratory Linguistic Subset Performance")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels(subsets)
ax.set_ylim(0, 100)
ax.legend()

save(fig, "06_linguistic_subset_accuracy")


print(f"Recreated six paper figures in: {OUT.resolve()}")
