from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results") / "tables"

INPUT_FILE = RESULTS_DIR / "all_model_errors.csv"
OUTPUT_FILE = RESULTS_DIR / "finbert_error_coded.csv"
SUMMARY_FILE = RESULTS_DIR / "finbert_error_taxonomy_summary.csv"


# =========================================================
# 1. Load ONLY FinBERT errors
# =========================================================

df = pd.read_csv(INPUT_FILE)

errors = (
    df[df["model"] == "Finance BERT"]
    .copy()
    .reset_index(drop=True)
)

if len(errors) != 35:
    raise ValueError(
        f"Expected 35 FinBERT errors, found {len(errors)}."
    )

errors["error_direction"] = (
    errors["true_class"]
    + "_to_"
    + errors["predicted_class"]
)


# =========================================================
# 2. Manual coding of all 35 errors
# =========================================================
#
# Categories:
#
# neutral_boundary_lexical_overinterpretation
# implicit_business_event_consequence
# comparative_financial_reasoning
# future_expectation_realization
# contrast_composition
# recommendation_benchmark_semantics
#
# 'snippet' is used as a safety check so that coding cannot
# silently attach to the wrong sentence.
# =========================================================

coding = [

    # 1
    {
        "snippet": "made redundant",
        "category": "implicit_business_event_consequence",
        "explanation":
            "The negative meaning depends on recognizing layoffs and "
            "redundancies as adverse employment events.",
        "confidence": "high",
    },

    # 2
    {
        "snippet": "retained its guidance",
        "category": "contrast_composition",
        "explanation":
            "The sentence combines negative 'soft' results with retained "
            "guidance. FinBERT predicts negative rather than integrating "
            "the competing clauses into the annotated neutral meaning.",
        "confidence": "high",
    },

    # 3
    {
        "snippet": "insurance division turned",
        "category": "neutral_boundary_lexical_overinterpretation",
        "explanation":
            "The word 'profit' creates positive lexical evidence even "
            "though the dataset annotates the factual statement as neutral.",
        "confidence": "medium",
    },

    # 4
    {
        "snippet": "wants a new and better frequency",
        "category": "future_expectation_realization",
        "explanation":
            "The sentence expresses a desired future improvement rather "
            "than a realized positive outcome.",
        "confidence": "high",
    },

    # 5
    {
        "snippet": "aims to find growth outside Finland",
        "category": "future_expectation_realization",
        "explanation":
            "The positive annotation involves a growth-oriented objective, "
            "but FinBERT treats the unrealized aim as neutral.",
        "confidence": "medium",
    },

    # 6
    {
        "snippet": "principal suppliers of ICT solutions",
        "category": "implicit_business_event_consequence",
        "explanation":
            "Being selected as a principal supplier for a valuable contract "
            "is a positive commercial event whose polarity is implicit.",
        "confidence": "high",
    },

    # 7
    {
        "snippet": "expected to raise some euro300 million",
        "category": "future_expectation_realization",
        "explanation":
            "FinBERT treats an expected financing outcome as positive even "
            "though the statement is annotated neutral.",
        "confidence": "high",
    },

    # 8
    {
        "snippet": "awarded more than $ 350,000",
        "category": "neutral_boundary_lexical_overinterpretation",
        "explanation":
            "The term 'awarded' appears favorable, but here the sentence is "
            "a factual description annotated neutral.",
        "confidence": "medium",
    },

    # 9
    {
        "snippet": "expected to be on par",
        "category": "future_expectation_realization",
        "explanation":
            "The model assigns negative sentiment to an expectation of "
            "unchanged sales, although 'on par' is annotated neutral.",
        "confidence": "high",
    },

    # 10
    {
        "snippet": "help make it easier",
        "category": "neutral_boundary_lexical_overinterpretation",
        "explanation":
            "Promotional words such as 'help', 'easier', and 'enabling' "
            "produce positive sentiment for a statement annotated neutral.",
        "confidence": "high",
    },

    # 11
    {
        "snippet": "Unit costs for flight operations fell",
        "category": "comparative_financial_reasoning",
        "explanation":
            "A fall sounds negative lexically, but falling costs are "
            "economically favorable.",
        "confidence": "high",
    },

    # 12
    {
        "snippet": "was the cheapest also",
        "category": "neutral_boundary_lexical_overinterpretation",
        "explanation":
            "The model interprets 'cheapest' as favorable even though the "
            "sentence is annotated as a neutral comparative fact.",
        "confidence": "medium",
    },

    # 13
    {
        "snippet": "buy ' recommendations",
        "category": "recommendation_benchmark_semantics",
        "explanation":
            "The sentence contains explicit positive analyst recommendations, "
            "but FinBERT fails to map the domain-specific recommendation "
            "language to the positive label.",
        "confidence": "high",
    },

    # 14
    {
        "snippet": "Diluted loss per share",
        "category": "comparative_financial_reasoning",
        "explanation":
            "Loss per share falls from 0.26 to 0.15. The word 'loss' is "
            "negative, but the directional change is economically positive.",
        "confidence": "high",
    },

    # 15
    {
        "snippet": "reports a loss for the period",
        "category": "comparative_financial_reasoning",
        "explanation":
            "The company remains loss-making, but the loss narrows from "
            "1.9 million to 0.4 million, which is an improvement.",
        "confidence": "high",
    },

    # 16
    {
        "snippet": "contracts of the employees",
        "category": "implicit_business_event_consequence",
        "explanation":
            "Ending employee contracts is an adverse employment event even "
            "though explicit sentiment vocabulary is limited.",
        "confidence": "high",
    },

    # 17
    {
        "snippet": "high level of automation",
        "category": "neutral_boundary_lexical_overinterpretation",
        "explanation":
            "Positive-sounding product language such as a 'high level of "
            "automation' is interpreted positively despite a neutral label.",
        "confidence": "medium",
    },

    # 18
    {
        "snippet": "posted a net profit of 7.99",
        "category": "comparative_financial_reasoning",
        "explanation":
            "The company reports a positive profit level, but profit has "
            "declined relative to the previous year, making the sentence negative.",
        "confidence": "high",
    },

    # 19
    {
        "snippet": "most likely to remain Finnish",
        "category": "neutral_boundary_lexical_overinterpretation",
        "explanation":
            "The model assigns negative polarity to a largely factual statement; "
            "terms such as 'emergency supply' may contribute negative associations.",
        "confidence": "medium",
    },

    # 20
    {
        "snippet": "Return on investment ROI",
        "category": "comparative_financial_reasoning",
        "explanation":
            "ROI deteriorates from 43.8 percent to 4.1 percent, but FinBERT "
            "predicts positive despite the unfavorable comparison.",
        "confidence": "high",
    },

    # 21
    {
        "snippet": "market for automated liquid handling devices",
        "category": "neutral_boundary_lexical_overinterpretation",
        "explanation":
            "The model treats the statement that one market is larger than "
            "another as positive, although it is annotated neutral.",
        "confidence": "medium",
    },

    # 22
    {
        "snippet": "significant financial impact",
        "category": "neutral_boundary_lexical_overinterpretation",
        "explanation":
            "'Significant financial impact' does not specify positive or "
            "negative direction, but FinBERT assigns negative sentiment.",
        "confidence": "high",
    },

    # 23
    {
        "snippet": "financing will also be provided to expand",
        "category": "implicit_business_event_consequence",
        "explanation":
            "Financing for expansion and new business areas is annotated "
            "positive, but the model treats the planned commercial event as neutral.",
        "confidence": "medium",
    },

    
    # 24
    {
        "snippet": "putting a stake in the ground",
        "category": "neutral_boundary_lexical_overinterpretation",
        "explanation":
            "The sentence describes vendors and customers responding to a "
            "government deadline and is annotated neutral. FinBERT assigns "
            "positive sentiment to what is primarily a factual statement, "
            "possibly reflecting favorable associations with language such "
            "as 'focused' and 'meeting the deadline'.",
        "confidence": "medium",
    },
    # 25
    {
        "snippet": "Loss after taxes",
        "category": "comparative_financial_reasoning",
        "explanation":
            "The loss narrows from 2.6 million to 1.2 million. Correct sentiment "
            "requires combining the negative metric with an improving direction.",
        "confidence": "high",
    },

    # 26
    {
        "snippet": "ordered nine Airbus",
        "category": "implicit_business_event_consequence",
        "explanation":
            "A major aircraft order and becoming the lead airline are positive "
            "business developments, but their polarity is not explicitly stated.",
        "confidence": "medium",
    },

    # 27
    {
        "snippet": "plans to expand into the Moscow market",
        "category": "future_expectation_realization",
        "explanation":
            "FinBERT treats a future expansion plan as an already realized "
            "positive outcome, while the annotation is neutral.",
        "confidence": "high",
    },

    # 28
    {
        "snippet": "To be number one means creating added value",
        "category": "neutral_boundary_lexical_overinterpretation",
        "explanation":
            "Aspirational corporate language such as 'number one' and "
            "'added value' is interpreted positively despite a neutral annotation.",
        "confidence": "high",
    },

    # 29
    {
        "snippet": "reiterated its forecast",
        "category": "future_expectation_realization",
        "explanation":
            "The model interprets a reiterated sales forecast as positive "
            "even though it is primarily a neutral forecast statement.",
        "confidence": "high",
    },

    # 30
    {
        "snippet": "commercial vessels had got stuck",
        "category": "implicit_business_event_consequence",
        "explanation":
            "Understanding that vessels being stuck in ice is operationally "
            "negative requires reasoning about the consequence of the event.",
        "confidence": "high",
    },

    # 31
    {
        "snippet": "number of shares in the Swedish company will grow",
        "category": "neutral_boundary_lexical_overinterpretation",
        "explanation":
            "The word 'grow' receives positive polarity although the sentence "
            "only states a change in the number of shares.",
        "confidence": "high",
    },

    # 32
    {
        "snippet": "business which fits well",
        "category": "implicit_business_event_consequence",
        "explanation":
            "The positive meaning is implied by management's assessment that "
            "the business fits well strategically, rather than by a financial metric.",
        "confidence": "medium",
    },

    # 33
    {
        "snippet": "rejected a hostile",
        "category": "implicit_business_event_consequence",
        "explanation":
            "The annotation treats the rejection and possibility of a counteroffer "
            "as positive, while FinBERT assigns negative sentiment to the hostile "
            "takeover context.",
        "confidence": "medium",
    },

    # 34
    {
        "snippet": "expects the consolidation trend",
        "category": "future_expectation_realization",
        "explanation":
            "The sentence describes an expectation rather than a realized outcome, "
            "but FinBERT converts the prospective growth-related language into "
            "positive sentiment.",
        "confidence": "high",
    },

    # 35
    {
        "snippet": "Operating loss increased",
        "category": "comparative_financial_reasoning",
        "explanation":
            "The word 'increased' is commonly positive, but increasing an operating "
            "loss from 10.8 million to 17 million is economically negative.",
        "confidence": "high",
    },
]


# =========================================================
# 3. Safety checks + attach coding
# =========================================================

if len(coding) != len(errors):
    raise ValueError(
        f"Coding has {len(coding)} rows but errors has {len(errors)}."
    )

for i, item in enumerate(coding):
    sentence = str(errors.loc[i, "sentence"])

    if item["snippet"].lower() not in sentence.lower():
        raise ValueError(
            f"\nCoding alignment failed at row {i + 1}.\n"
            f"Expected snippet: {item['snippet']}\n"
            f"Actual sentence: {sentence}"
        )


errors["error_category"] = [
    x["category"] for x in coding
]

errors["failure_explanation"] = [
    x["explanation"] for x in coding
]

errors["interpretation_confidence"] = [
    x["confidence"] for x in coding
]


# =========================================================
# 4. Create taxonomy summary
# =========================================================

summary = (
    errors["error_category"]
    .value_counts()
    .rename_axis("error_category")
    .reset_index(name="count")
)

summary["share_of_finbert_errors"] = (
    summary["count"] / len(errors)
)


# =========================================================
# 5. Error type x error direction
# =========================================================

direction_table = pd.crosstab(
    errors["error_category"],
    errors["error_direction"],
)


# =========================================================
# 6. Save
# =========================================================

errors.to_csv(
    OUTPUT_FILE,
    index=False,
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)

direction_table.to_csv(
    RESULTS_DIR / "finbert_error_category_by_direction.csv"
)


# =========================================================
# 7. Print
# =========================================================

print("\n====================================")
print("FINBERT ERROR TAXONOMY")
print("====================================")

print(summary.to_string(index=False))

print("\n====================================")
print("ERROR CATEGORY x DIRECTION")
print("====================================")

print(direction_table)

print("\n====================================")
print("CONFIDENCE")
print("====================================")

print(
    errors["interpretation_confidence"]
    .value_counts()
)

print("\nTotal FinBERT errors:", len(errors))

print("\nSaved:")
print("finbert_error_coded.csv")
print("finbert_error_taxonomy_summary.csv")
print("finbert_error_category_by_direction.csv")