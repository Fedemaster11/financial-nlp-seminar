from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results") / "tables"

input_file = RESULTS_DIR / "qualitative_disagreement_sample.csv"
output_file = RESULTS_DIR / "qualitative_disagreement_coded.csv"

df = pd.read_csv(input_file)

# Protect against accidentally coding a different sample/order.
if len(df) != 30:
    raise ValueError(
        f"Expected 30 qualitative examples, found {len(df)}."
    )


# =========================================================
# Manual qualitative coding
# =========================================================
#
# These codes describe the main phenomenon/failure mechanism
# visible in each disagreement example.
#
# The coding is interpretive qualitative analysis, not a new
# ground-truth label.
# =========================================================

primary_phenomena = [
    # 1
    "quantitative_directionality",

    # 2
    "evaluative_business_language",

    # 3
    "business_event_semantics",

    # 4
    "business_event_semantics",

    # 5
    "quantitative_directionality",

    # 6
    "quantitative_directionality",

    # 7
    "future_intention_vs_realization",

    # 8
    "surface_lexical_polarity",

    # 9
    "comparative_financial_reasoning",

    # 10
    "comparative_financial_reasoning",

    # 11
    "comparative_financial_reasoning",

    # 12
    "comparative_financial_reasoning",

    # 13
    "expectation_vs_realization",

    # 14
    "comparative_financial_reasoning",

    # 15
    "implicit_business_evaluation",

    # 16
    "event_consequence_knowledge",

    # 17
    "event_consequence_knowledge",

    # 18
    "business_event_semantics",

    # 19
    "event_consequence_knowledge",

    # 20
    "business_event_semantics",

    # 21
    "lexical_coverage",

    # 22
    "lexical_coverage",

    # 23
    "lexical_coverage",

    # 24
    "lexical_coverage",

    # 25
    "contextual_financial_transition",

    # 26
    "evaluative_business_language",

    # 27
    "benchmark_expectation_semantics",

    # 28
    "surface_lexical_polarity",

    # 29
    "future_intention_vs_realization",

    # 30
    "contrast_composition",
]


failure_explanations = [

    # 1
    (
        "The sentence expresses negative sentiment through a measured "
        "decrease in readership. FinBERT captures the direction of the "
        "change, while the rule and TF-IDF models return neutral."
    ),

    # 2
    (
        "Positive meaning is expressed through the broader evaluation "
        "'supports ... extremely well' rather than a simple financial "
        "change word. FinBERT captures this evaluative context while "
        "the simpler models return neutral."
    ),

    # 3
    (
        "The negative meaning follows from the combination of tougher "
        "competition and substantially reduced demand. FinBERT captures "
        "the adverse business context while the simpler models do not."
    ),

    # 4
    (
        "The positive label depends on interpreting fleet expansion and "
        "new tanker orders as a favorable business-development event. "
        "FinBERT captures this event-level meaning while the simpler "
        "models classify it as neutral."
    ),

    # 5
    (
        "The key signal is a 40 percent decrease in sales. FinBERT "
        "correctly interprets the quantitative direction as negative; "
        "TF-IDF returns neutral and the rule model produces the wrong polarity."
    ),

    # 6
    (
        "Growth in revenue passenger kilometres and passenger numbers "
        "indicates positive performance. FinBERT combines these quantitative "
        "changes correctly while the simpler models remain neutral."
    ),

    # 7
    (
        "The sentence describes something the company wants rather than "
        "an achieved improvement. The words 'new' and 'better' appear to "
        "pull the rule model and FinBERT toward positive sentiment, while "
        "TF-IDF correctly preserves the neutral label."
    ),

    # 8
    (
        "The sentence contains favorable promotional language such as "
        "'help', 'easier', and 'enabling', but the annotated statement is "
        "neutral. The rule model and FinBERT appear to overinterpret this "
        "positive surface language."
    ),

    # 9
    (
        "The word 'loss' is negative in isolation, but loss per share "
        "improves from 0.26 to 0.15. The rule model and FinBERT appear "
        "dominated by the negative concept 'loss', whereas TF-IDF correctly "
        "captures the favorable comparison."
    ),

    # 10
    (
        "The company still reports a loss, but the loss narrows from "
        "1.9 million to 0.4 million. Correct polarity therefore requires "
        "reasoning jointly about the financial metric and the direction "
        "of change. TF-IDF succeeds while FinBERT and the rule model do not."
    ),

    # 11
    (
        "Return on investment falls sharply from 43.8 percent to 4.1 percent. "
        "TF-IDF correctly recognizes the negative comparison, while FinBERT "
        "assigns positive sentiment despite the deterioration."
    ),

    # 12
    (
        "The loss after taxes narrows from 2.6 million to 1.2 million, "
        "which is favorable despite the repeated negative word 'loss'. "
        "TF-IDF captures the comparative meaning while FinBERT and the "
        "rule model classify the sentence as negative."
    ),

    # 13
    (
        "The sentence reports an expectation that an existing consolidation "
        "trend will continue rather than a realized positive outcome. "
        "TF-IDF and FinBERT interpret the expectation as positive, while "
        "the rule model preserves the annotated neutral sentiment."
    ),

    # 14
    (
        "An operating loss increases from 10.8 million to 17 million. "
        "The word 'increased' is normally associated with positive growth, "
        "but here increasing a loss is negative. TF-IDF and FinBERT fail "
        "to combine the metric identity with the direction of change."
    ),

    # 15
    (
        "Positive sentiment is implied by the judgment that the acquired "
        "business 'fits well' into the company. None of the models captures "
        "this relatively implicit business evaluation."
    ),

    # 16
    (
        "Negative sentiment follows from understanding that commercial "
        "vessels becoming stuck in thick ice is an adverse operational event. "
        "All three models return neutral, suggesting difficulty with sentiment "
        "that depends on event consequences rather than explicit polarity words."
    ),

    # 17
    (
        "Redundancies and temporary layoffs are adverse employment events, "
        "but all three systems classify the sentence as neutral. The polarity "
        "depends on understanding the real-world consequence of the event."
    ),

    # 18
    (
        "The positive annotation depends on interpreting a major aircraft "
        "order and lead-airline status as favorable business developments. "
        "All three models remain neutral, indicating difficulty with implicit "
        "event-level business sentiment."
    ),

    # 19
    (
        "Ending employee contracts is an adverse employment event even "
        "without explicit sentiment vocabulary. All three models return "
        "neutral, suggesting limited event-consequence understanding."
    ),

    # 20
    (
        "Being chosen as a principal supplier for a valuable multi-year "
        "contract is a positive commercial event. All three systems classify "
        "the sentence as neutral, failing to infer sentiment from the business event."
    ),

    # 21
    (
        "Winning a large contract is explicitly favorable, and both learned "
        "models classify it correctly. The rule-based system returns neutral, "
        "consistent with limited dictionary coverage for this construction."
    ),

    # 22
    (
        "The decrease in net sales is correctly recognized as negative by "
        "TF-IDF and FinBERT. The rule model returns neutral, indicating that "
        "its fixed lexicon does not reliably cover this wording."
    ),

    # 23
    (
        "Revenue growth is correctly classified as positive by both learned "
        "models. The rule system returns neutral, illustrating the limited "
        "coverage of the fixed financial dictionary in this sentence."
    ),

    # 24
    (
        "The phrase 'up from' expresses a favorable increase in net sales. "
        "TF-IDF and FinBERT capture this relation, while the rule system "
        "returns neutral because its lexical rules do not represent the "
        "full comparative construction."
    ),

    # 25
    (
        "The phrase 'swung into profit' describes a transition from a worse "
        "financial state to a profitable one. FinBERT and the rule model "
        "capture the positive meaning, while TF-IDF fails to recognize the "
        "multiword financial transition."
    ),

    # 26
    (
        "Positive sentiment is conveyed through explicitly evaluative language "
        "such as 'excited', 'proud', and 'Product of the Year'. FinBERT and "
        "the rule model capture this evaluation, while TF-IDF returns neutral."
    ),

    # 27
    (
        "The positive meaning depends on financial benchmark language: "
        "'good first quarter' and performance 'above or in line with consensus'. "
        "FinBERT and the rule system classify it correctly, while TF-IDF "
        "misses the domain-specific evaluative combination."
    ),

    # 28
    (
        "The phrase 'significant financial impact' does not specify whether "
        "the impact is positive or negative, so the annotation is neutral. "
        "FinBERT appears to infer negative polarity from the word 'impact', "
        "while the other models remain neutral."
    ),

    # 29
    (
        "The sentence describes a future plan to expand rather than a realized "
        "improvement. FinBERT treats expansion language as positive, whereas "
        "the rule and TF-IDF models correctly retain the neutral annotation."
    ),

    # 30
    (
        "The sentence contains competing signals: the results are described "
        "as 'soft', but the company retained its guidance. The human label "
        "is neutral, suggesting the clauses offset one another. FinBERT "
        "overweights the negative evaluation, while the simpler models "
        "produce the neutral label."
    ),
]


interpretation_confidence = [
    "high",    # 1
    "high",    # 2
    "high",    # 3
    "medium",  # 4
    "high",    # 5
    "high",    # 6
    "high",    # 7
    "medium",  # 8
    "high",    # 9
    "high",    # 10
    "high",    # 11
    "high",    # 12
    "high",    # 13
    "high",    # 14
    "medium",  # 15
    "high",    # 16
    "high",    # 17
    "medium",  # 18
    "high",    # 19
    "medium",  # 20
    "high",    # 21
    "high",    # 22
    "high",    # 23
    "high",    # 24
    "high",    # 25
    "high",    # 26
    "medium",  # 27
    "medium",  # 28
    "high",    # 29
    "high",    # 30
]


# Sanity checks
assert len(primary_phenomena) == 30
assert len(failure_explanations) == 30
assert len(interpretation_confidence) == 30


df["primary_phenomenon"] = primary_phenomena
df["failure_explanation"] = failure_explanations
df["interpretation_confidence"] = interpretation_confidence


df.to_csv(
    output_file,
    index=False
)

print("\nCreated coded qualitative dataset:")
print(output_file)

print("\nRows:", len(df))

print("\nPrimary phenomenon counts:")
print(
    df["primary_phenomenon"]
    .value_counts()
)

print("\nConfidence counts:")
print(
    df["interpretation_confidence"]
    .value_counts()
)