from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("data") / "controlled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

rows = [

    # =====================================================
    # A. Comparative / quantitative financial reasoning
    # =====================================================

    {
        "id": "C1",
        "phenomenon": "comparative_financial_reasoning",
        "pair": "profit_direction",
        "variant": "increase",
        "sentence":
            "Operating profit increased from EUR 10 million to EUR 15 million.",
        "intended_label_name": "positive",
    },

    {
        "id": "C2",
        "phenomenon": "comparative_financial_reasoning",
        "pair": "profit_direction",
        "variant": "decrease",
        "sentence":
            "Operating profit decreased from EUR 15 million to EUR 10 million.",
        "intended_label_name": "negative",
    },

    {
        "id": "C3",
        "phenomenon": "comparative_financial_reasoning",
        "pair": "loss_direction",
        "variant": "increase",
        "sentence":
            "Operating loss increased from EUR 10 million to EUR 15 million.",
        "intended_label_name": "negative",
    },

    {
        "id": "C4",
        "phenomenon": "comparative_financial_reasoning",
        "pair": "loss_direction",
        "variant": "decrease",
        "sentence":
            "Operating loss decreased from EUR 15 million to EUR 10 million.",
        "intended_label_name": "positive",
    },

    {
        "id": "C5",
        "phenomenon": "comparative_financial_reasoning",
        "pair": "cost_direction",
        "variant": "increase",
        "sentence":
            "Operating costs increased from EUR 10 million to EUR 15 million.",
        "intended_label_name": "negative",
    },

    {
        "id": "C6",
        "phenomenon": "comparative_financial_reasoning",
        "pair": "cost_direction",
        "variant": "decrease",
        "sentence":
            "Operating costs decreased from EUR 15 million to EUR 10 million.",
        "intended_label_name": "positive",
    },

    {
        "id": "C7",
        "phenomenon": "comparative_financial_reasoning",
        "pair": "revenue_direction",
        "variant": "increase",
        "sentence":
            "Revenue increased from EUR 10 million to EUR 15 million.",
        "intended_label_name": "positive",
    },

    {
        "id": "C8",
        "phenomenon": "comparative_financial_reasoning",
        "pair": "revenue_direction",
        "variant": "decrease",
        "sentence":
            "Revenue decreased from EUR 15 million to EUR 10 million.",
        "intended_label_name": "negative",
    },


    # =====================================================
    # B. Negation
    # =====================================================

    {
        "id": "N1",
        "phenomenon": "negation",
        "pair": "profitability_negation",
        "variant": "affirmative",
        "sentence":
            "The company achieved profitability this year.",
        "intended_label_name": "positive",
    },

    {
        "id": "N2",
        "phenomenon": "negation",
        "pair": "profitability_negation",
        "variant": "negated",
        "sentence":
            "The company did not achieve profitability this year.",
        "intended_label_name": "negative",
    },

    {
        "id": "N3",
        "phenomenon": "negation",
        "pair": "contract_negation",
        "variant": "affirmative",
        "sentence":
            "The company secured a major contract.",
        "intended_label_name": "positive",
    },

    {
        "id": "N4",
        "phenomenon": "negation",
        "pair": "contract_negation",
        "variant": "negated",
        "sentence":
            "The company did not secure a major contract.",
        "intended_label_name": "negative",
    },


    # =====================================================
    # C. Intention versus realization
    # =====================================================

    {
        "id": "R1",
        "phenomenon": "intention_vs_realization",
        "pair": "capacity_expansion",
        "variant": "intention",
        "sentence":
            "The company plans to expand production capacity next year.",
        "intended_label_name": "neutral",
    },

    {
        "id": "R2",
        "phenomenon": "intention_vs_realization",
        "pair": "capacity_expansion",
        "variant": "realized",
        "sentence":
            "The company expanded production capacity this year.",
        "intended_label_name": "positive",
    },

    {
        "id": "R3",
        "phenomenon": "intention_vs_realization",
        "pair": "market_entry",
        "variant": "intention",
        "sentence":
            "The company is considering entering a new market.",
        "intended_label_name": "neutral",
    },

    {
        "id": "R4",
        "phenomenon": "intention_vs_realization",
        "pair": "market_entry",
        "variant": "realized",
        "sentence":
            "The company entered a new market and began commercial sales.",
        "intended_label_name": "positive",
    },


    # =====================================================
    # D. Contrast / composition
    # =====================================================

    {
        "id": "X1",
        "phenomenon": "contrast_composition",
        "pair": "revenue_profit",
        "variant": "both_positive",
        "sentence":
            "Revenue increased and operating profit increased.",
        "intended_label_name": "positive",
    },

    {
        "id": "X2",
        "phenomenon": "contrast_composition",
        "pair": "revenue_profit",
        "variant": "revenue_up_profit_down",
        "sentence":
            "Revenue increased, but operating profit decreased.",
        "intended_label_name": "neutral",
    },

    {
        "id": "X3",
        "phenomenon": "contrast_composition",
        "pair": "revenue_profit",
        "variant": "revenue_down_profit_up",
        "sentence":
            "Revenue decreased, but operating profit increased.",
        "intended_label_name": "neutral",
    },

    {
        "id": "X4",
        "phenomenon": "contrast_composition",
        "pair": "revenue_profit",
        "variant": "both_negative",
        "sentence":
            "Revenue decreased and operating profit decreased.",
        "intended_label_name": "negative",
    },


    # =====================================================
    # E. Uncertainty
    # =====================================================

    {
        "id": "U1",
        "phenomenon": "uncertainty",
        "pair": "positive_uncertainty",
        "variant": "certain",
        "sentence":
            "Operating profit will increase next year.",
        "intended_label_name": "positive",
    },

    {
        "id": "U2",
        "phenomenon": "uncertainty",
        "pair": "positive_uncertainty",
        "variant": "uncertain",
        "sentence":
            "Operating profit may increase next year.",
        "intended_label_name": "positive",
    },

    {
        "id": "U3",
        "phenomenon": "uncertainty",
        "pair": "negative_uncertainty",
        "variant": "certain",
        "sentence":
            "Operating profit will decrease next year.",
        "intended_label_name": "negative",
    },

    {
        "id": "U4",
        "phenomenon": "uncertainty",
        "pair": "negative_uncertainty",
        "variant": "uncertain",
        "sentence":
            "Operating profit may decrease next year.",
        "intended_label_name": "negative",
    },
]

df = pd.DataFrame(rows)

df["intended_label"] = (
    df["intended_label_name"]
    .map(LABELS)
)

output_file = OUTPUT_DIR / "minimal_pairs.csv"

df.to_csv(
    output_file,
    index=False
)

print(df.to_string(index=False))
print("\nSaved:", output_file)
print("Total probes:", len(df))