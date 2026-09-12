from datasets import load_dataset

print("SCRIPT STARTED")

dataset = load_dataset(
    "takala/financial_phrasebank",
    "sentences_75agree"
)

print(dataset)
print(dataset["train"][0])
print(dataset["train"].features)