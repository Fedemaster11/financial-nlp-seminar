# From Words to Market Moves

Seminar project for **From Neurons to Transformers: Cognitive Principles in AI Research** at Heidelberg University.

**Author:** Federico David Macias Orozco  
**Matriculation No.:** 4730600

## Project

This project compares three approaches to financial sentiment classification:

- Rule-based model using the Loughran–McDonald financial dictionary
- TF-IDF with logistic regression
- Finance BERT fine-tuned on Financial PhraseBank

The main goal is to compare how these models represent and interpret financial language, not only how accurate they are.

## Dataset

Financial PhraseBank (`sentences_75agree`)

- 2,413 training examples
- 517 validation examples
- 518 test examples

## Main Results

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Rule-based | 64.86% | 48.42% |
| TF-IDF + Logistic Regression | 81.85% | 73.44% |
| Finance BERT | 93.24% | 91.28% |

The project also includes error analysis, model disagreement analysis, linguistic subsets, and controlled diagnostic probes.

## Main Scripts

- `prepare_data.py`
- `train_rule_based.py`
- `train_tfidf.py`
- `train_finbert.py`
- `analyze_models.py`
- `analyze_errors_by_class.py`
- `create_minimal_pairs.py`
- `analyze_minimal_pairs.py`
- `make_figures.py`

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run the scripts from the project directory.

The final seminar paper is stored in the `paper/` folder.
