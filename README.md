# From Words to Market Moves

Seminar project for **From Neurons to Transformers: Cognitive Principles in AI Research** at Heidelberg University.

**Author:** Federico David Macias Orozco

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


## AI Tools Declaration

AI-assisted tools were used during the development and documentation of this project:

* **Grammarly:** Used for English grammar, spelling, and language improvements.
* **ChatGPT (OpenAI):** Used for brainstorming research ideas, planning the experimental methodology, assisting with Python code development and debugging, generating visualizations, and drafting and editing parts of the seminar paper.
* **Gemini (Google):** Used to compare alternative approaches, gather ideas, and explore different perspectives on the research topic.

These tools supported the research and development process. The project includes experiments carried out using the implemented models, their recorded outputs, and subsequent analysis of the results.

