# Mini-BERT: BERT Pretraining from Scratch

An educational implementation of the BERT pretraining architecture using PyTorch. This project implements the core components of BERT, including Transformer encoder layers, Masked Language Modeling (MLM), and Next Sentence Prediction (NSP), without relying on a pre-trained BERT model.

The objective of this project is to understand the architecture, training objectives, and implementation details behind BERT through a compact, independently implemented model.

---

## Project Overview

BERT (Bidirectional Encoder Representations from Transformers) is a Transformer-based language representation model introduced by Google Research.

This project implements a smaller version of the BERT pretraining pipeline to study:

- Tokenization using a custom WordPiece tokenizer.
- Token, positional, and segment embeddings.
- Multi-head self-attention.
- Transformer encoder blocks.
- Masked Language Modeling.
- Next Sentence Prediction.
- Dynamic masking during training.
- Model training and validation.
- Qualitative evaluation of masked token predictions.

The implementation is designed for educational and research-oriented exploration rather than production-scale language modeling.

---

## Original Paper

**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**

Authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

Paper:
https://arxiv.org/abs/1810.04805

The implementation is inspired by the architecture and pretraining objectives described in the original BERT paper.

---

## Dataset

The project uses the **WikiText-103 dataset**, accessed through the Hugging Face Datasets library.

Dataset source:
https://huggingface.co/datasets/Salesforce/wikitext

The dataset contains Wikipedia-derived text and is used to construct sentence pairs for BERT-style pretraining.

---

### Dataset Processing

The preprocessing pipeline includes:

1. Loading the WikiText-103 raw dataset.
2. Selecting a manageable subset of the dataset for experimentation.
3. Splitting the text into documents and sentences.
4. Training a custom WordPiece tokenizer.
5. Generating positive and negative sentence pairs for NSP.
6. Adding special tokens such as `[CLS]` and `[SEP]`.
7. Applying dynamic masking for the MLM objective.
8. Padding or truncating sequences to a maximum length of 128 tokens.

The implementation uses a vocabulary size of approximately 8,000 tokens.

> Note: The dataset subset and model size are considerably smaller than those used in the original BERT paper.

---

## Model Configuration

The project uses a compact BERT-style architecture.

| Configuration | Value |
|---|---:|
| Hidden size | 256 |
| Number of Transformer layers | 4 |
| Number of attention heads | 4 |
| Intermediate feed-forward size | 1,024 |
| Maximum sequence length | 128 |
| Vocabulary size | Approximately 8,000 |
| Dropout | 0.1 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Optimizer | AdamW |
| Training epochs | 6 |

The model includes:

- Token embeddings
- Positional embeddings
- Segment embeddings
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Residual connections
- GELU activation
- MLM prediction head
- NSP classification head

---

## Pretraining Objectives

### 1. Masked Language Modeling

A proportion of input tokens is masked, and the model is trained to predict the original tokens.

The masking strategy follows the BERT-style approach:

- 80% of selected tokens are replaced with `[MASK]`.
- 10% are replaced with a random token.
- 10% remain unchanged.

Special tokens are excluded from the masking process.

### 2. Next Sentence Prediction

The model receives two sentences and predicts whether the second sentence logically follows the first sentence.

- Positive example: The second sentence follows the first sentence in the original text.
- Negative example: The second sentence is sampled from another location.

The NSP objective is implemented as a binary classification task.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/mini-bert-from-scratch.git
cd mini-bert-from-scratch
```

Replace `<your-username>` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux/macOS

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebook

Open the notebook using Jupyter:

```bash
jupyter notebook
```

Alternatively, upload the notebook to Google Colab and execute the cells sequentially.

The notebook handles:

- Dataset loading
- Tokenizer training
- Dataset preprocessing
- Model initialization
- Training
- Validation
- Checkpoint saving
- Qualitative evaluation

---

## Results

The model was trained for six epochs using a compact dataset subset and a reduced BERT-style architecture.

### Final Evaluation Results

| Metric | Result |
|---|---:|
| Validation loss | 4.9673 |
| MLM loss | 4.4275 |
| NSP loss | 0.5398 |
| NSP accuracy | 73.52% |

### Training Progress

| Epoch | Training Loss | Validation Loss |
|---:|---:|---:|
| 1 | 6.1429 | 5.9028 |
| 2 | 5.9098 | 5.6396 |
| 3 | 5.6850 | 5.4246 |
| 4 | 5.4927 | 5.2530 |
| 5 | 5.3310 | 5.1023 |
| 6 | 5.1904 | 4.9624 |

The training and validation losses decreased consistently throughout the six training epochs.

The NSP accuracy reached approximately 73.52% on the validation set. This indicates that the model learned patterns useful for distinguishing the constructed positive and negative sentence pairs.

---

### Qualitative Evaluation

The model was also evaluated by masking tokens in sample sentences and observing its predictions.

The results showed that the model could produce context-related predictions, although it frequently predicted common tokens and subword units. This behavior is expected given the compact model architecture, limited vocabulary, dataset subset, and short training duration.

## Comparison with the Original BERT Paper

This project is an educational implementation inspired by the original BERT paper and is not an exact reproduction of the reported experiments.

| Component | Original BERT | This Project |
|---|---|---|
| Model scale | Large-scale Transformer models | Compact Transformer model |
| Hidden size | 768 for BERT Base | 256 |
| Transformer layers | 12 for BERT Base | 4 |
| Attention heads | 12 for BERT Base | 4 |
| Training data | Large-scale BookCorpus and Wikipedia data | Subset of WikiText-103 |
| Vocabulary | Approximately 30,000 WordPiece tokens | Approximately 8,000 tokens |
| Sequence length | Up to 512 tokens | 128 tokens |
| Pretraining objectives | MLM and NSP | MLM and NSP |
| Training scale | Large-scale pretraining | Small-scale educational experiment |
| Evaluation | GLUE, SQuAD, and other downstream tasks | Validation loss, NSP accuracy, and qualitative MLM predictions |

### Key Differences

1. **Model Capacity:** The implemented model is significantly smaller than BERT Base.
2. **Training Data:** The project uses a limited subset of WikiText-103 rather than the larger corpus used in the original research.
3. **Training Duration:** The model is trained for six epochs in a resource-constrained environment.
4. **Evaluation:** The project focuses on pretraining metrics and qualitative predictions rather than downstream benchmark performance.
5. **Implementation Scope:** The project prioritizes understanding the architecture and training process over reproducing the original results.

Therefore, the reported results should not be directly compared with the performance numbers from the original BERT paper.

---

## What I Learned

This project helped develop a practical understanding of Transformer-based language model pretraining.

### Technical Learnings

- Understanding the components of the Transformer encoder architecture.
- Implementing multi-head self-attention from scratch.
- Understanding query, key, and value projections.
- Implementing residual connections and layer normalization.
- Understanding the role of positional and segment embeddings.
- Implementing dynamic token masking for MLM.
- Creating positive and negative sentence pairs for NSP.
- Building a custom WordPiece tokenizer.
- Designing a PyTorch dataset and data-loading pipeline.
- Implementing a combined MLM and NSP training objective.
- Monitoring training and validation losses.
- Evaluating model behavior using qualitative predictions.

---

### Research-Oriented Learnings

- Understanding the difference between architectural implementation and experimental reproduction.
- Observing the impact of model size and dataset scale on language model performance.
- Interpreting training and validation metrics.
- Identifying limitations in small-scale pretraining experiments.
- Understanding the importance of controlled evaluation and downstream task validation.
- Recognizing the challenges involved in reproducing results from large-scale research papers using limited computational resources.

---

## Limitations

The current implementation has several limitations:

- The model is substantially smaller than the original BERT models.
- Training is performed on a limited dataset subset.
- The vocabulary size is reduced.
- The training duration is relatively short.
- Evaluation is limited to pretraining metrics and qualitative examples.
- No downstream fine-tuning experiments have been performed.
- The implementation has not been evaluated on benchmarks such as GLUE or SQuAD.
- The results are intended for educational analysis rather than production use.

---

## Future Improvements

Potential future improvements include:

1. Training on a larger and more diverse corpus.
2. Increasing the vocabulary size.
3. Increasing model depth, hidden size, and attention heads.
4. Implementing a more comprehensive learning-rate schedule.
5. Training for a longer duration.
6. Improving dataset construction and document-level train-validation separation.
7. Adding more systematic MLM evaluation with controlled masking.
8. Fine-tuning the pretrained model on downstream NLP tasks.
9. Evaluating sentence representations on classification tasks.
10. Comparing the implementation with established BERT checkpoints.
11. Experimenting with different masking strategies.
12. Investigating the impact of removing or modifying the NSP objective.

---

## Repository Structure

```text
mini-bert-from-scratch/
│
├── notebooks/
│   └── mini_bert_pretraining.ipynb
│
├── README.md
├── summary.md
├── requirements.txt
└── .gitignore
```
