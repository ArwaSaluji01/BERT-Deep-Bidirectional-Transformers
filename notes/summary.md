# Project Summary: Mini-BERT from Scratch

## Overview

This project implements a compact BERT-style language model from scratch using PyTorch. The primary objective was to understand the internal architecture and pretraining process of BERT rather than using an existing pretrained model.

The implementation includes Transformer encoder layers, multi-head self-attention, token and positional embeddings, segment embeddings, Masked Language Modeling, and Next Sentence Prediction.

## Research Foundation

The project is based on the concepts introduced in the original paper:

**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**

Paper: https://arxiv.org/abs/1810.04805

The implementation follows the main BERT pretraining objectives while using a smaller model and a limited dataset subset for practical experimentation.

## Implementation

The model was developed using PyTorch and trained on a subset of WikiText-103.

Key implementation components include:

- Custom WordPiece tokenizer.
- Token, positional, and segment embeddings.
- Multi-head self-attention.
- Transformer encoder blocks.
- Dynamic Masked Language Modeling.
- Next Sentence Prediction.
- Joint MLM and NSP training.
- Validation and qualitative model evaluation.

The model configuration consists of four Transformer layers, four attention heads, a hidden size of 256, an intermediate size of 1,024, and a maximum sequence length of 128 tokens.

## Results

The model was trained for six epochs.

| Metric | Result |
|---|---:|
| Validation loss | 4.9673 |
| MLM loss | 4.4275 |
| NSP loss | 0.5398 |
| NSP accuracy | 73.52% |

Training and validation losses decreased consistently during training. The model also demonstrated the ability to produce context-related masked token predictions, although its predictions were often concentrated on common tokens and subword units.

## Comparison with the Original Paper

Unlike the original BERT experiments, this project uses:

- A smaller Transformer architecture.
- A reduced vocabulary.
- A limited WikiText-103 dataset subset.
- A shorter training duration.
- No downstream task fine-tuning.

Consequently, the project should be viewed as an implementation and learning exercise rather than a direct reproduction of the original BERT results.

## Key Learnings

The project provided practical experience with:

- Implementing Transformer components.
- Understanding self-attention and contextual representations.
- Designing language model pretraining objectives.
- Developing tokenization and data preprocessing pipelines.
- Training and validating neural language models.
- Interpreting model performance and limitations.
- Distinguishing between a research paper's original experimental setup and a resource-constrained implementation.

## Future Work

Future improvements include increasing the model and dataset scale, improving the training schedule, conducting controlled evaluation of MLM predictions, ensuring stronger document-level dataset separation, and fine-tuning the model on downstream NLP tasks.

## Project Significance

This project demonstrates an effort to understand the underlying mechanisms of modern language models through implementation rather than solely relying on high-level libraries or pretrained checkpoints. It provides a foundation for further exploration of Transformer architectures, representation learning, and NLP research.
