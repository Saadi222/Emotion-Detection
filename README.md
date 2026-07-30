# Emotion Classification with Fine-Tuned BERT

Fine-tuning `bert-base-uncased` to classify short text into six emotions — sadness, joy, love, anger, fear, and surprise — benchmarked against a classical TF-IDF baseline and a frozen-BERT baseline.

## Results

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 86.90% | 83.45% | 87.13% |
| Frozen BERT + Logistic Regression | 50.90% | 44.98% | 53.16% |
| **Fine-tuned BERT** | **93.70%** | **89.19%** | **93.53%** |

Fine-tuning is what makes BERT competitive here, not simply having access to a pretrained model — frozen BERT embeddings alone perform *worse* than the TF-IDF baseline, since BERT's `[CLS]` token isn't pretrained to be a general-purpose sentence representation on its own.

## Dataset

[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (Saravia et al., 2018) — English-language tweets, labelled via distant supervision from the hashtags authors attached to their own posts.

| Split | Examples |
|---|---|
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

Classes are imbalanced (joy ≈ 34%, surprise ≈ 3.6%, a 9.4× gap), which is why **macro-F1** is used as the primary metric throughout rather than accuracy alone.

## Method

- **Tokenizer:** `bert-base-uncased` WordPiece, no manual text cleaning
- **Max sequence length:** 64 tokens, chosen empirically from the training-set token-length distribution (truncates 0.16% of examples vs. 17.79% at length 32)
- **Model:** `bert-base-uncased` (109.5M parameters) + a linear classification head over pooled `[CLS]`
- **Training:** AdamW, weight decay excluded from bias/LayerNorm parameters, linear warmup + decay schedule, gradient clipping, early stopping on validation macro-F1 (patience 2, up to 5 epochs)
- **Hyperparameter search:** grid over learning rate `{2e-5, 3e-5, 5e-5}` × batch size `{16, 32}` — best: **lr = 5e-5, batch size = 32** (90.67% validation macro-F1)
- **Class imbalance:** tested directly as an ablation (standard vs. class-weighted cross-entropy). Weighting improved the rarest class (surprise, +1.84 F1) but reduced love (−2.42) and joy (−1.11), for a very slightly *lower* overall macro-F1 (91.26% vs. 91.59%) — so standard cross-entropy was used in the final model
- **Baselines:** TF-IDF + Logistic Regression, and Logistic Regression on frozen BERT `[CLS]` embeddings

## Key findings

- The two largest sources of confusion are **surprise → fear** (28.8% of surprise examples) and **love → joy** (24.5% of love examples) — both semantically adjacent emotion pairs
- Performance tracks training-class size closely: surprise (3.6% of training data) has the lowest test F1 at 75.21% (recall 66.67%), while joy and sadness — the two largest classes — both exceed 95% F1
- A data-hygiene check found minor train/test text overlap (11 of 2,000 test examples) — small enough to bound any resulting optimism in the test score to a fraction of a percentage point, but disclosed rather than ignored

## Repository structure

```
├── emotion_classification_bert.ipynb   # Main notebook: EDA → preprocessing → baselines →
│                                        # fine-tuning → hyperparameter tuning → evaluation
├── emotion_bert_report.docx            # 3-page written report
└── README.md
```

## Getting started

```bash
pip install transformers datasets torch scikit-learn matplotlib seaborn pandas
```

Open `emotion_classification_bert.ipynb` and run all cells top to bottom. A GPU is strongly recommended. Approximate runtimes on a free Google Colab T4 (set **Runtime → Change runtime type → T4 GPU** first):

| Stage | Approx. time |
|---|---|
| EDA + baselines | ~4 min |
| Hyperparameter search (6 configs) | ~17 min |
| Final training (2 loss variants, up to 5 epochs each) | ~25 min |
| **Total** | **~45 min** |

## Limitations

- Labels come from hashtag-based distant supervision, not manual annotation, so some label noise is inherent to the data
- The rarest class (surprise) has only 66 test examples, making its score comparatively noisy
- Results reflect a single training run on English-language tweets only

## References

- Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019) 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', *Proceedings of NAACL-HLT 2019*, pp. 4171–4186.
- Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J. and Chen, Y.-S. (2018) 'CARER: Contextualized Affect Representations for Emotion Recognition', *Proceedings of EMNLP 2018*, pp. 3687–3697.
