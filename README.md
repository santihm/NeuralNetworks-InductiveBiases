# Inductive Biases: When Does Architecture Matter?

Group project for **Neural Networks** (Master's in Applied Artificial Intelligence, UC3M), by Alejandro Riego Velasco, Javier Aceituno Monja, Santiago Herron Mulet and Rodrigo Toldos Frutos.

> The notebook and the poster are written in Spanish. This README summarizes them in English.

## The question

An MLP, a CNN, an LSTM and a Transformer assume different things about the structure of the data: none, locality and shared weights, sequential order, and relations between all elements. Using Fashion-MNIST, we test with experiments **when those assumptions help, when they hurt, and why**.

## Setup

Four models of about 100,000 parameters each, trained with the same protocol and five seeds. Each model sees the same image in its own way: the MLP a vector of 784 values, the CNN the 28×28 image, the LSTM a sequence of 28 rows, and the Transformer a sequence of 49 patches of 4×4.

| Model | Parameters | Test accuracy (mean ± std, 5 seeds) |
|---|---|---|
| MLP | 101,706 | 89.27 ± 0.14 |
| CNN | 99,122 | 91.98 ± 0.16 |
| LSTM | 101,674 | 90.46 ± 0.12 |
| Transformer | 105,546 | 88.38 ± 0.37 |

**Common protocol.** AdamW, batch size 128, cosine decay with warmup, 8,000 steps, best checkpoint on validation, no early stopping and no data augmentation (it would add invariances that contaminate the experiments). The learning rate of each model is chosen with the same sweep on validation. Split: 55,000 train, 5,000 validation and 10,000 test images. Fixed seeds everywhere.

**Predictions written before running anything.** For each of the four hypotheses we wrote down what we expected, and the verdicts use the criteria written in `sesgos/verdicts.py`. Differences are reported with Welch 95% confidence intervals and Holm correction for multiple comparisons.

## Results

| Hypothesis | Experiment | Verdict | Key result |
|---|---|---|---|
| **H1** Input structure | Fixed permutation of pixels or of 4×4 blocks, models retrained | Confirmed | With permuted pixels the CNN loses 3.9 points and the LSTM 2.8, while the MLP does not change. The Transformer only drops with pixels: permuting blocks is equivalent to permuting its learned positions, which it relearns. |
| **H2** Amount of data | Train with 100 to 55,000 images | Confirmed, with a nuance | With 100 images the best and worst models are 18.6 points apart (3.6 with 55,000), but only because the LSTM and the Transformer collapse. The CNN does not beat the MLP with 100 images, starts to beat it between 300 and 1,000, and its advantage grows with data. |
| **H3** Transformations at test time | Shifts and rotations, no retraining | Confirmed | Shifted by 4 pixels, the CNN keeps 95% and 92% of its accuracy (horizontal, vertical) against 30% and 60% for the MLP. The LSTM only tolerates vertical shifts. No model tolerates rotation. |
| **H4** Memorization | Train on 0 to 100% random labels | Confirmed | All four memorize fully random labels, with test accuracy at chance, but at very different speeds: 59 epochs for the Transformer, 73 for the LSTM, 117 for the CNN and 110 for the MLP (with learning rate 1e-3; with its protocol learning rate the MLP stalls). Weight decay delays memorization without improving test accuracy. |

**Main conclusion.** An inductive bias helps when the data has the structure the bias assumes, and it can hurt when it does not. The CNN is the best model on real images and falls below the MLP when the pixels are shuffled. It tolerates shifts because convolution respects them, and it does not tolerate rotations because nothing in it does.

**Explainability.** First-layer weights (the MLP learns the same templates in a different order after pixel permutation), Integrated Gradients (the CNN's explanation moves with the garment when it is shifted, the others do not) and attention maps of the Transformer. The notebook also includes exact controls, for example showing that a permuted MLP follows the same training trajectory as the original.

**Limitations.** One dataset with centred garments, models of about 100,000 parameters, one learning rate chosen per model, one regularization technique, and five seeds per condition, so small differences with p close to 0.05 should not be read as effects.

## Repository

| Path | Content |
|---|---|
| `sesgos_inductivos.ipynb` | Main notebook, self-contained, with the full code and all the analysis |
| `sesgos/` | Common code: data, models, training, experiments H1 to H4, explainability, statistics and figures |
| `sesgos/config.py` | Frozen experimental protocol (seeds, budget, sizes, transformations) |
| `results/raw/` | One CSV per experiment and model |
| `results/checkpoints/` | Weights of every trained model |
| `results/logs/` | Log of the original run |
| `figures/` | Figures of the notebook and of the poster |
| `poster/` | A0 poster: `poster.pdf` and its HTML/CSS source |
| `tools/` | Scripts to rebuild the notebook, the poster and the figures |
| `tests/` | Quick checks of the code |
| `assets/fonts/` | Caladea and Carlito fonts (OFL license) used in figures and poster |

## How to run

```bash
git clone https://github.com/santihm/NeuralNetworks-InductiveBiases
cd NeuralNetworks-InductiveBiases
pip install -r requirements.txt
```

Then open `sesgos_inductivos.ipynb` and choose the mode in the configuration cell:

- `RUN_MODE = "cargar"` (default): does not train. It reads `results/` and reproduces all tables, figures and explanations in a few minutes.
- `RUN_MODE = "completo"`: trains everything that is missing (about 360 trainings, roughly one day on an Apple Silicon laptop or a Colab T4 GPU).
- `RUN_MODE = "prueba"`: same code with a minimal budget, to check in minutes that everything works. Its numbers are not valid results.

The results were generated on a Mac with Apple Silicon (PyTorch with MPS). On other devices or PyTorch versions, retraining gives results that are equivalent in distribution, not identical bit by bit. Fashion-MNIST is downloaded automatically with torchvision. The versions used are in `requirements.txt`.

Command-line examples:

```bash
python -m pytest -q                      # quick checks of the code
python -m sesgos.run all --smoke         # whole pipeline in minutes, numbers not valid
python -m sesgos.run h1 --models cnn     # baseline and H1 for one model, 5 seeds
```

## AI use

Claude (Anthropic) was used as support for programming, analysis and writing, as declared in the notebook.
