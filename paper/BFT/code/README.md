# BFT: full paper implementation

Implementation of the experiments in **"Backpropagation-Free Test-Time
Adaptation for Lightweight EEG-Based BCIs"**, IEEE Journal of Biomedical and
Health Informatics, 2026.
[arXiv](https://arxiv.org/abs/2601.07556) ·
[supplement](../supp/2026_BFT_JBHI_Li_supp.pdf)

This directory holds the code that *runs* the experiments. It does not hold the
scripts that turn their outputs into the manuscript's figures and tables, nor the
result files those scripts consumed; producing the paper is not what a code
release is for.

## How this relates to `tl/bft.py`

The repository ships BFT twice, on purpose.

- **`tl/bft.py`** is the benchmark entry point. It is one self-contained file
  that follows the same conventions as every other method in `tl/`, so BFT can
  be run head to head against Tent, T3A, SAR, DELTA, CoTTA and T-TIME on the
  same MOABB loaders, the same source checkpoints and the same eleven seeds. It
  covers the motor-imagery classification setting only, and it substitutes a
  closed-form soft-rank surrogate for the SoDeep mapping module so that it needs
  no extra checkpoint.
- **This directory** is the paper implementation. It covers all five datasets,
  both tasks, the real SoDeep mapping module, the corruption robustness study
  and the quantization and latency study.

Both implement the same method and the same equations. Where either departs from
the manuscript, the departure is flagged in a comment at the point where it
happens.

## Layout

```
BFT-classify/          Classification: source model, ranking module, inference
  train_pre_model.py     source model g and task head h (EEGNet)
  train_loss_model.py    ranking module r, Eq. (6), supervised through sodeep
  test.py                BFT-A and BFT-D inference, Eq. (5) and Eq. (7)
  quantization.py        post-training INT8 accuracy
  augment.py             knowledge-guided augmentations, the T_k of Eq. (1)
  dropout.py             deterministic mask bank, Eq. (2) and Eq. (3)
  models/                EEGNet, ranking module, augmented-trainset builder
  utils/                 Euclidean Alignment, data loading
BFT-regression/        Regression counterpart (Driving, SEED-VIG)
  train_regression_model.py, train_loss_model.py, test.py
  augment_utils.py       Eq. (1) and Eq. (8) for regression
  dropout_utils.py       Eq. (2), Eq. (3) and Eq. (8) for regression
  models/                EEGNet, EEG Conformer, Deformer, ranking module
  utils/                 alignment, PSD features, data loading
sodeep/                The mapping module m, Eq. (4): a Bi-LSTM trained to
                       imitate a sort, used to put Eq. (6) in a rank space
corruptions.py         The seven test-time corruptions of Section IV-F, shared
                       by the classification and the regression test scripts
latency/               Per-sample latency measurement, Section IV-H
  latency_decomposition.py   the per-stage breakdown of Table VIII
  latency_sweep.py           the same breakdown as a function of K
requirements.txt
```

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The five datasets are not bundled, because of their size. Zhou2016,
BNCI2014001 and HighGamma (Schirrmeister2017) are the motor-imagery
classification datasets; Driving and SEED-VIG are the driver-drowsiness
regression datasets. Obtain them from their original sources and place the
preprocessed arrays where the loaders in `BFT-classify/utils/` and
`BFT-regression/utils/` expect them. Evaluation is leave-one-subject-out on the
first session only, matching Section IV-B.

Before the first run, set the output and data roots. They appear as
`/PATH/TO/SAVE/MODEL/` and `/PATH/TO/AUGED/DATA/` in `BFT-classify/test.py`,
`BFT-classify/quantization.py`, `BFT-classify/models/losspredictor.py` and
`BFT-regression/train_loss_model.py`.

## Order of operations

The three stages must run in order, because each consumes the previous one's
checkpoint.

1. **Mapping module m**, Eq. (4). One sorter per bank size:
   `cd sodeep && python train.py -slen 12 --name 12th_100epochs` and again with
   `-slen 10 --name 10th_100epochs`. These write
   `sodeep/weights/<name>_best_model.pth.tar`, which is what the ranking-module
   trainers load.
2. **Source model** g and head h: `train_pre_model.py` for classification,
   `train_regression_model.py` for regression.
3. **Ranking module r**, Eq. (6): `train_loss_model.py`, which needs both of
   the above. Then `test.py`.

## Running each experiment

**Classification (Section IV-D).** Run the three stages above under
`BFT-classify/`, then `python test.py`. The dataset list is at the top of
`test.py`. Each of the six methods reported in the paper prints its own line:
the source model, augmentation or masking with a uniform mean, BN-adapt, and the
two BFT variants.

**Regression (Section IV-E).** The same three stages under `BFT-regression/`,
then `python test.py`. The regression head replaces the classifier, and
aggregation uses the top-`ceil(K/2)` average of Eq. (8) rather than the
temperature-weighted softmax of Eq. (7).

**Test-time robustness (Section IV-F).** Set `args.corruption` and
`args.severity` near the top of either `test.py` and rerun. `corruptions.py`
defines the seven corruptions; `'temporal_segment_noise'` and `'channel_noise'`
are the temporal and spatial Gaussian noise the paper reports, and
`'mixed_artifact'` is the worst case. The corruption is injected before
Euclidean Alignment, so the whole test-time pipeline sees the degraded signal.
`args.corruption = 'clean'` is the uncorrupted condition and leaves the
classification and regression results unchanged.

**Ranking and mapping modules (Section IV-G).** The mapping-module ablation
compares three ways of supervising the ranking module: train
`BFT-classify/train_loss_model.py` with the sorter, without it, and with a
plain regression target, then evaluate each with `test.py`. `sodeep/train.py`
trains the sorter itself, and its loss is Eq. (4).

**Quantization and latency (Section IV-H).** Accuracy under post-training INT8
comes from `BFT-classify/quantization.py`. The per-sample latency breakdown
comes from `latency/latency_decomposition.py`, which needs no checkpoints and no
data because latency depends on architecture and shape rather than on weights:

```sh
python latency/latency_decomposition.py --devices cpu,cuda --csv latency.csv
python latency/latency_sweep.py --out latency_vs_K.csv --devices cpu,cuda
```

## Equation map

| Equation | What it is | Where |
|---|---|---|
| (1) | `z^(k) = g(T_k(x))`, knowledge-guided augmentations | `BFT-classify/augment.py`, `BFT-regression/augment_utils.py` |
| (2) | the deterministic mask `I^(k)` | `BFT-classify/dropout.py`, `BFT-regression/dropout_utils.py` |
| (3) | `z = 1/(1-p) I^(k) . g(x)` | same files as (2) |
| (4) | `L_mapping`, pretraining the sorter m | `sodeep/train.py`, `sodeep/dataset.py` |
| (5) | `w_{i,k} = softmax(r(z_i^(k)))` | `models/losspredictor.py`, `models/lossPredictor.py` |
| (6) | `L_ranking` | `train_loss_model.py` in both task directories |
| (7) | classification aggregation with temperature | `BFT-classify/test.py` |
| (8) | regression top-`ceil(K/2)` average | `BFT-regression/augment_utils.py`, `dropout_utils.py` |

Comments in the source name the equation each block implements, and flag the
places where the code departs from the manuscript.

## Repairs applied to this release

Eight defects prevented the code from starting or from importing. None of them
touched the algorithm, and all are fixed here.

- `models/losspredictor.py` imported its siblings assuming `models/` was the
  working directory. It now extends `sys.path` from its own location, and
  resolves `sodeep/` the same way.
- `models/augment_trainset.py` imported `pyhht` and `PyEMD`, neither used nor a
  dependency, so the module could not be imported at all. Both are removed.
  `pywt` is used and stays; `PyWavelets` and `scikit-learn` are now declared.
- `models/Conformer.py` imported `matplotlib`, `PIL` and
  `torchvision.transforms`, none of them used and none of them declared, so
  importing the Conformer backbone required two packages the method never
  touches. All are removed.
- The sorter path was built as `PATH_TO_SODDEP + 'weights/...'` without a
  separator, giving `../sodeepweights/...`.
- `sodeep`'s `save_checkpoint` wrote `./weights/best_<name>.pth.tar` while the
  loaders opened `weights/<name>_best_model.pth.tar`. It now writes the name the
  loaders expect, in the package directory rather than the working directory,
  and creates it if absent.
- The trainers named checkpoints by iteration, producing `EEGNet_epoch_7200.pth`
  while `test.py` loaded `EEGNet_epoch_200.pth`, a file nothing ever wrote. They
  now name by epoch.
- `interval_iter` evaluated to zero for any `max_epoch` under 10, so
  `iter_num % interval_iter` raised `ZeroDivisionError`. It is now `max(1, ...)`.
- `sodeep.load_sorter` calls `torch.load(..., weights_only=...)`, which needs
  torch 1.13, while `requirements.txt` allowed 1.12.
