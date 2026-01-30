# Continual Learning for Motor Imagery EEG
## Online and Offline Domain-Incremental Learning with EEGNet

## Abstract
This repository provides a modular experimental framework for studying continual learning in Motor Imagery EEG (MI‑EEG) using the PhysioNet EEG Motor Movement/Imagery dataset (EEGBCI), accessed via Braindecode/MNE. The framework supports (i) Leave‑One‑Subject‑Out (LOSO), (ii) Offline Domain‑Incremental Learning (Offline DIL), and (iii) Online Domain‑Incremental Learning (Online DIL). Performance is reported using final average accuracy (ACC) and continual‑learning transfer metrics (BWT, FWT).

## Dataset
The EEGBCI dataset is downloaded and cached automatically through Braindecode/MNE on first run (no manual download required). Default subject range and run selection are defined in `config.py`.

## Installation
Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Reproducibility
All reported experiments use the default random seed:

- `seed = 42`

(Seed is configurable via `--set seed=<int>`.)

## Baseline Configuration (Online DIL)
The baseline used for the Online DIL ablation study is:

| Hyperparameter | Value |
|---|---:|
| learning rate | 1e-2 |
| weight decay | 0 |
| batch size | 4 |
| dropout | 0.0 |
| temporal filters (F1) | 8 |
| spatial filters (D) | 1 |
| temporal kernel length | 16 |
| depthwise kernel length | 8 |
| pooling mode | mean |

## Running Experiments
All experiments are executed via `run.py`.

### 1) Online Domain‑Incremental Learning (Online DIL)
Baseline:
```bash
python run.py --exp online --ablation baseline
```

### 2) Offline Domain‑Incremental Learning (Offline DIL)
```bash
python run.py --exp offline
```

### 3) Leave‑One‑Subject‑Out (LOSO)
```bash
python run.py --exp loso
```

## Online DIL Ablations
Ablations are defined as named presets in `run.py` and each changes **one** hyperparameter relative to the baseline.

Example:
```bash
python run.py --exp online --ablation lr_1e-3
```

### Ablation presets (Online DIL)
- **Learning rate**: `lr_5e-4`, `lr_1e-3`, `lr_5e-3`, `lr_3e-2`
- **Weight decay**: `wd_1e-4`, `wd_1e-3`, `wd_1e-2`
- **Batch size**: `bs_8`, `bs_16`, `bs_32`
- **Dropout**: `drop_0.10`, `drop_0.25`, `drop_0.50`
- **Temporal filters (F1)**: `F1_16`, `F1_32`
- **Spatial filters (D)**: `D_2`, `D_4`, `D_8`
- **Temporal kernel length**: `klen_32`, `klen_64`, `klen_128`
- **Depthwise kernel length**: `dwklen_16`, `dwklen_32`, `dwklen_64`
- **Pooling mode**: `pool_max`

## Outputs
Each run writes an output folder under `outputs/`. For Online DIL, results are stored in:
- `outputs/<tag>/online_domain_incremental/results.json`

Key fields used to fill the ablation table:
- `ACC_final`, `FWT_final`, `BWT_final`

## Author
Narges Najiantabriz
