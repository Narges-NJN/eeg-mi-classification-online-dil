# Continual Learning for Motor Imagery EEG

## Online and Offline Domain-Incremental Learning with EEGNet

------------------------------------------------------------------------

## Abstract

This repository implements a modular framework for evaluating
**continual learning strategies in Motor Imagery EEG (MI-EEG)** using
the PhysioNet EEG Motor Movement/Imagery dataset (EEGBCI).

The framework supports:

-   **Leave-One-Subject-Out (LOSO)**
-   **Offline Domain-Incremental Learning (Offline DIL)**
-   **Online Domain-Incremental Learning (Online DIL)**

The backbone architecture is **EEGNet**, and experiments are evaluated
using standard continual learning metrics:

-   Final Average Accuracy (**ACC**)
-   Backward Transfer (**BWT**)
-   Forward Transfer (**FWT**)

A structured **ablation study is conducted on the Online
Domain-Incremental setting**, systematically varying training and
architectural hyperparameters.

The codebase is fully reproducible and can run on **CPU or GPU**.

------------------------------------------------------------------------

## Dataset

Experiments use the:

**EEG Motor Movement/Imagery Dataset (EEGBCI)**\
Source: PhysioNet

The dataset is automatically downloaded using `braindecode` and `mne` on
first execution. No manual download is required.

### Default Configuration

-   Subjects: `1–80`
-   Domain-Incremental protocol:
    -   Training runs: `4, 8`
    -   Testing run: `12`
-   LOSO protocol:
    -   Runs: `4, 8, 12`

------------------------------------------------------------------------

## Environment Setup

### Option 1 --- Recreate from Environment File (Recommended)

``` bash
conda env create -f environment.yml
conda activate eegmi
```

### Option 2 --- Manual Setup

``` bash
conda create -n eegmi python=3.10
conda activate eegmi
conda install pytorch -c pytorch
pip install -r requirements.txt
```

If a GPU is available, install CUDA-enabled PyTorch following the
official PyTorch instructions.

The framework runs correctly on CPU; GPU acceleration is optional.

------------------------------------------------------------------------

## Reproducibility

All reported experiments use:

    seed = 42

The seed is set globally to ensure deterministic behavior.

------------------------------------------------------------------------

## Baseline Configuration (Online Domain-Incremental Learning)

The official baseline used in the ablation study is:

  Hyperparameter            Value
  ------------------------- -------
  Learning rate             1e-2
  Weight decay              0
  Batch size                4
  Dropout                   0.0
  Temporal filters (F1)     8
  Spatial filters (D)       1
  Temporal kernel length    16
  Depthwise kernel length   8
  Pooling mode              mean

Each ablation modifies **exactly one parameter relative to this
baseline**.

------------------------------------------------------------------------

## Running Experiments

All experiments are controlled via:

``` bash
python run.py
```

------------------------------------------------------------------------

### Online Domain-Incremental Learning (Baseline)

``` bash
python run.py --exp online --ablation baseline
```

------------------------------------------------------------------------

### Offline Domain-Incremental Learning

``` bash
python run.py --exp offline
```

------------------------------------------------------------------------

### Leave-One-Subject-Out (LOSO)

``` bash
python run.py --exp loso
```

------------------------------------------------------------------------

## Ablation Study (Online DIL Only)

Each ablation corresponds to a predefined configuration in `run.py`.

Example:

``` bash
python run.py --exp online --ablation lr_1e-3
```

### Available Ablations

-   Learning rate variations
-   Weight decay variations
-   Batch size variations
-   Dropout variations
-   Temporal filter size (F1)
-   Spatial filter size (D)
-   Kernel length variations
-   Depthwise kernel variations
-   Pooling mode change (mean → max)

------------------------------------------------------------------------

## Output Structure

Each experiment produces:

    outputs/
     └── online_<ABL>_seed42_<timestamp>/
         └── online_domain_incremental/
             └── results.json

The file `results.json` contains:

-   `ACC_final`
-   `BWT_final`
-   `FWT_final`

These values populate the ablation table.

------------------------------------------------------------------------

## Evaluation Metrics

-   **ACC**: Final average accuracy across domains\
-   **BWT (Backward Transfer)**: Measures forgetting across domains\
-   **FWT (Forward Transfer)**: Measures positive transfer to future
    domains

------------------------------------------------------------------------

## Project Structure

    config.py
    run.py
    experiments.py
    models.py
    train_utils.py
    data.py
    plots.py
    utils.py
    environment.yml

------------------------------------------------------------------------

## Reproducing Reported Results

To reproduce the baseline result:

``` bash
conda activate eegmi
python run.py --exp online --ablation baseline
```

To reproduce the full ablation table, execute each ablation sequentially
using the same seed.

All results are deterministic under the default configuration.

------------------------------------------------------------------------

## Author

Narges\
BSc in Applied Computer Science and Artificial Intelligence\
Sapienza University of Rome
