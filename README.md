# Cardiac Function Prediction Pipeline (Baseline vs. Mamba)

Reproducible video regression pipeline predicting Left-Ventricular Ejection Fraction (EF) under missing frames. Implements a **3D CNN Baseline** (`baseline` / R3D-18) and a **Mamba-inspired Selective SSM** (`mamba` / GapSSM) with continuous-time gap handling, epoch checkpointing, and robustness evaluation.

---

## 1. Quickstart: Kaggle Notebook (1-Click Run)

The repository includes a ready-to-run notebook in [`notebooks/kaggle_cardiac_ai.ipynb`](notebooks/kaggle_cardiac_ai.ipynb).

### Steps to Run on Kaggle:
1. **Upload**: In Kaggle, create a new Notebook and upload this project folder into your session (or clone your repo into `/kaggle/working`).
2. **Accelerator**: In the right sidebar under **Session options**, set **Accelerator** to **GPU T4 x2** or **GPU P100**.
3. **Dataset** *(Optional)*:
   - If you have EchoNet-Dynamic, click **+ Add Input**, search or upload your private EchoNet-Dynamic dataset.
   - If no dataset is attached, the notebook **automatically detects this and generates a synthetic demo dataset** so you can test the full end-to-end pipeline with zero errors.
4. **Execute**: Open `notebooks/kaggle_cardiac_ai.ipynb` and click **Run All**.
5. **Output**: All results, plots, metrics tables, and epoch checkpoint files are saved in `/kaggle/working/cardiac_runs` and automatically zipped to `cardiac_experiment_results.zip` for instant download.

---

## 2. Local Setup & Installation

Requires Python 3.9+:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

To verify your local setup immediately:
```bash
python scripts/smoke.py --out runs/smoke_test
```

---

## 3. Dataset Structure

Place the official EchoNet-Dynamic dataset under `data/EchoNet-Dynamic/`:
```text
data/EchoNet-Dynamic/
    FileList.csv          # Columns: FileName, EF, Split
    VolumeTracings.csv    # Columns: FileName, Frame, X1, Y1, X2, Y2
    Videos/
        <FileName>.avi
```

---

## 4. Pipeline Execution Instructions

### Step 1: Generate Dataset Manifest
Audits video files, checks frames and dimensions, extracts End-Diastole (ED) and End-Systole (ES) tracing polygons:
```bash
python -m cardiac.cli manifest --data data/EchoNet-Dynamic --out runs/manifest.json --allow-subset
```

### Step 2: Freeze Test Mask Bank
Generates fixed, reproducible missing-frame masks across ratios (0%, 10%, 25%, 50%, 75%) and patterns (`random`, `block`, `ed`, `es`, `systole`, `diastole`):
```bash
python -m cardiac.cli bank --manifest runs/manifest.json --out runs/test_bank.json --split TEST
```

### Step 3: Train the Models
Train both models independently using their respective configuration files.

#### A. Train Baseline Model (R3D-18 3D CNN):
```bash
python -m cardiac.cli train \
  --data data/EchoNet-Dynamic \
  --manifest runs/manifest.json \
  --config configs/baseline.json \
  --out runs/baseline
```

#### B. Train Mamba Model (GapSSM Selective State-Space Model):
```bash
python -m cardiac.cli train \
  --data data/EchoNet-Dynamic \
  --manifest runs/manifest.json \
  --config configs/mamba.json \
  --out runs/mamba
```

#### Checkpoints & Epoch Files
During training, the engine automatically records and stores:
- `runs/<model>/best.pt`: Model weights with the lowest validation MAE.
- `runs/<model>/last.pt`: Latest model state for seamless resumption.
- `runs/<model>/epochs/checkpoint_epoch_<N>.pt`: Per-epoch weights, optimizer state, and loss metrics.
- `runs/<model>/training.csv`: Full epoch-by-epoch loss and validation history.

To resume training if interrupted:
```bash
python -m cardiac.cli train ... --resume runs/<model>/last.pt
```

---

## 5. Model Evaluation & Robustness Study

Evaluate both models across missing-frame ratios and imputation techniques (`zero`, `removal`, `repeat`, `interpolate`) with Monte Carlo dropout uncertainty:

```bash
# Evaluate Baseline
python -m cardiac.cli evaluate \
  --data data/EchoNet-Dynamic \
  --manifest runs/manifest.json \
  --bank runs/test_bank.json \
  --checkpoint runs/baseline/best.pt \
  --model-name baseline \
  --out runs/baseline/evaluation \
  --handlings zero removal repeat interpolate \
  --mc-passes 20

# Evaluate Mamba
python -m cardiac.cli evaluate \
  --data data/EchoNet-Dynamic \
  --manifest runs/manifest.json \
  --bank runs/test_bank.json \
  --checkpoint runs/mamba/best.pt \
  --model-name mamba \
  --out runs/mamba/evaluation \
  --handlings zero removal repeat interpolate \
  --mc-passes 20
```

---

## 6. Comparison Reporting & Visualizations

Generate comparative metrics (MAE, RMSE, $R^2$, Pearson $r$, prediction shift) and training curve plots:

```bash
# Generate comparative tables & summary
python -m cardiac.cli report \
  --predictions runs/baseline/evaluation/predictions.csv runs/mamba/evaluation/predictions.csv \
  --out runs/comparison

# Plot training and validation loss curves
python scripts/plot_training.py \
  --runs runs/baseline runs/mamba \
  --out runs/comparison/training_curves.png
```

---

## 7. Model Configurations Reference

- **Baseline (`configs/baseline.json`)**:
  - `model`: `"baseline"` (R3D-18 video backbone)
  - `corruption`: `false` (trained on intact video clips)
  - `epochs`: 45
  - `save_epochs`: `true`

- **Mamba (`configs/mamba.json`)**:
  - `model`: `"mamba"` (Gap-aware Selective SSM)
  - `width`: 128
  - `state_size`: 16
  - `layers`: 2
  - `encoder`: `"resnet18"`
  - `corruption`: `true` (trained with random and block frame dropout)
  - `epochs`: 45
  - `save_epochs`: `true`
