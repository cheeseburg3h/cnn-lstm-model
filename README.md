# CNN-LSTM-Attention Malware Classifier

This repository provides a modular TensorFlow/Keras implementation of a malware classifier for the EMBER dataset that combines convolutional layers, bidirectional LSTMs, and an additive attention mechanism. The project focuses on reproducible experiments for 4-fold cross-validation, internal holdout evaluation, and external generalization testing against newer EMBER releases.

## Project Structure

- `model_def.py` – neural network definition and custom attention layer.
- `train_utils.py` – reusable utilities for data scaling, training loops, callbacks, and evaluation.
- `experiments.py` – experiment orchestration, dataset loading, and CLI entry point.
- `requirements.txt` – pinned dependencies.

Artifacts from experiments (metrics, checkpoints, scalers) are saved under `results/`.

## Setup

1. **Python environment**
   - Use Python 3.9 or later.
   - Create a virtual environment and install dependencies:
     ```bash
     python -m venv .venv
     source .venv/bin/activate  # On Windows use .venv\Scripts\activate
     pip install --upgrade pip
     pip install -r requirements.txt
     ```
2. **EMBER dataset**
   - Clone the [EMBER repository](https://github.com/elastic/ember) and follow its instructions to obtain the raw dataset files (2018 and optionally 2024 updates). These archives contain feature CSVs and metadata, not raw PE binaries.
   - **Important:** EMBER relies on [LIEF](https://lief.quarkslab.com/). Pin LIEF to version `0.9.0` for compatibility:
     ```bash
     pip install lief==0.9.0
     ```
   - Install the EMBER helpers:
     ```bash
     pip install git+https://github.com/elastic/ember.git
     # or
     python setup.py install  # from within the cloned ember repository
     ```
   - Create vectorized features (only needs to be done once per dataset):
     ```bash
     python -m ember create-vectorized-features /path/to/ember2018
     python -m ember create-vectorized-features /path/to/ember2024  # optional
     ```
     The scripts in this repository will automatically call `ember.create_vectorized_features` if needed, but running it manually allows you to monitor progress. This process does **not** download malware binaries.
3. **Apple Silicon / Docker note**
   - On Apple M1/M2 hardware, prefer running the project inside a Docker container with an x86_64 base image to ensure TensorFlow compatibility. Example base image: `tensorflow/tensorflow:2.12.0`. Mount your data directories and run the experiments inside the container.

## Running Experiments

All experiments share the same CLI defined in `experiments.py`. Enable logging by setting `TF_CPP_MIN_LOG_LEVEL=1` or adjusting the logging configuration within the script.

### 1. 4-fold Cross-Validation (EMBER 2018)
```bash
python experiments.py --run cv --ember_root /data/ember2018 --epochs 20 --batch_size 256 --n_splits 4
```
This command trains four models with stratified folds and averages their metrics. Fold-specific checkpoints are saved to `results/checkpoints/cv_<fold>.h5`, and aggregated metrics are written to `results/cv_metrics.json`.
Each fold now also saves its fitted scaler alongside the weights (`results/checkpoints/cv_<fold>.joblib`), and a summary file `results/cv_metadata.json` captures the metrics plus the best fold by F1 for easy reuse. The `--ember_root` path is never fetched from the internet—it is only read locally from your machine (e.g., `--ember_root "C:\\Users\\Justin\\Desktop\\dataset\\ember"` on Windows), so no data is uploaded or downloaded during training.

### 2. Internal Holdout Evaluation (90/10)
```bash
python experiments.py --run internal --ember_root /data/ember2018 --epochs 30 --test_size 0.1
```
The script splits the vectorized features into 90% training and 10% validation data, trains the final model with early stopping, and stores metrics and artifacts under `results/`.

### 3. External Generalization Test (EMBER 2024)
```bash
python experiments.py --run external \
  --ember_root /data/ember2018 \
  --ember_root_2024 /data/ember2024 \
  --epochs 30
```
This experiment fits the scaler on EMBER 2018 only, trains a model, and evaluates on the combined EMBER 2024 vectorized dataset without refitting normalization parameters.

## Reproducibility Tips

- Random seeds for NumPy and TensorFlow are set to `42` inside `experiments.py`. Adjust as needed for different runs.
- For GPU training, ensure deterministic settings if required (TensorFlow `TF_DETERMINISTIC_OPS=1`) and consider enabling mixed precision (`tf.keras.mixed_precision.set_global_policy('mixed_float16')`).
- Keep track of environment details (CUDA/cuDNN versions, TensorFlow build) for full reproducibility.

## Safety Notice

The EMBER dataset contains only feature representations derived from PE files and does **not** provide executable malware samples. Do not attempt to download raw binaries from untrusted sources. Rely exclusively on EMBER’s feature generation tools (`ember.create_vectorized_features`) as demonstrated in this project.

