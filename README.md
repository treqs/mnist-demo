# mnist-demo

A minimal MNIST classification pipeline for demonstrating [roar](https://github.com/treqs/roar) provenance tracking.

## Pipeline

The pipeline has three steps:

1. **`extract.py`** — Reads an MNIST parquet file (from Hugging Face), extracts 28x28 pixel images as flattened 784-dimensional feature vectors, and saves them as a `.npz` file.
2. **`train.py`** — Trains a logistic regression classifier (scikit-learn, SAGA solver) on the extracted features. Logs training loss per epoch to Weights & Biases. Saves the model as a `.pkl` file.
3. **`evaluate.py`** — Loads a trained model and test features, computes accuracy/precision/recall, prints a classification report, and writes metrics to a `.json` file.

## Data

Training and test data are MNIST parquet files hosted on Hugging Face:

- [`train-00000-of-00001.parquet`](https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet)
- [`test-00000-of-00001.parquet`](https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet)

## Dependencies

- Python 3.10+
- numpy, scikit-learn, pyarrow, Pillow, wandb

```bash
pip install numpy scikit-learn pyarrow Pillow wandb
```

## Usage

```bash
# Download data
wget "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet?download=true" -O train-00000-of-00001.parquet
wget "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet?download=true" -O test-00000-of-00001.parquet

# Extract features
python extract.py --input train-00000-of-00001.parquet --output train_feats.npz
python extract.py --input test-00000-of-00001.parquet --output test_feats.npz

# Train
python train.py --input train_feats.npz --output model.pkl

# Evaluate
python evaluate.py --model model.pkl --input test_feats.npz --output metrics.json
```
