import argparse
import io

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Extract features from MNIST parquet")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output .npz file")
    args = parser.parse_args()

    table = pq.read_table(args.input)
    n = table.num_rows
    print(f"Reading {n} images from {args.input}")

    images = table.column("image")
    labels = table.column("label").to_pylist()

    X = np.empty((n, 784), dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    for i in range(n):
        img_bytes = images[i]["bytes"].as_py()
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        X[i] = np.array(img, dtype=np.float32).flatten() / 255.0

    print(f"Extracted features: X={X.shape}, y={y.shape}")
    np.savez(args.output, X=X, y=y)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
