import argparse
import json
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate MNIST classifier")
    parser.add_argument("--model", required=True, help="Trained model .pkl file")
    parser.add_argument("--input", required=True, help="Test features .npz file")
    parser.add_argument("--output", required=True, help="Output metrics .json file")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    data = np.load(args.input)
    X, y = data["X"], data["y"]
    print(f"Evaluating on {X.shape[0]} samples")

    preds = model.predict(X)
    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds, average="macro", zero_division=0)
    recall = recall_score(y, preds, average="macro", zero_division=0)

    print(f"\nAccuracy: {accuracy:.4f}  Precision(macro): {precision:.4f}  Recall(macro): {recall:.4f}\n")
    print(classification_report(y, preds))

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "num_samples": int(X.shape[0]),
    }
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
