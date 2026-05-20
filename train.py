import argparse
import pickle

import numpy as np
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def main():
    parser = argparse.ArgumentParser(description="Train MNIST classifier")
    parser.add_argument("--input", required=True, help="Input .npz features file")
    parser.add_argument("--output", required=True, help="Output model .pkl file")
    args = parser.parse_args()

    data = np.load(args.input)
    X, y = data["X"], data["y"]
    print(f"Training on {X.shape[0]} samples, {X.shape[1]} features")

    config = {"solver": "saga", "max_iter": 20, "tol": 0.1, "C": 1.0}
    wandb.init(project="mnist", config=config)

    model = LogisticRegression(
        solver="saga",
        max_iter=1,
        warm_start=True,
        tol=0,
        C=config["C"],
        verbose=1,
    )

    for epoch in range(config["max_iter"]):
        model.fit(X, y)
        loss = log_loss(y, model.predict_proba(X))
        wandb.log({"train_log_loss": loss, "epoch": epoch})
        print(f"Epoch {epoch}: log_loss={loss:.4f}")

    wandb.finish()

    with open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
