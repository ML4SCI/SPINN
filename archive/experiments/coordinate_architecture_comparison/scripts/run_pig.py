"""Train the PIG coordinate-projection model.

    python -m experiments.coordinate_architecture_comparison.scripts.run_pig
"""

import torch

from experiments.coordinate_architecture_comparison.training.train_one_model import run_model

if __name__ == "__main__":
    run_model("pig", device="cuda" if torch.cuda.is_available() else "cpu")
