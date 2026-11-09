
"""Forward-only smoke test of the fusion model using synthetic data.

The script is intentionally runnable both as ``python scripts/verify_fusion.py``
and as ``python -m scripts.verify_fusion`` from the repository root.
"""

from pathlib import Path
import sys

# A direct execution puts ``scripts/`` on sys.path, not the repository root.
# Keep the verifier usable as documented without requiring PYTHONPATH changes.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.multimodal_fusion import MultimodalFusionModel


def main() -> None:
    torch.manual_seed(42)
    count = 32
    sensor = torch.randn(count, 64)
    text = torch.randn(count, 768)

    model = MultimodalFusionModel(use_has_log=False)
    model.eval()

    with torch.inference_mode():
        predictions = model(sensor, text)

    assert predictions.shape == (count,)
    assert torch.isfinite(predictions).all().item()

    print("Synthetic forward-only smoke test: PASS")
    print("Sensor shape:", tuple(sensor.shape))
    print("Text shape:", tuple(text.shape))
    print("Prediction count:", predictions.numel())
    print("NOTE: No optimizer, backward pass, training loop, or artifact write is performed.")


if __name__ == "__main__":
    main()
