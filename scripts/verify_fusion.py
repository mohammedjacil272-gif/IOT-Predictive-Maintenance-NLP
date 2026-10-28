"""Smoke-test the fusion pipeline using synthetic data only."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.multimodal_fusion import MultimodalFusionModel
from training.fusion_metrics import regression_metrics


def main():
    torch.manual_seed(42)
    count = 32

    sensor = torch.randn(count, 64)
    text = torch.randn(count, 768)
    targets = torch.randn(count) * 10 + 100

    dataset = TensorDataset(sensor, text, targets)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = MultimodalFusionModel(use_has_log=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for sensor_batch, text_batch, target_batch in loader:
        optimizer.zero_grad(set_to_none=True)
        predictions = model(sensor_batch, text_batch)
        loss = loss_fn(predictions, target_batch)
        assert torch.isfinite(loss), "Non-finite loss"
        loss.backward()
        optimizer.step()

    model.eval()
    predictions, actual = [], []
    with torch.no_grad():
        for sensor_batch, text_batch, target_batch in loader:
            output = model(sensor_batch, text_batch)
            assert output.shape == target_batch.shape
            assert torch.isfinite(output).all()
            predictions.extend(output.tolist())
            actual.extend(target_batch.tolist())

    print("Synthetic smoke test: PASS")
    print("Sensor shape:", tuple(sensor.shape))
    print("Text shape:", tuple(text.shape))
    print("Prediction count:", len(predictions))
    print("Synthetic metrics:", regression_metrics(actual, predictions))
    print("NOTE: These are mock-data metrics, not real model performance.")


if __name__ == "__main__":
    main()
