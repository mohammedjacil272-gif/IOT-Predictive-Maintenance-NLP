from pathlib import Path

from scripts.validate_repository_datasets import build_inventory


def test_inventory_keeps_fd001_and_synthetic_fixture_separate():
    report = build_inventory(Path(__file__).parents[1])
    datasets = {item["descriptor"]["namespace"]: item for item in report["datasets"]}
    assert datasets["fd001_benchmark"]["descriptor"]["classification"] == "benchmark"
    assert datasets["fd001_benchmark"]["suitability"]["nlp"] is False
    assert datasets["fd001_nlp_fixture"]["descriptor"]["classification"] == "synthetic_fixture"
    assert datasets["fd001_nlp_fixture"]["suitability"]["genuine_multimodal"] is False
    assert report["policy"]["cross_dataset_pairing_inferred"] is False
