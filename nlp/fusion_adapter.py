"""Convert NLP artifacts into validated fusion text records.

This module converts formats only. It does not establish that a maintenance
log genuinely corresponds to a sensor observation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader)


def _required_text(row: dict[str, Any], field: str, source: str) -> str:
    value = row.get(field)
    if value is None or not str(value).strip():
        raise ValueError(f"Missing {field} in {source}")
    return str(value).strip()


def _integer(value: Any, field: str, source: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {field} in {source}: {value!r}") from None

    if str(value).strip() != str(parsed):
        raise ValueError(f"Invalid {field} in {source}: {value!r}")
    return parsed


def _load_entities(path: Path) -> dict[str, list[dict[str, Any]]]:
    entities: dict[str, list[dict[str, Any]]] = {}

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                entity = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from exc

            if not isinstance(entity, dict):
                raise ValueError(
                    f"Entity at line {line_number} must be a JSON object"
                )

            log_id = _required_text(entity, "log_id", f"{path}:{line_number}")
            entities.setdefault(log_id, []).append(entity)

    return entities


def build_text_records(
    embedding_path: str | Path,
    metadata_path: str | Path,
    severity_path: str | Path,
    entities_path: str | Path,
) -> list[dict[str, Any]]:
    """Load and validate NLP artifacts for the fusion interface.

    The output uses the fusion builder's ``split`` field. It does not claim
    that a record is genuinely paired with sensor data.
    """
    embedding_path = Path(embedding_path)
    metadata_path = Path(metadata_path)
    severity_path = Path(severity_path)
    entities_path = Path(entities_path)

    for path in (embedding_path, metadata_path, severity_path, entities_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required NLP artifact not found: {path}")

    embeddings = torch.load(
        embedding_path, map_location="cpu", weights_only=True
    )
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("Embeddings artifact must contain a tensor")
    if embeddings.ndim != 2 or embeddings.shape[1] != 768:
        raise ValueError(
            f"Expected embeddings shape [n, 768], got {tuple(embeddings.shape)}"
        )
    if not torch.is_floating_point(embeddings):
        raise ValueError("Embeddings must be floating-point values")
    if not torch.isfinite(embeddings).all().item():
        raise ValueError("Embeddings contain NaN or Inf")

    metadata = _read_csv(metadata_path)
    severity_rows = _read_csv(severity_path)
    entities_by_id = _load_entities(entities_path)

    if len(metadata) != len(embeddings):
        raise ValueError("Metadata and embedding row counts differ")

    severity_by_id: dict[str, dict[str, str]] = {}
    for row in severity_rows:
        log_id = _required_text(row, "log_id", "severity CSV")
        if log_id in severity_by_id:
            raise ValueError(f"Duplicate severity log_id: {log_id}")
        severity_by_id[log_id] = row

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_rows: set[int] = set()

    for row in metadata:
        log_id = _required_text(row, "log_id", "embedding metadata")
        if log_id in seen_ids:
            raise ValueError(f"Duplicate metadata log_id: {log_id}")
        seen_ids.add(log_id)

        row_index = _integer(row.get("row_index"), "row_index", log_id)
        if not 0 <= row_index < len(embeddings):
            raise ValueError(f"row_index out of range for {log_id}")
        if row_index in seen_rows:
            raise ValueError(f"Duplicate row_index: {row_index}")
        seen_rows.add(row_index)

        unit_id = _integer(row.get("unit_id"), "unit_id", log_id)
        cycle = _integer(row.get("cycle"), "cycle", log_id)
        if unit_id <= 0 or cycle <= 0:
            raise ValueError(
                f"unit_id and cycle must be positive for {log_id}"
            )

        split = _required_text(row, "source_split", log_id)
        if split not in {"train", "validation"}:
            raise ValueError(
                f"Unsupported split {split!r} for {log_id}; "
                "only train and validation are accepted"
            )

        severity = severity_by_id.get(log_id)
        if severity is None:
            raise ValueError(f"Missing severity record for {log_id}")

        # Ensure severity metadata agrees with embedding metadata.
        for field, expected in (
            ("unit_id", str(unit_id)),
            ("cycle", str(cycle)),
            ("source_split", split),
        ):
            actual = _required_text(severity, field, f"severity for {log_id}")
            if actual != expected:
                raise ValueError(
                    f"Severity {field} mismatch for {log_id}"
                )

        try:
            severity_score = float(
                _required_text(severity, "severity_score", f"severity for {log_id}")
            )
        except ValueError:
            raise ValueError(f"Invalid severity score for {log_id}") from None

        if not 0 <= severity_score <= 4:
            raise ValueError(f"Severity score outside [0, 4] for {log_id}")

        raw_terms = severity.get("matched_terms", "")
        try:
            matched_terms = json.loads(raw_terms) if raw_terms else []
        except json.JSONDecodeError:
            raise ValueError(f"Invalid matched_terms JSON for {log_id}") from None
        if not isinstance(matched_terms, list):
            raise ValueError(f"matched_terms must be a list for {log_id}")

        log_entities = entities_by_id.get(log_id, [])
        for entity in log_entities:
            for field, expected in (
                ("unit_id", str(unit_id)),
                ("cycle", str(cycle)),
                ("source_split", split),
            ):
                actual = entity.get(field)
                if actual is not None and str(actual).strip() != expected:
                    raise ValueError(
                        f"Entity {field} mismatch for {log_id}"
                    )

        records.append({
            "log_id": log_id,
            "unit_id": unit_id,
            "cycle": cycle,
            "split": split,
            "text_embedding": embeddings[row_index].to(torch.float32),
            "entities": log_entities,
            "severity_score": severity_score,
            "severity_label": severity.get("severity_label"),
            "matched_terms": matched_terms,
            "explanation": severity.get("explanation"),
        })

    if seen_rows != set(range(len(embeddings))):
        raise ValueError("Metadata row_index values must cover every embedding")

    extra_severity_ids = set(severity_by_id) - seen_ids
    if extra_severity_ids:
        raise ValueError(
            f"Severity records without embeddings: {sorted(extra_severity_ids)}"
        )

    extra_entity_ids = set(entities_by_id) - seen_ids
    if extra_entity_ids:
        raise ValueError(
            f"Entities without embedding metadata: {sorted(extra_entity_ids)}"
        )

    return records

def export_text_records(
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write validated fusion text records to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_records: list[dict[str, Any]] = []

    for record in records:
        embedding = record.get("text_embedding")

        if not isinstance(embedding, torch.Tensor):
            raise ValueError("Each text_embedding must be a torch.Tensor")
        if embedding.shape != (768,):
            raise ValueError(
                f"Expected each text_embedding to have shape (768,), "
                f"got {tuple(embedding.shape)}"
            )
        if not torch.is_floating_point(embedding):
            raise ValueError("Text embeddings must be floating-point values")
        if not torch.isfinite(embedding).all().item():
            raise ValueError("Text embeddings contain NaN or Inf")

        item = dict(record)
        item["text_embedding"] = embedding.detach().cpu().tolist()
        serializable_records.append(item)

    output_path.write_text(
        json.dumps(serializable_records, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate NLP artifacts and export fusion text records."
    )
    parser.add_argument("--embeddings", required=True, help="Path to embeddings.pt")
    parser.add_argument("--metadata", required=True, help="Path to embedding_metadata.csv")
    parser.add_argument("--severity", required=True, help="Path to severity.csv")
    parser.add_argument("--entities", required=True, help="Path to entities.jsonl")
    parser.add_argument("--output", required=True, help="Destination JSON file")
    args = parser.parse_args()

    records = build_text_records(
        embedding_path=args.embeddings,
        metadata_path=args.metadata,
        severity_path=args.severity,
        entities_path=args.entities,
    )
    output_path = export_text_records(records, args.output)
    print(f"Exported {len(records)} validated text records to {output_path}")


if __name__ == "__main__":
    main()
