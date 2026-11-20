from pathlib import Path

import pandas as pd
import pytest

from experiments.edp_wind.schema import (
    EDPValidationError,
    EDPSourceMetadata,
    audit_workbook,
    classify_sheet,
    normalized_column,
)


class _FakeWorkbook:
    sheet_names = ["SCADA 2016", "Event logs", "Failure records"]


def _fake_read_excel(path, *, sheet_name):
    if sheet_name == "SCADA 2016":
        return pd.DataFrame({"Timestamp": ["2016-01-01T00:00:00Z"], "Turbine": ["T01"], "Power": [1.2]})
    if sheet_name == "Event logs":
        return pd.DataFrame({"Timestamp": ["2016-01-01T00:10:00Z"], "Turbine": ["T01"], "Remark": ["Alarm"]})
    return pd.DataFrame({"Timestamp": ["2016-02-01T00:00:00Z"], "Turbine": ["T01"], "Description": ["Gearbox"]})


def test_edp_audit_is_read_only_and_records_hash_schema_and_samples(tmp_path: Path):
    workbook = tmp_path / "edp.xlsx"
    workbook.write_bytes(b"fixture workbook bytes")
    report = audit_workbook(
        workbook,
        metadata=EDPSourceMetadata(dataset_version="mendeley-v1"),
        excel_file_factory=lambda path: _FakeWorkbook(),
        read_excel_fn=_fake_read_excel,
    )
    assert report["sheet_count"] == 3
    assert report["file"]["bytes"] == len(b"fixture workbook bytes")
    assert report["sheets"][0]["kind"] == "scada"
    assert report["sheets"][1]["kind"] == "event"
    assert report["sheets"][2]["kind"] == "failure"
    assert report["policy"]["timestamps_invented"] is False
    assert workbook.read_bytes() == b"fixture workbook bytes"


def test_real_temporary_xlsx_fixture_is_audited(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = tmp_path / "fixture.xlsx"
    book = openpyxl.Workbook()
    scada = book.active
    scada.title = "SCADA 2016"
    scada.append(["Timestamp", "Turbine", "Power"])
    scada.append(["2016-01-01T00:00:00Z", "T01", 1.2])
    scada.append(["2016-01-01T00:00:00Z", "T01", 1.2])
    events = book.create_sheet("Event logs")
    events.append(["Timestamp", "Turbine", "Remark"])
    events.append(["2016-01-01T00:10:00Z", "T01", "gearbox alarm"])
    book.save(workbook)
    book.close()
    report = audit_workbook(workbook)
    scada_report = report["sheets"][0]
    assert scada_report["rows"] == 2
    assert scada_report["duplicate_row_count"] == 2
    assert "Turbine" in scada_report["candidate_identifier_columns"]
    assert "Timestamp" in scada_report["candidate_timestamp_columns"]
    assert "Remark" in report["sheets"][1]["candidate_text_columns"]
    assert report["relationship_candidates"]


def test_column_normalization_and_classification():
    assert normalized_column(" Event Timestamp (UTC) ") == "event_timestamp_utc"
    assert classify_sheet("Failures", ["Turbine", "Description"]) == "failure"
    assert classify_sheet("unknown", ["value"]) == "unknown"


def test_missing_workbook_and_invalid_sample_size_fail(tmp_path: Path):
    with pytest.raises(EDPValidationError, match="missing or unsafe"):
        audit_workbook(tmp_path / "missing.xlsx", sample_rows=0)
    workbook = tmp_path / "edp.xlsx"
    workbook.write_bytes(b"fixture")
    with pytest.raises(EDPValidationError, match="sample_rows"):
        audit_workbook(workbook, sample_rows=-1)
