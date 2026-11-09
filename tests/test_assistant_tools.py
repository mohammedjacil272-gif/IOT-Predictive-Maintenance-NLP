import pytest

from assistant.storage import AssistantStorage
from assistant.tools import MaintenanceToolError, ReadOnlyMaintenanceTools


def test_machine_health_tool_is_read_only_and_allowlisted(tmp_path):
    storage = AssistantStorage(tmp_path / "assistant.sqlite3")
    storage.save_machine("m-1", {"machine_id": "m-1", "display_name": "Fixture"})
    tools = ReadOnlyMaintenanceTools(storage)
    result = tools.call("machine_health", {"machine_id": "m-1"})
    assert result["status"] == "available"
    assert tools.list_tools()[0]["read_only"] is True
    assert storage.list_machines()[0]["machine_id"] == "m-1"


def test_unknown_tools_and_machine_ids_fail_without_mutation(tmp_path):
    storage = AssistantStorage(tmp_path / "assistant.sqlite3")
    tools = ReadOnlyMaintenanceTools(storage)
    with pytest.raises(MaintenanceToolError, match="Unsupported"):
        tools.call("delete_machine", {"machine_id": "m-1"})
    with pytest.raises(MaintenanceToolError, match="non-empty"):
        tools.machine_health("")
    assert storage.list_machines() == []
