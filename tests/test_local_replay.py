import tempfile
import unittest
from pathlib import Path

from iot.local_replay import SensorReplayError, iter_fd001_training_messages, replay_fd001_training_messages
from iot.sensor_message import FD001_SENSOR_FEATURES, SensorMessage, validate_sensor_message


FIRST_FD001_ROW = (
    "1 1 -0.0007 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 "
    "21.61 554.36 2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 "
    "8.4195 0.03 392 2388 100.00 39.06 23.4190"
)

SECOND_FD001_ROW = (
    "1 2 0.0019 -0.0003 100.0 518.67 642.15 1591.82 1403.14 14.62 "
    "21.61 553.75 2388.04 9044.07 1.30 47.49 522.28 2388.07 8131.49 "
    "8.4318 0.03 392 2388 100.00 39.00 23.4236"
)


def write_temp_fd001(directory: Path, contents: str) -> Path:
    path = directory / "train_FD001.txt"
    path.write_text(contents, encoding="utf-8")
    return path


class TestLocalReplay(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_directory = tempfile.TemporaryDirectory()
        self.temp_path = Path(self._temp_directory.name)

    def tearDown(self) -> None:
        self._temp_directory.cleanup()

    def test_valid_replay_output_uses_selected_features_only(self) -> None:
        path = write_temp_fd001(self.temp_path, FIRST_FD001_ROW + "\n")

        messages = replay_fd001_training_messages(path)

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertIsInstance(message, SensorMessage)
        self.assertEqual(message.unit_id, 1)
        self.assertEqual(message.cycle, 1)
        self.assertEqual(tuple(message.features), FD001_SENSOR_FEATURES)
        self.assertEqual(message.features["setting_1"], -0.0007)
        self.assertEqual(message.features["sensor_2"], 641.82)
        self.assertEqual(message.features["sensor_21"], 23.419)

        flat = message.to_dict()
        self.assertNotIn("rul", flat)
        self.assertNotIn("failure_label", flat)
        self.assertNotIn("timestamp", flat)
        self.assertNotIn("sensor_1", flat)

    def test_every_replayed_message_passes_current_validator(self) -> None:
        path = write_temp_fd001(self.temp_path, FIRST_FD001_ROW + "\n" + SECOND_FD001_ROW + "\n")

        for message in iter_fd001_training_messages(path):
            normalized = validate_sensor_message(message.to_dict())
            self.assertEqual(normalized, message)

    def test_replay_limit_bounds_emitted_messages(self) -> None:
        path = write_temp_fd001(self.temp_path, FIRST_FD001_ROW + "\n" + SECOND_FD001_ROW + "\n")

        limited = replay_fd001_training_messages(path, limit=1)
        empty = replay_fd001_training_messages(path, limit=0)

        self.assertEqual([(message.unit_id, message.cycle) for message in limited], [(1, 1)])
        self.assertEqual(empty, [])

    def test_missing_input_file_is_rejected(self) -> None:
        missing_path = Path(tempfile.gettempdir()) / "missing_fd001_train_file.txt"

        with self.assertRaises(FileNotFoundError):
            replay_fd001_training_messages(missing_path, limit=1)

    def test_malformed_column_count_is_rejected(self) -> None:
        path = write_temp_fd001(self.temp_path, "1 1 -0.0007\n")

        with self.assertRaisesRegex(SensorReplayError, "expected 26 columns"):
            replay_fd001_training_messages(path)

    def test_non_numeric_feature_is_rejected(self) -> None:
        bad_row = FIRST_FD001_ROW.replace("641.82", "bad-value")
        path = write_temp_fd001(self.temp_path, bad_row + "\n")

        with self.assertRaisesRegex(SensorReplayError, "could not convert string to float"):
            replay_fd001_training_messages(path)

    def test_malformed_identifier_is_rejected(self) -> None:
        bad_row = FIRST_FD001_ROW.replace("1 1", "1.0 1", 1)
        path = write_temp_fd001(self.temp_path, bad_row + "\n")

        with self.assertRaisesRegex(SensorReplayError, "unit_id must be an integer"):
            replay_fd001_training_messages(path)

    def test_invalid_limit_is_rejected(self) -> None:
        path = write_temp_fd001(self.temp_path, FIRST_FD001_ROW + "\n")

        with self.assertRaisesRegex(SensorReplayError, "non-negative"):
            replay_fd001_training_messages(path, limit=-1)


if __name__ == "__main__":
    unittest.main()
