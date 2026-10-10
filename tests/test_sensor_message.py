import math
import unittest

from iot.sensor_message import (
    FD001_SENSOR_FEATURES,
    SensorMessage,
    SensorMessageValidationError,
    validate_sensor_message,
)


def valid_message() -> dict[str, int | float]:
    message: dict[str, int | float] = {"unit_id": 1, "cycle": 30}
    for index, feature in enumerate(FD001_SENSOR_FEATURES, start=1):
        message[feature] = float(index)
    return message


class TestSensorMessage(unittest.TestCase):
    def test_valid_message_returns_normalized_message(self) -> None:
        result = validate_sensor_message(valid_message())

        self.assertIsInstance(result, SensorMessage)
        self.assertEqual(result.unit_id, 1)
        self.assertEqual(result.cycle, 30)
        self.assertEqual(tuple(result.features), FD001_SENSOR_FEATURES)
        self.assertTrue(all(isinstance(value, float) for value in result.features.values()))
        self.assertEqual(result.to_dict()["sensor_21"], 17.0)

    def test_missing_feature_is_rejected(self) -> None:
        message = valid_message()
        del message["sensor_21"]

        with self.assertRaisesRegex(SensorMessageValidationError, "Missing required fields"):
            validate_sensor_message(message)

    def test_unexpected_feature_is_rejected(self) -> None:
        message = valid_message()
        message["sensor_1"] = 0.0

        with self.assertRaisesRegex(SensorMessageValidationError, "Unexpected fields"):
            validate_sensor_message(message)

    def test_timestamp_is_rejected_as_unexpected_field(self) -> None:
        message = valid_message()
        message["timestamp"] = 1_704_067_200.0

        with self.assertRaisesRegex(SensorMessageValidationError, "Unexpected fields"):
            validate_sensor_message(message)

    def test_non_positive_unit_id_is_rejected(self) -> None:
        message = valid_message()
        message["unit_id"] = 0

        with self.assertRaisesRegex(SensorMessageValidationError, "unit_id must be greater than 0"):
            validate_sensor_message(message)

    def test_non_positive_cycle_is_rejected(self) -> None:
        message = valid_message()
        message["cycle"] = -1

        with self.assertRaisesRegex(SensorMessageValidationError, "cycle must be greater than 0"):
            validate_sensor_message(message)

    def test_boolean_identifier_is_rejected(self) -> None:
        message = valid_message()
        message["unit_id"] = True

        with self.assertRaisesRegex(SensorMessageValidationError, "unit_id must be a positive integer"):
            validate_sensor_message(message)

    def test_fractional_identifier_is_rejected(self) -> None:
        message = valid_message()
        message["cycle"] = 30.5

        with self.assertRaisesRegex(SensorMessageValidationError, "cycle must be a positive integer"):
            validate_sensor_message(message)

    def test_string_identifier_is_rejected(self) -> None:
        message = valid_message()
        message["unit_id"] = "1"

        with self.assertRaisesRegex(SensorMessageValidationError, "unit_id must be a positive integer"):
            validate_sensor_message(message)

    def test_boolean_feature_value_is_rejected(self) -> None:
        message = valid_message()
        message["sensor_2"] = False

        with self.assertRaisesRegex(SensorMessageValidationError, "sensor_2 must be a finite numeric value"):
            validate_sensor_message(message)

    def test_non_numeric_feature_value_is_rejected(self) -> None:
        message = valid_message()
        message["sensor_2"] = "642.15"

        with self.assertRaisesRegex(SensorMessageValidationError, "sensor_2 must be a finite numeric value"):
            validate_sensor_message(message)

    def test_nan_feature_value_is_rejected(self) -> None:
        message = valid_message()
        message["sensor_2"] = math.nan

        with self.assertRaisesRegex(SensorMessageValidationError, "sensor_2 must be finite"):
            validate_sensor_message(message)

    def test_positive_infinity_feature_value_is_rejected(self) -> None:
        message = valid_message()
        message["sensor_2"] = math.inf

        with self.assertRaisesRegex(SensorMessageValidationError, "sensor_2 must be finite"):
            validate_sensor_message(message)

    def test_negative_infinity_feature_value_is_rejected(self) -> None:
        message = valid_message()
        message["sensor_2"] = -math.inf

        with self.assertRaisesRegex(SensorMessageValidationError, "sensor_2 must be finite"):
            validate_sensor_message(message)


if __name__ == "__main__":
    unittest.main()
