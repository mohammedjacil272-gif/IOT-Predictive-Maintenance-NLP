import unittest

from iot.hardware.simulation import SyntheticSensorSimulator


class TestSyntheticSimulation(unittest.TestCase):
    def test_cycle_covers_normal_ramp_abnormal_failure_and_recovery(self):
        sim = SyntheticSensorSimulator()
        statuses = [sim.at(index).operating_status for index in range(19)]
        self.assertEqual(statuses, ["normal"] * 6 + ["warming"] * 4 + ["abnormal"] * 4 + ["failure"] * 3 + ["recovery"] * 2)
        self.assertEqual(sim.at(19), sim.at(0))

    def test_metrics_rise_and_recover_gradually_within_contract_ranges(self):
        sim = SyntheticSensorSimulator()
        normal, warming, abnormal, failure, recovery = sim.at(0), sim.at(7), sim.at(12), sim.at(14), sim.at(18)
        self.assertLess(normal.temperature, warming.temperature)
        self.assertLess(warming.vibration, abnormal.vibration)
        self.assertEqual(failure.rpm, 2100)
        self.assertGreater(failure.pressure, recovery.pressure)
        for item in (normal, warming, abnormal, failure, recovery):
            self.assertGreaterEqual(item.temperature, -40)
            self.assertLessEqual(item.temperature, 150)
            self.assertGreaterEqual(item.vibration, 0)
            self.assertLessEqual(item.vibration, 50)

    def test_invalid_sample_number_rejected(self):
        with self.assertRaises(ValueError):
            SyntheticSensorSimulator().at(-1)
