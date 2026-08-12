"""Tests for DSAC-specific training transformations."""

import unittest

from trainer import _optimization_reward


class RewardScalingTests(unittest.TestCase):
    def test_scaling_is_enabled_by_default(self):
        self.assertEqual(_optimization_reward(2500.0, 10000.0), 0.25)

    def test_scaling_can_be_disabled(self):
        self.assertEqual(
            _optimization_reward(2500.0, 10000.0, reward_scaling=False),
            2500.0,
        )

    def test_invalid_divisor_is_rejected_only_when_scaling(self):
        with self.assertRaises(ValueError):
            _optimization_reward(1.0, 0.0)
        self.assertEqual(_optimization_reward(1.0, 0.0, False), 1.0)


if __name__ == "__main__":
    unittest.main()
