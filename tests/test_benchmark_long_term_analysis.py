import json
import unittest
from unittest.mock import patch

import benchmark_long_term_analysis


class LongTermBenchmarkCliTests(unittest.TestCase):
    def test_cli_defaults_to_90_and_180_days_and_reports_capacity_result(self):
        with (
            patch(
                "benchmark_long_term_analysis.run_long_term_performance_benchmark",
                side_effect=lambda settings: {
                    "dataset": {"days": settings.days},
                    "timings_ms": {"total": float(settings.days)},
                    "peak_memory_mb": 10.0,
                },
            ) as run,
            patch("builtins.print") as output,
        ):
            exit_code = benchmark_long_term_analysis.main([])

        payload = json.loads(output.call_args.args[0])
        self.assertEqual(0, exit_code)
        self.assertTrue(payload["passed"])
        self.assertEqual([90, 180], [row["dataset"]["days"] for row in payload["results"]])
        self.assertEqual(2, run.call_count)


if __name__ == "__main__":
    unittest.main()
