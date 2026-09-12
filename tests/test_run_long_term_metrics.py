import unittest

from run_long_term_metrics import _metric_settings


class LongTermMetricsCliTests(unittest.TestCase):
    def test_report_configuration_builds_metric_settings(self):
        settings = _metric_settings(
            {
                "reports": {
                    "windows_days": [14, 60],
                    "bucket_days": 14,
                    "min_independent_events_for_trend": 4,
                    "source_concentration_threshold": 0.8,
                    "source_mix_change_threshold": 0.4,
                    "max_clusters": 500,
                    "max_memberships_per_cluster": 200,
                    "min_snapshot_member_count": 2,
                }
            }
        )

        self.assertEqual((14, 60), settings.windows_days)
        self.assertEqual(14, settings.bucket_days)
        self.assertEqual(4, settings.min_independent_events_for_trend)
        self.assertEqual(0.8, settings.source_concentration_threshold)
        self.assertEqual(0.4, settings.source_mix_change_threshold)
        self.assertEqual(500, settings.max_clusters)
        self.assertEqual(200, settings.max_memberships_per_cluster)
        self.assertEqual(2, settings.min_snapshot_member_count)


if __name__ == "__main__":
    unittest.main()
