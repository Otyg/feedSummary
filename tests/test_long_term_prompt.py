import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "config/prompts/long_term/landscape_report_v1.yaml"
FIXTURE_PATH = (
    PROJECT_ROOT
    / "config/prompts/long_term/_fixtures/landscape_report_v1.yaml"
)


class LongTermPromptTests(unittest.TestCase):
    def setUp(self):
        self.prompt = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
        self.fixtures = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_prompt_has_versioned_inputs_and_machine_readable_output(self):
        self.assertEqual(1, self.prompt["schema_version"])
        self.assertEqual("landscape-report-v1", self.prompt["prompt_version"])
        placeholders = {
            "{profile_context}",
            "{analysis_period}",
            "{metrics}",
            "{coverage_warnings}",
            "{cluster_snapshots}",
            "{previous_report}",
        }
        self.assertTrue(placeholders.issubset(set(self._placeholders())))
        self.assertIn("Returnera endast ett JSON-objekt", self.prompt["system"])

    def test_output_schema_requires_evidence_and_bounded_forecasts(self):
        schema = self.prompt["output_schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("forecast", schema["required"])
        forecast = schema["properties"]["forecast"]
        self.assertEqual(3, forecast["maxItems"])
        self.assertEqual(
            ["low", "medium"],
            forecast["items"]["properties"]["confidence"]["enum"],
        )
        assessment = schema["$defs"]["evidenced_assessment"]
        self.assertIn("evidence_cluster_ids", assessment["required"])

    def test_five_initial_review_scenarios_are_present(self):
        cases = self.fixtures["fixtures"]
        self.assertEqual("landscape-report-v1", self.fixtures["prompt_version"])
        self.assertGreaterEqual(len(cases), 5)
        self.assertEqual(len(cases), len({case["name"] for case in cases}))
        names = {case["name"] for case in cases}
        self.assertIn("sparse_evidence", names)
        self.assertIn("duplicate_reporting_is_not_a_trend", names)
        self.assertIn("conflicting_sources", names)
        self.assertIn("changed_source_mix", names)

    def _placeholders(self):
        import string

        return [
            "{" + name + "}"
            for _, name, _, _ in string.Formatter().parse(self.prompt["user_template"])
            if name
        ]


if __name__ == "__main__":
    unittest.main()
