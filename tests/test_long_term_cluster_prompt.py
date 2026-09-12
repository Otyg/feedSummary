import unittest
from pathlib import Path

import yaml
from feedsummary_core.prompts.loader import load_prompt_package_map


class LongTermClusterPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.package = yaml.safe_load(
            (root / "config/prompts/long_term/cluster_update_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        cls.fixtures = yaml.safe_load(
            (
                root
                / "config/prompts/long_term/_fixtures/cluster_update_v1.yaml"
            ).read_text(encoding="utf-8")
        )

    def test_prompt_is_versioned_and_requires_evidence_bound_json(self):
        self.assertEqual("cluster-update-v1", self.package["prompt_version"])
        self.assertEqual(0.0, self.package["temperature"])
        self.assertIn("allowed_article_ids", self.package["user_template"])
        self.assertIn("Returnera endast ett JSON-objekt", self.package["system"])
        required = set(self.package["output_schema"]["required"])
        self.assertIn("membership_revision", required)
        self.assertIn("insufficient_evidence", required)
        fact_schema = self.package["output_schema"]["properties"]["facts"]["items"]
        self.assertIn("evidence_article_ids", fact_schema["required"])

    def test_prompt_has_initial_regression_fixtures(self):
        fixtures = self.fixtures["fixtures"]
        self.assertEqual(5, len(fixtures))
        self.assertEqual(5, len({fixture["name"] for fixture in fixtures}))

    def test_private_fixture_directory_is_not_loaded_as_prompt_packages(self):
        prompt_root = Path(__file__).resolve().parents[1] / "config/prompts"

        packages = load_prompt_package_map(prompt_root)

        self.assertIn("cluster_update_v1", packages)
        self.assertEqual(
            "cluster-update-v1", packages["cluster_update_v1"]["prompt_version"]
        )
        self.assertEqual(
            "cluster-update-v2", packages["cluster_update_v2"]["prompt_version"]
        )

    def test_v2_bounds_structured_output_for_cloud_generation(self):
        root = Path(__file__).resolve().parents[1]
        package = yaml.safe_load(
            (root / "config/prompts/long_term/cluster_update_v2.yaml").read_text(
                encoding="utf-8"
            )
        )
        properties = package["output_schema"]["properties"]

        self.assertEqual("cluster-update-v2", package["prompt_version"])
        self.assertEqual(8, properties["facts"]["maxItems"])
        self.assertEqual(8, properties["timeline"]["maxItems"])
        self.assertEqual(6, properties["mitre_techniques"]["maxItems"])
        self.assertEqual(6, properties["uncertainties"]["maxItems"])


if __name__ == "__main__":
    unittest.main()
