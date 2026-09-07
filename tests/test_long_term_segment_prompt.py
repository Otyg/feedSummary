import unittest
from pathlib import Path

import yaml


class LongTermSegmentPromptTests(unittest.TestCase):
    def test_segment_prompt_preserves_snapshot_identity_and_returns_json(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "config/prompts/long_term/landscape_segment_v1.yaml"
        )
        prompt = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual("landscape-segment-v1", prompt["prompt_version"])
        self.assertIn("{segment_id}", prompt["user_template"])
        self.assertIn("{cluster_snapshots}", prompt["user_template"])
        schema = prompt["output_schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("snapshots", schema["required"])
        fields = schema["properties"]["snapshots"]["items"]["required"]
        self.assertIn("id", fields)
        self.assertIn("cluster_id", fields)


if __name__ == "__main__":
    unittest.main()
