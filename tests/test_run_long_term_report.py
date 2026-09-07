import unittest

from run_long_term_report import (
    _model_label,
    _reduce_llm_config,
    _reduce_settings,
)


class LongTermReportCliTests(unittest.TestCase):
    def test_local_reduce_provider_selects_only_local_llm(self):
        config = {
            "llm": [
                {"provider": "ollama_cloud", "model": "cloud"},
                {"provider": "ollama_local", "model": "local"},
            ],
            "llm_fallback": {"provider": "ollama_cloud", "model": "fallback"},
        }
        selected = _reduce_llm_config(config, "local")

        self.assertEqual("ollama_local", selected["llm"]["provider"])
        self.assertNotIn("llm_fallback", selected)

    def test_reduce_budget_uses_safest_limits_across_provider_chain(self):
        llm_config = {
            "llm": [
                {
                    "provider": "ollama_cloud",
                    "model": "large",
                    "context_window_tokens": 16000,
                    "prompt_safety_margin": 400,
                },
                {
                    "provider": "ollama_local",
                    "model": "small",
                    "context_window_tokens": 8000,
                    "prompt_safety_margin": 600,
                },
            ]
        }
        settings = _reduce_settings(
            {
                "llm": {"reduce_max_output_tokens": 1800},
                "reports": {
                    "max_snapshots_per_segment": 8,
                    "max_snapshot_records": 500,
                },
            },
            llm_config,
        )

        self.assertEqual(8000, settings.max_context_tokens)
        self.assertEqual(600, settings.safety_margin_tokens)
        self.assertEqual(1800, settings.max_output_tokens)
        self.assertEqual(8, settings.max_snapshots_per_segment)
        self.assertEqual(
            "ollama_cloud:large -> ollama_local:small",
            _model_label(llm_config),
        )


if __name__ == "__main__":
    unittest.main()
