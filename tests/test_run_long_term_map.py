import unittest

from run_long_term_map import _map_llm_config, _map_settings


class LongTermMapCliTests(unittest.TestCase):
    def test_local_map_provider_selects_only_local_llm(self):
        config = {
            "llm": [
                {"provider": "ollama_cloud", "model": "cloud"},
                {
                    "provider": "ollama_local",
                    "model": "local",
                    "context_window_tokens": 4096,
                    "prompt_safety_margin": 400,
                },
            ],
            "llm_fallback": {"provider": "ollama_cloud"},
        }
        selected = _map_llm_config(config, "local")
        self.assertEqual("ollama_local", selected["llm"]["provider"])
        self.assertNotIn("llm_fallback", selected)

    def test_map_settings_combine_long_term_and_provider_limits(self):
        settings = _map_settings(
            {
                "llm": {"map_max_output_tokens": 900},
                "cluster_updates": {
                    "min_pending_articles": 2,
                    "max_update_delay_hours": 12,
                    "max_articles_per_call": 8,
                    "article_clip_chars": 1500,
                },
            },
            {
                "llm": {
                    "provider": "ollama_local",
                    "context_window_tokens": 4096,
                    "prompt_safety_margin": 400,
                }
            },
        )
        self.assertEqual(2, settings.min_pending_articles)
        self.assertEqual(4096, settings.max_context_tokens)
        self.assertEqual(900, settings.max_output_tokens)
        self.assertEqual(400, settings.safety_margin_tokens)


if __name__ == "__main__":
    unittest.main()
