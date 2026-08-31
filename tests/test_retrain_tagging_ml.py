import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from retrain_tagging_ml import main


class FakeStore:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeTagger:
    settings = None
    force = None
    result = True

    def __init__(self, settings):
        type(self).settings = settings

    def refresh_from_store(self, store, *, force=False):
        type(self).force = force
        return type(self).result


class RetrainTaggingMLTests(unittest.TestCase):
    def setUp(self):
        FakeTagger.settings = None
        FakeTagger.force = None
        FakeTagger.result = True

    def _write_config(self, directory: str, *, enabled: bool = True) -> Path:
        path = Path(directory) / "config.yaml"
        path.write_text(
            "\n".join(
                [
                    "store:",
                    "  provider: mongodb",
                    "  uri: mongodb://localhost:27017",
                    "  database: feedsummary",
                    "tagging:",
                    "  ml:",
                    f"    enabled: {'true' if enabled else 'false'}",
                    "    classifier: logistic_regression",
                    "    categories:",
                    "      - DOMAIN_ENTITY",
                    "    model_path: models/tagger.joblib",
                    "    min_label_support: 2",
                    "    min_training_articles: 10",
                    "llm:",
                    "  - provider: ollama_local",
                    "    embedding_model: test-embedding",
                ]
            ),
            encoding="utf-8",
        )
        return path

    def test_forces_full_retraining_and_resolves_model_relative_to_config(self):
        with TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            store = FakeStore()
            stdout = StringIO()

            with (
                patch("retrain_tagging_ml.create_store", return_value=store),
                patch("retrain_tagging_ml.EmbeddingSGDTagger", FakeTagger),
                redirect_stdout(stdout),
            ):
                exit_code = main(["--config", str(config_path)])

            self.assertEqual(0, exit_code)
            self.assertTrue(FakeTagger.force)
            self.assertEqual(
                str((Path(directory) / "models/tagger.joblib").resolve()),
                FakeTagger.settings.model_path,
            )
            self.assertEqual("test-embedding", FakeTagger.settings.embedding_model)
            self.assertTrue(store.closed)
            self.assertIn("Full omträning klar", stdout.getvalue())

    def test_disabled_ml_fails_before_opening_store(self):
        with TemporaryDirectory() as directory:
            config_path = self._write_config(directory, enabled=False)
            stderr = StringIO()

            with (
                patch("retrain_tagging_ml.create_store") as create_store,
                redirect_stderr(stderr),
            ):
                exit_code = main(["--config", str(config_path)])

            self.assertEqual(2, exit_code)
            create_store.assert_not_called()
            self.assertIn("tagging.ml.enabled", stderr.getvalue())

    def test_failed_training_closes_store_and_reports_preserved_model(self):
        with TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            store = FakeStore()
            FakeTagger.result = False
            stderr = StringIO()

            with (
                patch("retrain_tagging_ml.create_store", return_value=store),
                patch("retrain_tagging_ml.EmbeddingSGDTagger", FakeTagger),
                redirect_stderr(stderr),
            ):
                exit_code = main(["--config", str(config_path)])

            self.assertEqual(2, exit_code)
            self.assertTrue(store.closed)
            self.assertIn("befintliga modellen har behållits", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
