#!/usr/bin/env python3
"""Force a complete rebuild of the configured production ML tag model."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

from feedsummary_core.llm_client import get_local_embedding_model
from feedsummary_core.persistence import create_store
from feedsummary_core.tagging_ml.embedding_sgd import (
    EmbeddingSGDSettings,
    EmbeddingSGDTagger,
)


def _resolve_config_path(raw_path: Optional[str]) -> Path:
    candidate = raw_path or os.environ.get("FEEDSUMMARY_CONFIG") or "config.yaml"
    return Path(os.path.expandvars(os.path.expanduser(candidate))).resolve()


def _resolve_relative_path(raw_path: str, config_path: Path) -> str:
    path = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Konfigurationsfilen saknas: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Konfigurationen måste vara ett YAML-objekt: {path}")
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Träna om produktionsmodellen för ML-taggning helt från lagrade "
            "artikel-embeddings och taggar."
        )
    )
    parser.add_argument(
        "--config",
        help=(
            "Sökväg till YAML-konfigurationen "
            "(standard: FEEDSUMMARY_CONFIG eller config.yaml)"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    store = None
    raw_uri = ""
    try:
        config_path = _resolve_config_path(args.config)
        config = _load_config(config_path)

        store_config = config.get("store")
        if not isinstance(store_config, dict):
            raise ValueError("Konfigurationen saknar 'store'")
        provider = str(store_config.get("provider") or store_config.get("type") or "tinydb")
        if provider.strip().lower() not in {"mongo", "mongodb", "couch", "couchdb"}:
            raise ValueError("ML-omträning stöds endast med MongoDB eller CouchDB")

        settings = EmbeddingSGDSettings.from_config(
            config,
            embedding_model=get_local_embedding_model(config),
        )
        if not settings.enabled:
            raise ValueError(
                "tagging.ml.enabled måste vara true för att träna produktionsmodellen"
            )
        settings = replace(
            settings,
            model_path=_resolve_relative_path(settings.model_path, config_path),
        )

        read_only_store_config = dict(store_config)
        read_only_store_config["initialize_schema"] = False
        raw_uri = str(
            read_only_store_config.get("uri")
            or read_only_store_config.get("url")
            or ""
        )
        store = create_store(read_only_store_config)

        tagger = EmbeddingSGDTagger(settings)
        if not tagger.refresh_from_store(store, force=True):
            raise RuntimeError(
                "Träningsunderlaget räcker inte för den aktuella konfigurationen; "
                "den befintliga modellen har behållits"
            )

        print(f"Full omträning klar. Modell: {settings.model_path}")
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary converts failures to exit codes
        message = str(error)
        if raw_uri:
            message = message.replace(raw_uri, "<storage-uri>")
        print(
            f"Omträningen misslyckades ({type(error).__name__}): {message}",
            file=sys.stderr,
        )
        return 2
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
