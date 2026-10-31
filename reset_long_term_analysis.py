#!/usr/bin/env python3
"""Reset all persisted long-term analysis state without deleting articles."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from feedsummary_core.persistence import create_store


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise TypeError("Konfigurationen måste vara ett YAML-objekt")
    return config


def _store_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    raw = config.get("store")
    if not isinstance(raw, dict):
        raise TypeError("Konfigurationen saknar ett giltigt 'store'-objekt")
    result = dict(raw)
    if result.get("path"):
        path = Path(str(result["path"])).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        result["path"] = str(path.resolve())
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Radera state; utan denna flagga görs endast en dry-run",
    )
    return parser


def execute_reset(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
    store = create_store(_store_config(config, config_path))
    try:
        counts = store.reset_long_term_analysis(dry_run=not args.yes)
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    return {
        "dry_run": not args.yes,
        "deleted_or_found": counts,
        "preserved": ["articles", "tags", "article_tags", "embeddings"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = execute_reset(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"Reset av långtidsanalys misslyckades: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())