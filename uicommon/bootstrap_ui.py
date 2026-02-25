# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "FeedSummary"
BUNDLED_CONFIG_TEMPLATE = "config.yaml.dist"  # <-- REPLACED config.yaml.defaults

_LOGGER = logging.getLogger(__name__)


def _setup_logging_if_needed() -> None:
    """
    Ensures bootstrap logs are visible even early in startup.
    Control via env:
      FEEDSUMMARY_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
    """
    if _LOGGER.handlers:
        return

    root = logging.getLogger()
    if not root.handlers:
        level_name = (os.environ.get("FEEDSUMMARY_LOG_LEVEL") or "INFO").upper().strip()
        level = getattr(logging, level_name, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    _LOGGER.setLevel(logging.getLogger().level)


@dataclass(frozen=True)
class RuntimePaths:
    """
    base_dir: where bundled resources live (PyInstaller _MEIPASS) or repo root
    app_data_dir: per-user app data dir (FeedSummary/)
    config_path: resolved config.yaml to use
    is_frozen: running as PyInstaller/frozen
    """

    base_dir: Path
    app_data_dir: Path
    config_path: Path
    is_frozen: bool


def is_pyinstaller() -> bool:
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def _bundled_base_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS")) if hasattr(sys, "_MEIPASS") else Path.cwd()


def _repo_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def get_app_data_dir(app_name: str = APP_NAME) -> Path:
    """
    Windows: %APPDATA%/FeedSummary
    macOS: ~/Library/Application Support/FeedSummary
    Linux: $XDG_DATA_HOME/FeedSummary or ~/.local/share/FeedSummary
    """
    system = platform.system().lower()

    if system == "windows":
        appdata = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
        if not appdata:
            appdata = str(Path.home() / "AppData" / "Roaming")
        return Path(appdata) / app_name

    if system == "darwin":
        return Path.home() / "Library" / "Application Support" / app_name

    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / app_name
    return Path.home() / ".local" / "share" / app_name


def resource_path(rel: str) -> Path:
    """
    Resolve a relative resource path regardless of frozen or source run.
    """
    base = _bundled_base_dir() if is_pyinstaller() else _repo_base_dir()
    return (base / rel).resolve()


def _safe_copy2(
    src: Path, dst: Path, *, retries: int = 12, sleep_s: float = 0.12
) -> bool:
    """
    Copy with retries. In frozen/PyInstaller, AV/Defender can briefly lock files in _MEI*.
    Never crash startup due to transient PermissionError.

    Returns True if copied, False if failed.
    """
    if not src.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for i in range(retries):
        try:
            shutil.copy2(src, dst)
            return True
        except (PermissionError, OSError) as e:
            last_err = e
            time.sleep(sleep_s * (i + 1))

    _LOGGER.warning("Copy failed after retries: %s -> %s (%s)", src, dst, last_err)
    return False


def _safe_write_text(dst: Path, text: str) -> bool:
    """
    Best-effort write without crashing bootstrap.
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
        return True
    except Exception as e:
        _LOGGER.warning("Write failed: %s (%s)", dst, e)
        return False


def _copy_tree(src_dir: Path, dst_dir: Path) -> None:
    """
    Copy a directory tree. If dst exists, merge (do not delete).
    Robust: never crash on copy errors.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.rglob("*"):
        rel = item.relative_to(src_dir)
        target = dst_dir / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        ok = _safe_copy2(item, target)
        if not ok:
            _LOGGER.warning("Skipped copying resource (unreadable/locked?): %s", item)


def ensure_app_data_initialized(app_name: str = APP_NAME) -> Path:
    """
    Ensure the app data dir exists and is populated.

    - Idempotent: even if AppData exists, ensure baseline resources are present.
    - Robust: never crash if bundled resources (in _MEI*) are temporarily locked.
    """
    _setup_logging_if_needed()
    app_dir = get_app_data_dir(app_name)

    if app_dir.exists():
        _LOGGER.info("AppData exists: %s", app_dir)
    else:
        _LOGGER.info("AppData missing; creating: %s", app_dir)
        app_dir.mkdir(parents=True, exist_ok=True)

    # Copy config.yaml.dist (if present) -> AppData, if missing
    dist_src = resource_path("config.yaml.dist")
    dist_dst = app_dir / "config.yaml.dist"
    if not dist_dst.exists():
        if dist_src.exists():
            if _safe_copy2(dist_src, dist_dst):
                _LOGGER.info("Copied config.yaml.dist -> %s", dist_dst)
            else:
                _LOGGER.warning(
                    "Could not copy config.yaml.dist from bundle: %s", dist_src
                )
        else:
            _LOGGER.warning("Missing bundled config.yaml.dist at: %s", dist_src)

    # Copy config directory (if present) -> merge into AppData/config
    config_dir_src = resource_path("config")
    config_dir_dst = app_dir / "config"
    if config_dir_src.exists() and config_dir_src.is_dir():
        _copy_tree(config_dir_src, config_dir_dst)
        _LOGGER.info("Ensured config/ -> %s", config_dir_dst)
    else:
        _LOGGER.warning("Missing bundled config/ dir at: %s", config_dir_src)

    return app_dir


def _ensure_user_config_from_template(app_dir: Path) -> Path:
    """
    Ensure app_dir/config.yaml exists.

    Replacement for the old config.yaml.defaults flow:
      - Use config.yaml.dist as template (bundled + copied to AppData).
      - Avoid reading config directly from _MEI* (can be locked).
      - Never crash: if copying fails, write a minimal YAML config.

    Returns the config.yaml path (in AppData).
    """
    user_cfg = app_dir / "config.yaml"
    if user_cfg.exists():
        return user_cfg

    # 1) Try to create from AppData/config.yaml.dist (preferred, stable location)
    dist_in_appdata = app_dir / "config.yaml.dist"
    if dist_in_appdata.exists():
        if _safe_copy2(dist_in_appdata, user_cfg):
            _LOGGER.info("Created user config from AppData template: %s", user_cfg)
            return user_cfg
        _LOGGER.warning(
            "Could not copy AppData template to user config: %s", dist_in_appdata
        )

    # 2) Try to create from bundled template (may be in _MEI*)
    bundled_template = resource_path(BUNDLED_CONFIG_TEMPLATE)
    if bundled_template.exists():
        if _safe_copy2(bundled_template, user_cfg):
            _LOGGER.info("Created user config from bundled template: %s", user_cfg)
            return user_cfg
        _LOGGER.warning(
            "Could not copy bundled template to user config: %s", bundled_template
        )

    # 3) Last resort: write a minimal config so app can still start and show UI/errors
    minimal = (
        "# Auto-generated minimal config because template could not be copied.\n"
        "store: {}\n"
        "ingest:\n"
        "  lookback: 24h\n"
        "prompts:\n"
        "  path: config/prompts.yaml\n"
    )
    if _safe_write_text(user_cfg, minimal):
        _LOGGER.warning(
            "Wrote minimal user config (template unavailable): %s", user_cfg
        )
        return user_cfg

    # Absolute last resort: return AppData path anyway (caller will likely fail later)
    _LOGGER.error("Failed to create any user config at: %s", user_cfg)
    return user_cfg


def resolve_config_path(app_name: str = APP_NAME) -> RuntimePaths:
    """
    Shared startup logic.

    If frozen (PyInstaller):
      1) Ensure app data dir exists (create + copy config.yaml.dist + config/) [robust]
      2) Ensure app_data/config.yaml exists from config.yaml.dist [robust]
      3) Use app_data/config.yaml

    If not frozen:
      1) If FEEDSUMMARY_CONFIG is set -> use it
      2) else if ./config.yaml exists -> use it
      3) else use ./config.yaml.dist if exists
    """
    _setup_logging_if_needed()

    frozen = is_pyinstaller()
    base_dir = _bundled_base_dir() if frozen else _repo_base_dir()
    app_dir = get_app_data_dir(app_name)

    _LOGGER.info("Bootstrap start (frozen=%s, base_dir=%s)", frozen, base_dir)
    _LOGGER.info("AppData dir resolved: %s", app_dir)

    env_cfg = os.environ.get("FEEDSUMMARY_CONFIG")
    if env_cfg:
        p = Path(env_cfg).expanduser().resolve()
        _LOGGER.info("FEEDSUMMARY_CONFIG set -> using: %s", p)
        return RuntimePaths(
            base_dir=base_dir,
            app_data_dir=app_dir,
            config_path=p,
            is_frozen=frozen,
        )

    if frozen:
        app_dir = ensure_app_data_initialized(app_name)
        cfg_path = _ensure_user_config_from_template(app_dir)

        if cfg_path.exists():
            _LOGGER.info("Using config: %s", cfg_path)
        else:
            _LOGGER.error("Config path does not exist: %s", cfg_path)

        return RuntimePaths(
            base_dir=base_dir,
            app_data_dir=app_dir,
            config_path=cfg_path,
            is_frozen=True,
        )

    # source-mode defaults
    repo = _repo_base_dir()
    cfg = repo / "config.yaml"
    dist = repo / "config.yaml.dist"

    if cfg.exists():
        cfg_path = cfg
        _LOGGER.info("Source-mode -> using repo config.yaml: %s", cfg_path)
    elif dist.exists():
        cfg_path = dist
        _LOGGER.info(
            "Source-mode -> config.yaml missing; using config.yaml.dist: %s", cfg_path
        )
    else:
        cfg_path = dist
        _LOGGER.warning(
            "Source-mode -> no config.yaml or config.yaml.dist found; expected at: %s",
            cfg_path,
        )

    return RuntimePaths(
        base_dir=repo,
        app_data_dir=app_dir,
        config_path=cfg_path,
        is_frozen=False,
    )
