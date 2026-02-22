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
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "FeedSummary"

_LOGGER = logging.getLogger(__name__)


def _setup_logging_if_needed() -> None:
    """
    Ensures bootstrap logs are visible even early in startup.
    Control via env:
      FEEDSUMMARY_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
    """
    if _LOGGER.handlers:
        return

    # If root has handlers, respect that and just set our level.
    root = logging.getLogger()
    if root.handlers:
        pass
    else:
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
    frozen = bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")
    return frozen


def _bundled_base_dir() -> Path:
    # Where bundled files are unpacked when frozen
    return Path(getattr(sys, "_MEIPASS")) if hasattr(sys, "_MEIPASS") else Path.cwd()


def _repo_base_dir() -> Path:
    # Repo root heuristic: directory containing this file
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


def _copy_tree(src_dir: Path, dst_dir: Path) -> None:
    """
    Copy a directory tree. If dst exists, merge (do not delete).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.rglob("*"):
        rel = item.relative_to(src_dir)
        target = dst_dir / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def ensure_app_data_initialized(app_name: str = APP_NAME) -> Path:
    """
    If the app data dir does not exist:
      - create it
      - copy config.yaml.dist and config/ into it

    Returns the app data dir path.
    """
    _setup_logging_if_needed()
    app_dir = get_app_data_dir(app_name)

    if app_dir.exists():
        _LOGGER.info("AppData exists: %s", app_dir)
        return app_dir

    _LOGGER.info("AppData missing; creating: %s", app_dir)
    app_dir.mkdir(parents=True, exist_ok=True)

    # Copy config.yaml.dist (if present in bundle/source)
    dist_src = resource_path("config.yaml.dist")
    dist_dst = app_dir / "config.yaml.dist"
    if dist_src.exists():
        shutil.copy2(dist_src, dist_dst)
        _LOGGER.info("Copied config.yaml.dist -> %s", dist_dst)
    else:
        _LOGGER.warning("Missing bundled config.yaml.dist at: %s", dist_src)

    # Copy config directory (if present)
    config_dir_src = resource_path("config")
    config_dir_dst = app_dir / "config"
    if config_dir_src.exists() and config_dir_src.is_dir():
        _copy_tree(config_dir_src, config_dir_dst)
        _LOGGER.info("Copied config/ -> %s", config_dir_dst)
    else:
        _LOGGER.warning("Missing bundled config/ dir at: %s", config_dir_src)

    return app_dir


def resolve_config_path(app_name: str = APP_NAME) -> RuntimePaths:
    """
    Shared startup logic.

    If frozen (PyInstaller):
      1) Ensure app data dir exists (create + copy config.yaml.dist + config/)
      2) If app_data/config.yaml exists -> use it
         else -> use bundled config.yaml.defaults

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

    # allow override always (useful for debugging even in frozen)
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
        user_cfg = app_dir / "config.yaml"
        defaults_cfg = resource_path("config.yaml.defaults")

        if user_cfg.exists():
            cfg_path = user_cfg
            _LOGGER.info("Using user config: %s", cfg_path)
        else:
            cfg_path = defaults_cfg
            if cfg_path.exists():
                _LOGGER.info(
                    "User config missing; using bundled defaults: %s", cfg_path
                )
            else:
                _LOGGER.error("Bundled defaults missing: %s", cfg_path)

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
