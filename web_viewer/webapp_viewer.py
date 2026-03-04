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

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import lru_cache
import markdown as md
from flask import Flask, abort, redirect, render_template, request, url_for, jsonify
import requests
import yaml

from uicommon import format_ts, get_store, load_config

logger = logging.getLogger(__name__)

# Use absolute paths so templates/static work no matter working directory
BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# Global app state (loaded once)
APP_CONFIG_PATH: str = ""
APP_CFG: Dict[str, Any] = {}
APP_STORE = None


@lru_cache(maxsize=8)
def _load_static_md(filename: str) -> str:
    p = BASE_DIR / "static" / filename
    return p.read_text(encoding="utf-8")


def _resolve_path_from_cwd(p: str) -> str:
    pp = Path(os.path.expandvars(os.path.expanduser(p)))
    if not pp.is_absolute():
        pp = (Path.cwd() / pp).resolve()
    return str(pp)


def _resolve_config_path(cli_path: Optional[str] = None) -> str:
    """
    Priority:
      1) CLI --config (only in __main__)
      2) ENV FEEDSUMMARY_CONFIG
      3) ./config.yaml (cwd), else ./config.yaml.dist
    """
    if cli_path:
        return _resolve_path_from_cwd(cli_path)

    env = os.environ.get("FEEDSUMMARY_CONFIG", "").strip()
    if env:
        return _resolve_path_from_cwd(env)

    p = (Path.cwd() / "config.yaml").resolve()
    if p.exists():
        return str(p)

    pd = (Path.cwd() / "config.yaml.dist").resolve()
    return str(pd) if pd.exists() else str(p)


def _abspath_cfg_paths(cfg: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    """
    Resolve selected cfg paths relative to the directory of config.yaml,
    so running from a different CWD doesn't accidentally use a different DB.
    """
    base = Path(config_path).resolve().parent

    def abs_if_rel(p: str) -> str:
        pp = Path(os.path.expandvars(os.path.expanduser(p)))
        if pp.is_absolute():
            return str(pp)
        return str((base / pp).resolve())

    cfg2 = dict(cfg)

    st = cfg2.get("store")
    if isinstance(st, dict) and st.get("path"):
        st2 = dict(st)
        st2["path"] = abs_if_rel(str(st2["path"]))
        cfg2["store"] = st2

    pr = cfg2.get("prompts")
    if isinstance(pr, dict) and pr.get("path"):
        pr2 = dict(pr)
        pr2["path"] = abs_if_rel(str(pr2["path"]))
        cfg2["prompts"] = pr2

    sch = cfg2.get("scheduler")
    if isinstance(sch, dict) and sch.get("path"):
        sch2 = dict(sch)
        sch2["path"] = abs_if_rel(str(sch2["path"]))
        cfg2["scheduler"] = sch2

    return cfg2


def init_app_state(config_path: str) -> None:
    global APP_CONFIG_PATH, APP_CFG, APP_STORE
    APP_CONFIG_PATH = str(Path(config_path).resolve())
    raw = load_config(APP_CONFIG_PATH)
    APP_CFG = _abspath_cfg_paths(raw, APP_CONFIG_PATH)
    APP_STORE = get_store(APP_CFG)

    sp = (APP_CFG.get("store") or {}).get("path")
    logger.info("Viewer config loaded: %s", APP_CONFIG_PATH)
    logger.info("Resolved store path: %s", sp)


def _md_to_html(text: str) -> str:
    return md.markdown(text or "", extensions=["extra"])


def _get_latest_summary(store) -> Optional[Dict[str, Any]]:
    """
    Robust: try get_latest_summary_doc; otherwise pick newest from list and refetch if needed.
    """
    fn = getattr(store, "get_latest_summary_doc", None)
    if callable(fn):
        try:
            d = fn()  # type: ignore
            if isinstance(d, dict) and (d.get("summary") or "").strip():
                return d
            if isinstance(d, dict) and d.get("id"):
                d2 = store.get_summary_doc(str(d.get("id")))
                if isinstance(d2, dict) and (d2.get("summary") or "").strip():
                    return d2
                return d
        except Exception:
            pass

    docs = store.list_summary_docs() or []
    docs = [d for d in docs if isinstance(d, dict)]
    if not docs:
        return None

    docs.sort(key=lambda x: int(x.get("created") or 0), reverse=True)

    for cand in docs[:10]:
        if (cand.get("summary") or "").strip():
            return cand
        sid = str(cand.get("id") or "").strip()
        if not sid:
            continue
        try:
            d = store.get_summary_doc(sid)
            if isinstance(d, dict):
                return d
        except Exception:
            continue

    return docs[0]


def _summary_list_item(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": d.get("id"),
        "created": int(d.get("created") or 0),
        "sources_count": len(d.get("sources") or []),
        "title": d.get("title") or "",
    }


def _article_list_item(a: Dict[str, Any]) -> Dict[str, Any]:
    text = str(a.get("text") or "")
    preview = (text[:400]).replace("\n", " ")
    return {
        "id": a.get("id"),
        "title": a.get("title") or "",
        "source": a.get("source") or "",
        "url": a.get("url") or "",
        "published_ts": int(a.get("published_ts") or 0),
        "fetched_at": int(a.get("fetched_at") or 0),
        "preview": preview,
    }


def _load_yaml_file(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _collect_ui_options(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read-only UI options for clients.
    - sources: feed/source names
    - topics: unique topic tags across feeds (if configured)
    - prompt_packages: keys in prompts.yaml
    """
    out = {"sources": [], "topics": [], "prompt_packages": []}

    # ---- Feeds: sources + topics ----
    feeds_path = None

    # common patterns in your config variants
    if isinstance(cfg.get("feeds"), dict) and cfg["feeds"].get("path"):
        feeds_path = str(cfg["feeds"]["path"])
    elif isinstance(cfg.get("feeds_path"), str) and cfg.get("feeds_path"):
        feeds_path = str(cfg["feeds_path"])
    else:
        # fallback: same dir as config.yaml
        feeds_path = str(
            (Path(APP_CONFIG_PATH).resolve().parent / "config" / "feeds.yaml").resolve()
        )

    feeds = _load_yaml_file(feeds_path)
    sources: List[str] = []
    topics: List[str] = []

    if isinstance(feeds, dict):
        for name, f in feeds.items():
            if isinstance(name, str) and name.strip():
                sources.append(name.strip())
            if isinstance(f, dict):
                ts = f.get("topics") or f.get("topic") or []
                if isinstance(ts, str) and ts.strip():
                    topics.append(ts.strip())
                elif isinstance(ts, list):
                    for t in ts:
                        t = str(t).strip()
                        if t:
                            topics.append(t)

    # ---- Prompts: prompt packages ----
    prompts_path = None
    if isinstance(cfg.get("prompts"), dict) and cfg["prompts"].get("path"):
        prompts_path = str(cfg["prompts"]["path"])
    else:
        prompts_path = str(
            (
                Path(APP_CONFIG_PATH).resolve().parent / "config" / "prompts.yaml"
            ).resolve()
        )

    prompts = _load_yaml_file(prompts_path)
    if isinstance(prompts, dict):
        out["prompt_packages"] = sorted(
            [k for k in prompts.keys() if isinstance(k, str)]
        )

    out["sources"] = sorted(list(dict.fromkeys(sources)))
    out["topics"] = sorted(list(dict.fromkeys(topics)))

    out["feeds_path"] = feeds_path
    out["prompts_path"] = prompts_path
    return out


def _worker_api_base(cfg: Dict[str, Any]) -> str:
    d = cfg.get("worker_api")
    if not isinstance(d, dict):
        d = {}
    host = str(d.get("host") or "127.0.0.1")
    port = int(d.get("port") or 8799)
    return f"http://{host}:{port}"


@app.route("/api/v1/schedule/trigger", methods=["POST"])
def api_schedule_trigger():
    body = request.get_json(silent=True) or {}
    name = str(body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "missing_name"}), 400

    base = _worker_api_base(APP_CFG)
    try:
        r = requests.post(f"{base}/trigger", json={"name": name}, timeout=5)
        return (
            r.content,
            r.status_code,
            {"Content-Type": r.headers.get("Content-Type", "application/json")},
        )
    except Exception as e:
        return jsonify(
            {"error": "worker_unreachable", "detail": str(e), "worker": base}
        ), 502


@app.route("/api/v1/schedule/trigger/<trigger_id>", methods=["GET"])
def api_schedule_trigger_status(trigger_id: str):
    base = _worker_api_base(APP_CFG)
    try:
        r = requests.get(f"{base}/trigger/{trigger_id}", timeout=5)
        return (
            r.content,
            r.status_code,
            {"Content-Type": r.headers.get("Content-Type", "application/json")},
        )
    except Exception as e:
        return jsonify(
            {"error": "worker_unreachable", "detail": str(e), "worker": base}
        ), 502


@app.route("/api/v1/summaries")
def api_summaries():
    """
    List summaries (like sidebar/list). Newest first.
    Query:
      limit= (default 200)
    """
    store = APP_STORE
    if store is None:
        abort(500)

    try:
        limit = int(request.args.get("limit", "200"))
    except Exception:
        limit = 200
    limit = max(1, min(limit, 2000))

    docs = store.list_summary_docs() or []
    docs = [d for d in docs if isinstance(d, dict)]
    docs.sort(key=lambda d: int(d.get("created") or 0), reverse=True)
    docs = docs[:limit]

    return jsonify({"items": [_summary_list_item(d) for d in docs]})


@app.route("/api/v1/summaries/latest")
def api_summaries_latest():
    """
    Latest summary doc (the same as default page redirect target).
    """
    store = APP_STORE
    if store is None:
        abort(500)

    latest = _get_latest_summary(store)
    if not isinstance(latest, dict):
        return jsonify({"item": None}), 404

    # fetch full doc if needed
    sid = str(latest.get("id") or "").strip()
    sdoc = store.get_summary_doc(sid) if sid else latest
    if not isinstance(sdoc, dict):
        return jsonify({"item": None}), 404

    return jsonify({"item": sdoc})


@app.route("/api/v1/summary/<summary_id>")
def api_summary(summary_id: str):
    """
    Full summary doc for reading/rendering.
    """
    store = APP_STORE
    if store is None:
        abort(500)

    sid = str(summary_id).strip()
    sdoc = store.get_summary_doc(sid)
    if not isinstance(sdoc, dict):
        abort(404)

    return jsonify({"item": sdoc})


@app.route("/api/v1/articles")
def api_articles():
    """
    List articles (like /articles page).
    Query:
      limit= (default 300, max 5000)
    """
    store = APP_STORE
    if store is None:
        abort(500)

    try:
        limit = int(request.args.get("limit", "300"))
    except Exception:
        limit = 300
    limit = max(1, min(limit, 5000))

    raw = store.list_articles(limit=limit) or []
    articles = [a for a in raw if isinstance(a, dict) and a.get("id")]

    def ts(a: Dict[str, Any]) -> int:
        p = a.get("published_ts")
        if isinstance(p, int) and p > 0:
            return p
        return int(a.get("fetched_at") or 0)

    articles.sort(key=ts, reverse=True)
    return jsonify({"items": [_article_list_item(a) for a in articles]})


@app.route("/api/v1/article/<article_id>")
def api_article(article_id: str):
    """
    Full article doc for reading.
    """
    store = APP_STORE
    if store is None:
        abort(500)

    a = store.get_article(str(article_id))
    if not isinstance(a, dict):
        abort(404)

    return jsonify({"item": a})


@app.route("/api/v1/pages/source")
def api_page_source():
    return jsonify({"markdown": _load_static_md("source.md")})


@app.route("/api/v1/pages/license")
def api_page_license():
    return jsonify({"markdown": _load_static_md("license.md")})


@app.route("/api/v1/prompt/<name>")
def api_prompt_package(name: str):
    """
    Return the YAML content for a single prompt package from prompts.yaml.
    Read-only.
    """
    pkg = str(name or "").strip()
    if not pkg:
        abort(404)

    # Reuse the same prompts path logic as /api/ui_options
    prompts_path = None
    if isinstance(APP_CFG.get("prompts"), dict) and APP_CFG["prompts"].get("path"):
        prompts_path = str(APP_CFG["prompts"]["path"])
    else:
        prompts_path = str(
            (
                Path(APP_CONFIG_PATH).resolve().parent / "config" / "prompts.yaml"
            ).resolve()
        )

    p = Path(prompts_path)
    if not p.exists():
        return jsonify(
            {"error": "prompts_yaml_not_found", "prompts_path": prompts_path}
        ), 404

    prompts = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(prompts, dict) or pkg not in prompts:
        return jsonify(
            {"error": "prompt_not_found", "name": pkg, "prompts_path": prompts_path}
        ), 404

    # Dump only that package as YAML (nice for display/copy)
    one = {pkg: prompts[pkg]}
    yaml_text = yaml.safe_dump(one, sort_keys=False, allow_unicode=True)

    return jsonify(
        {
            "name": pkg,
            "prompts_path": prompts_path,
            "yaml": yaml_text,
            "item": prompts[pkg],  # also return as JSON if you want programmatic use
        }
    )


@app.route("/api/v1/ui_options")
def api_ui_options():
    """
    Read-only options used by clients (Qt remote):
      {
        sources: [...],
        topics: [...],
        prompt_packages: [...],
        feeds_path: "...",
        prompts_path: "..."
      }
    """
    return jsonify(_collect_ui_options(APP_CFG))


@app.route("/")
def index():
    store = APP_STORE
    if store is None:
        abort(500)

    latest = _get_latest_summary(store)
    if not latest:
        return render_template(
            "index.html",
            summary=None,
            html="<p>Inga summaries ännu.</p>",
            summaries=[],
            default_selected=None,
        )

    sid = str(latest.get("id") or "")
    return redirect(url_for("view_summary", summary_id=sid))


@app.route("/summaries")
def list_summaries():
    store = APP_STORE
    if store is None:
        abort(500)

    docs = store.list_summary_docs() or []
    docs = [d for d in docs if isinstance(d, dict)]
    docs.sort(key=lambda d: int(d.get("created") or 0), reverse=True)
    return render_template("summaries.html", summaries=docs)


@app.route("/summary/<summary_id>")
def view_summary(summary_id: str):
    store = APP_STORE
    if store is None:
        abort(500)

    docs = store.list_summary_docs() or []
    docs = [d for d in docs if isinstance(d, dict)]
    docs.sort(key=lambda d: int(d.get("created") or 0), reverse=True)

    sid = str(summary_id).strip()
    sdoc = None
    try:
        sdoc = store.get_summary_doc(sid)
    except Exception:
        sdoc = None

    if not sdoc:
        for d in docs:
            if str(d.get("id") or "") == sid:
                sdoc = d
                break

    if not isinstance(sdoc, dict):
        abort(404)

    summary_text = str(sdoc.get("summary") or "").strip()
    if not summary_text:
        keys = ", ".join(sorted(list(sdoc.keys())))
        summary_text = (
            "*(Ingen summary-text hittades i dokumentet.)*\n\n"
            f"- requested id: `{sid}`\n"
            f"- doc id: `{sdoc.get('id')}`\n"
            f"- created: `{sdoc.get('created')}`\n"
            f"- keys: `{keys}`\n"
        )

    html = _md_to_html(summary_text)

    return render_template(
        "index.html",
        summary=sdoc,
        html=html,
        summaries=docs,
        default_selected=sid,
        format_ts=format_ts,
    )


@app.route("/articles")
def list_articles():
    store = APP_STORE
    if store is None:
        abort(500)

    try:
        limit = int(request.args.get("limit", "300"))
    except Exception:
        limit = 300
    limit = max(1, min(limit, 5000))

    raw = store.list_articles(limit=limit) or []
    articles: List[Dict[str, Any]] = []
    for a in raw:
        if isinstance(a, dict) and a.get("id"):
            articles.append(a)

    def ts(a: Dict[str, Any]) -> int:
        p = a.get("published_ts")
        if isinstance(p, int) and p > 0:
            return p
        f = a.get("fetched_at")
        return int(f or 0)

    articles.sort(key=ts, reverse=True)
    return render_template(
        "articles.html", articles=articles, format_ts=format_ts, error=None
    )


@app.route("/article/<article_id>")
def view_article(article_id: str):
    store = APP_STORE
    if store is None:
        abort(500)

    a = None
    try:
        a = store.get_article(str(article_id))
    except Exception as e:
        return render_template(
            "article.html",
            a={
                "title": "(Kunde inte läsa artikel)",
                "source": "",
                "published_ts": 0,
                "fetched_at": 0,
                "url": "",
                "text": f"Fel vid get_article({article_id}): {e}",
            },
            format_ts=format_ts,
        ), 500

    if not isinstance(a, dict):
        abort(404)

    for k in ("title", "source", "url", "text"):
        if a.get(k) is None:
            a[k] = ""

    if not str(a.get("text") or "").strip():
        keys = ", ".join(sorted(a.keys()))
        a["text"] = (
            "⚠️ Ingen artikeltext hittades i posten.\n\n"
            f"id: {a.get('id')}\n"
            f"keys: {keys}\n"
        )

    return render_template("article.html", a=a, format_ts=format_ts)


@app.route("/status")
def status():
    viewer_info = {
        "ok": True,
        "config": APP_CONFIG_PATH,
        "store_path": (
            (APP_CFG.get("store") or {}).get("path")
            if isinstance(APP_CFG, dict)
            else None
        ),
    }

    base = _worker_api_base(APP_CFG)
    try:
        r = requests.get(f"{base}/status", timeout=3)
        r.raise_for_status()
        worker_payload = r.json()
        return jsonify({"worker": worker_payload, "viewer": viewer_info}), 200
    except Exception as e:
        # Mirror intent: status depends on worker; if worker is down, service is unavailable
        return (
            jsonify(
                {
                    "worker": None,
                    "viewer": viewer_info,
                    "error": "worker_unavailable",
                    "worker_url": base,
                    "detail": str(e),
                }
            ),
            503,
        )


@app.route("/license")
def view_license():
    store = APP_STORE
    if store is None:
        abort(500)

    docs = store.list_summary_docs() or []
    docs = [d for d in docs if isinstance(d, dict)]
    docs.sort(key=lambda d: int(d.get("created") or 0), reverse=True)

    html = _md_to_html(_load_static_md("license.md"))
    return render_template(
        "index.html",
        summary={},
        html=html,
        summaries=docs,
        default_selected="__license__",
        format_ts=format_ts,
    )


@app.route("/source")
def view_source():
    store = APP_STORE
    if store is None:
        abort(500)

    docs = store.list_summary_docs() or []
    docs = [d for d in docs if isinstance(d, dict)]
    docs.sort(key=lambda d: int(d.get("created") or 0), reverse=True)

    html = _md_to_html(_load_static_md("source.md"))
    return render_template(
        "index.html",
        summary={},
        html=html,
        summaries=docs,
        default_selected="__source__",
        format_ts=format_ts,
    )


# ---- WSGI init (gunicorn/waitress): initialize from env/cwd ----
def _wsgi_init_once() -> None:
    global APP_STORE
    if APP_STORE is not None:
        return
    cfg_path = _resolve_config_path(None)
    init_app_state(cfg_path)


_wsgi_init_once()


# ---- Dev entrypoint (optional) ----
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="FeedSummary Viewer WebApp")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5000")))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    init_app_state(_resolve_config_path(args.config))
    app.run(host=args.host, port=args.port, debug=bool(args.debug))
