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

import markdown as md
from flask import Flask, abort, redirect, render_template, request, url_for

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
    sp = (APP_CFG.get("store") or {}).get("path")
    return {"config": APP_CONFIG_PATH, "store_path": sp, "viewer": "ok"}


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
