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

"""Read-only FeedSummary viewer.

The module deliberately builds a separate Flask application. It reuses the
regular viewer's read handlers and templates, but it does not register status,
resume, or schedule-trigger routes that communicate with the worker.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, abort, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

# Importing the regular viewer normally initializes its writable application
# state immediately for WSGI. Suppress that side effect here; this module sets up
# the shared read handlers with category initialization disabled.
_previous_auto_init = os.environ.get("FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT")
os.environ["FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT"] = "1"
try:
    from web_viewer import webapp_viewer as viewer
finally:
    if _previous_auto_init is None:
        os.environ.pop("FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT", None)
    else:
        os.environ["FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT"] = _previous_auto_init


BASE_DIR = Path(__file__).resolve().parent
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# Explicit allowlist: worker-related handlers can never be pulled into this app
# just because the regular viewer gains another route.
READ_ROUTES = (
    ("/api/v1/summaries", "api_summaries"),
    ("/api/v1/summaries/latest", "api_summaries_latest"),
    ("/api/v1/summary/<summary_id>", "api_summary"),
    ("/api/v1/long-term/reports", "api_long_term_reports"),
    ("/api/v1/long-term/attack-pressure", "api_long_term_attack_pressure"),
    ("/api/v1/long-term/reports/<report_id>", "api_long_term_report"),
    ("/api/v1/long-term/runs", "api_long_term_runs"),
    ("/api/v1/long-term/snapshots/<snapshot_id>", "api_long_term_snapshot"),
    ("/api/v1/long-term/clusters", "api_long_term_clusters"),
    ("/api/v1/long-term/clusters/<cluster_id>", "api_long_term_cluster"),
    ("/api/v1/long-term/reviews/<cluster_id>", "api_long_term_cluster_review"),
    ("/api/v1/articles", "api_articles"),
    ("/api/v1/article/<article_id>", "api_article"),
    ("/api/v1/pages/source", "api_page_source"),
    ("/api/v1/pages/license", "api_page_license"),
    ("/api/v1/prompt/<name>", "api_prompt_package"),
    ("/api/v1/ui_options", "api_ui_options"),
    ("/api/v1/tags", "api_get_all_tags"),
    ("/api/v1/article/<article_id>/tags", "api_get_article_tags"),
    ("/api/v1/tags/<int:tag_id>", "api_get_tag"),
    ("/api/v1/tags/<int:tag_id>/relations", "api_get_tag_relations"),
    ("/api/v1/categories", "api_get_all_categories"),
    ("/api/v1/categories/<int:category_id>", "api_get_category"),
    ("/api/v1/categories/<category_name>/tags", "api_get_tags_for_category"),
    ("/rss/summaries.xml", "rss_summaries"),
    ("/rss/articles.xml", "rss_articles"),
    ("/", "index"),
    ("/summaries", "list_summaries"),
    ("/summary/<summary_id>", "view_summary"),
    ("/summary/<summary_id>/proofread-audit", "view_summary_proofread_audit"),
    ("/long-term", "list_long_term_reports"),
    ("/long-term/report/<report_id>", "view_long_term_report"),
    ("/long-term/snapshot/<snapshot_id>", "view_long_term_snapshot"),
    ("/long-term/clusters", "list_long_term_clusters"),
    ("/long-term/review/<cluster_id>", "review_long_term_cluster"),
    ("/long-term/cluster/<cluster_id>", "view_long_term_cluster"),
    ("/articles", "list_articles"),
    ("/api/v1/article-filter-tags", "api_article_filter_tags"),
    ("/article/<article_id>", "view_article"),
    ("/tags", "manage_tags"),
    ("/categories", "manage_categories"),
    ("/tag-categories", "categorize_tags"),
    ("/license", "view_license"),
    ("/source", "view_source"),
    ("/<schema_name>", "view_latest_for_schema"),
)

WRITE_ROUTES = (
    ("/api/v1/long-term/clusters/<cluster_id>/review", "POST"),
    ("/api/v1/long-term/clusters/<cluster_id>/members", "POST"),
    ("/api/v1/tags/categories", "PUT"),
    ("/api/v1/tags", "POST"),
    ("/api/v1/article/<article_id>/tags", "POST"),
    ("/api/v1/article/<article_id>/tags/<int:tag_id>", "DELETE"),
    ("/api/v1/article/<article_id>/reclassify", "POST"),
    ("/api/v1/tags/<int:tag_id>", "PUT"),
    ("/api/v1/tags/<int:tag_id>", "DELETE"),
    ("/api/v1/tags/<int:tag_id>/relations", "PUT"),
    ("/api/v1/categories", "POST"),
    ("/api/v1/categories/<int:category_id>", "PUT"),
    ("/api/v1/categories/<int:category_id>", "DELETE"),
)

WORKER_PATH_PREFIXES = (
    "/api/v1/job/resume",
    "/api/v1/schedule/trigger",
)


def _read_only_response():
    return (
        jsonify({"error": "read_only", "detail": "Writing is disabled"}),
        401,
    )


def create_app(config_path: Optional[str] = None) -> Flask:
    read_only_app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    read_only_app.wsgi_app = ProxyFix(
        read_only_app.wsgi_app,
        x_host=1,
        x_port=1,
        x_proto=1,
        x_prefix=1,
    )
    read_only_app.config["FEEDSUMMARY_CONFIG_PATH"] = config_path

    @read_only_app.before_request
    def enforce_read_only():
        # Worker functionality is absent, including its read/status proxies.
        if request.path == "/status" or request.path.startswith(
            WORKER_PATH_PREFIXES
        ):
            abort(404)

        # Defense in depth for any unsafe URL not present in WRITE_ROUTES.
        if request.method not in SAFE_METHODS:
            return _read_only_response()

        if viewer.APP_STORE is None:
            selected_config = read_only_app.config.get("FEEDSUMMARY_CONFIG_PATH")
            viewer.init_app_state(
                viewer._resolve_config_path(selected_config),
                initialize_categories=False,
            )
        return None

    @read_only_app.context_processor
    def read_only_template_context():
        long_term = viewer.APP_CFG.get("long_term_analysis") or {}
        return {
            "read_only": True,
            "long_term_ui_enabled": viewer._long_term_ui_enabled(),
            "long_term_review_actions_enabled": False,
            "long_term_mode": (
                str(long_term.get("mode") or "")
                if isinstance(long_term, dict)
                else ""
            ),
        }

    for rule, endpoint in READ_ROUTES:
        read_only_app.add_url_rule(
            rule,
            endpoint=endpoint,
            view_func=viewer.app.view_functions[endpoint],
            methods=["GET"],
        )

    for index, (rule, method) in enumerate(WRITE_ROUTES):
        read_only_app.add_url_rule(
            rule,
            endpoint=f"read_only_write_{index}",
            view_func=_read_only_response,
            methods=[method],
        )

    return read_only_app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="FeedSummary read-only Viewer WebApp")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "5000"))
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.config["FEEDSUMMARY_CONFIG_PATH"] = args.config
    app.run(host=args.host, port=args.port, debug=bool(args.debug))
