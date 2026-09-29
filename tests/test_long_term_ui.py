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

import os
import unittest

from feedsummary_core.long_term import validate_cluster_merge_operation

_previous_auto_init = os.environ.get("FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT")
os.environ["FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT"] = "1"
try:
    from web_viewer import webapp_viewer
finally:
    if _previous_auto_init is None:
        os.environ.pop("FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT", None)
    else:
        os.environ["FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT"] = _previous_auto_init


def _report(report_id="report-2", *, period_end_ts=2_000_000):
    return {
        "id": report_id,
        "profile_id": "healthcare_europe",
        "period_start_ts": 1_000_000,
        "period_end_ts": period_end_ts,
        "created_at": period_end_ts + 100,
        "status": "published",
        "previous_report_id": "report-1" if report_id == "report-2" else None,
        "input_snapshot_ids": ["snapshot-1", "snapshot-2", "snapshot-3"],
        "input_signature": "a" * 64,
        "metrics": {"windows": [{"days": 90}]},
        "analysis": {
            "executive_summary": "Tre oberoende händelser visar ett mönster.",
            "observations": [
                {
                    "statement": "En händelse berörde vårdsektorn.",
                    "confidence": "high",
                    "evidence_cluster_ids": ["cluster-1"],
                }
            ],
            "changes": [],
            "alternative_explanations": [],
            "data_gaps": ["Begränsad geografisk täckning"],
        },
        "forecast": [
            {
                "hypothesis": "Liknande aktivitet kan fortsätta.",
                "horizon_days": 90,
                "confidence": "low",
                "rationale": "Flera oberoende händelser stödjer hypotesen.",
                "evidence_cluster_ids": ["cluster-1", "cluster-2", "cluster-3"],
                "leading_indicators": ["Nya oberoende incidenter"],
                "invalidation_conditions": ["Inga nya incidenter observeras"],
            }
        ],
        "markdown": "# Långtidsanalys",
        "quality": {
            "validation": "passed",
            "warning_codes": ["source_concentration"],
        },
        "prompt_version": "landscape-report-v1",
        "segment_prompt_version": None,
        "model": "test-model",
    }


class _LongTermStore:
    def __init__(self):
        self.reports = [
            _report(),
            _report("report-1", period_end_ts=1_500_000),
        ]
        self.list_calls = []
        self.cluster_list_calls = []
        self.snapshot = {
            "id": "snapshot-1",
            "profile_id": "healthcare_europe",
            "cluster_id": "cluster-1",
            "membership_revision": 2,
            "prompt_version": "cluster-update-v1",
            "created_at": 1_900_000,
            "input_article_ids": ["article-1", "article-expired"],
            "payload": {
                "title": "ERPNext Server-Side Template Injection",
                "summary": "En evidensbunden klustersammanfattning.",
                "confidence": "medium",
                "insufficient_evidence": False,
                "facts": [
                    {
                        "statement": "CVE-2026-65974 rapporterades.",
                        "status": "active",
                        "evidence_article_ids": ["article-1"],
                    }
                ],
                "timeline": [],
                "mitre_techniques": [],
                "uncertainties": ["Omfattningen är ännu okänd."],
            },
        }
        self.cluster = {
            "id": "cluster-1",
            "profile_id": "healthcare_europe",
            "status": "active",
            "first_seen_ts": 1_000_000,
            "last_seen_ts": 1_900_000,
            "embedding_model": "embedding-model",
            "embedding_dimension": 2,
            "embedding_instruction": "event",
            "centroid": [1.0, 0.0],
            "vector_sum": [2.0, 0.0],
            "member_count": 2,
            "membership_revision": 2,
            "algorithm_version": "online-centroid-v1",
            "summarized_revision": 2,
            "latest_snapshot_id": "snapshot-1",
        }
        self.other_cluster = {
            **self.cluster,
            "id": "cluster-2",
            "status": "dormant",
            "last_seen_ts": 1_800_000,
            "member_count": 1,
            "membership_revision": 1,
            "summarized_revision": 0,
            "latest_snapshot_id": None,
            "strong_indicators": ["organization:papercut"],
        }
        self.review_cluster = {
            **self.cluster,
            "id": "cluster-review",
            "status": "needs_review",
            "first_seen_ts": 1_700_000,
            "last_seen_ts": 1_700_000,
            "member_count": 1,
            "membership_revision": 1,
            "summarized_revision": 0,
            "latest_snapshot_id": None,
            "centroid": [0.999, 0.01],
            "vector_sum": [0.999, 0.01],
        }
        self.memberships = [
            {
                "profile_id": "healthcare_europe",
                "article_id": "article-1",
                "cluster_id": "cluster-1",
                "article_ts": 1_100_000,
                "assigned_at": 1_200_000,
                "assignment_action": "created",
                "similarity": None,
                "evidence": {
                    "title": "Tillgänglig artikel",
                    "source": "Källa A",
                    "url": "https://example.invalid/article-1",
                    "published_ts": 1_100_000,
                },
            },
            {
                "profile_id": "healthcare_europe",
                "article_id": "article-expired",
                "cluster_id": "cluster-1",
                "article_ts": 1_300_000,
                "assigned_at": 1_400_000,
                "assignment_action": "matched",
                "similarity": 0.91,
                "evidence": {
                    "title": "Bortgallrad artikel",
                    "source": "Källa B",
                    "url": "https://example.invalid/article-expired",
                    "published_ts": 1_300_000,
                },
            },
        ]
        self.other_membership = {
            "profile_id": "healthcare_europe",
            "article_id": "article-2",
            "cluster_id": "cluster-2",
            "article_ts": 1_500_000,
            "assigned_at": 1_500_100,
            "assignment_action": "created",
            "evidence": {"title": "Papercut incident", "source": "Källa D"},
        }
        self.review_membership = {
            "profile_id": "healthcare_europe",
            "article_id": "article-review",
            "cluster_id": "cluster-review",
            "article_ts": 1_700_000,
            "assigned_at": 1_700_100,
            "assignment_action": "needs_review",
            "assignment_reason": "ambiguous_candidates",
            "similarity": 0.91,
            "second_similarity": 0.90,
            "best_candidate_cluster_id": "cluster-1",
            "second_candidate_cluster_id": "cluster-2",
            "evidence": {
                "title": "Ambiguous healthcare incident",
                "source": "Källa C",
                "url": "https://example.invalid/review",
                "published_ts": 1_700_000,
            },
        }
        self.review_resolutions = []
        self.applied_reconciliations = []
        self.applied_membership_edits = []
        self.extra_clusters = []
        self.extra_memberships = []

    def list_threat_landscape_reports(self, profile_id, *, limit=100):
        self.list_calls.append((profile_id, limit))
        return [row for row in self.reports if row["profile_id"] == profile_id][
            :limit
        ]

    def get_threat_landscape_report(self, report_id):
        return next((row for row in self.reports if row["id"] == report_id), None)

    def get_cluster_snapshot(self, snapshot_id):
        return dict(self.snapshot) if snapshot_id == self.snapshot["id"] else None

    def get_threat_cluster(self, cluster_id):
        rows = (self.cluster, self.other_cluster, self.review_cluster, *self.extra_clusters)
        return next((dict(row) for row in rows if cluster_id == row["id"]), None)

    def list_threat_clusters(
        self,
        profile_id,
        *,
        statuses=None,
        limit=10000,
        offset=0,
        include_vectors=True,
        **filters,
    ):
        self.cluster_list_calls.append(
            {
                "profile_id": profile_id,
                "statuses": statuses,
                "limit": limit,
                "offset": offset,
                "include_vectors": include_vectors,
                **filters,
            }
        )
        rows = [self.cluster, self.other_cluster, self.review_cluster, *self.extra_clusters]
        rows = [row for row in rows if row["profile_id"] == profile_id]
        if statuses:
            rows = [row for row in rows if row["status"] in statuses]
        min_member_count = filters.get("min_member_count")
        if min_member_count is not None:
            rows = [
                row for row in rows
                if int(row.get("member_count") or 0) >= int(min_member_count)
            ]
        rows.sort(key=lambda row: (-row["last_seen_ts"], row["id"]))
        result = [dict(row) for row in rows[offset : offset + limit]]
        if not include_vectors:
            for row in result:
                row.pop("centroid", None)
                row.pop("vector_sum", None)
        return result

    def list_cluster_memberships(self, cluster_id, *, limit=10000):
        if any(row["id"] == cluster_id for row in self.extra_clusters):
            return [
                dict(row) for row in self.extra_memberships
                if row["cluster_id"] == cluster_id
            ][:limit]
        if cluster_id == self.cluster["id"]:
            return [dict(row) for row in self.memberships[:limit]]
        if cluster_id == self.review_cluster["id"]:
            return [dict(self.review_membership)][:limit]
        if cluster_id == self.other_cluster["id"]:
            return [dict(self.other_membership)][:limit]
        return []

    def resolve_threat_cluster_review(
        self, cluster_doc, *, expected_membership_revision
    ):
        if (
            self.review_cluster["status"] != "needs_review"
            or self.review_cluster["membership_revision"]
            != expected_membership_revision
        ):
            return False
        self.review_cluster = dict(cluster_doc)
        self.review_resolutions.append(dict(cluster_doc))
        return True

    def get_cluster_membership(self, profile_id, article_id):
        rows = [*self.memberships, self.review_membership]
        return next(
            (
                dict(row) for row in rows
                if row["profile_id"] == profile_id and row["article_id"] == article_id
            ),
            None,
        )

    def list_article_cluster_memberships(self, profile_id, article_id):
        rows = [*self.memberships, self.other_membership, self.review_membership,
                *self.extra_memberships]
        return [dict(row) for row in rows
                if row["profile_id"] == profile_id and row["article_id"] == article_id]

    def apply_cluster_membership_edit(self, operation):
        self.applied_membership_edits.append(dict(operation))
        return True

    def apply_cluster_reconciliation(self, operation):
        self.applied_reconciliations.append(dict(operation))
        return True

    def list_cluster_snapshots(self, profile_id, *, cluster_id=None, limit=1000):
        if profile_id != self.snapshot["profile_id"]:
            return []
        if cluster_id is not None and cluster_id != self.snapshot["cluster_id"]:
            return []
        return [dict(self.snapshot)][:limit]

    def get_articles_by_ids(self, article_ids):
        return (
            [{
                "id": "article-1",
                "title": "Tillgänglig artikel",
                "similarity_embedding_vector": [1.0, 0.0],
                "similarity_embedding_model": "embedding-model",
                "similarity_embedding_instruction": "event",
            }]
            if "article-1" in article_ids
            else []
        )

    def get_all_tags(self):
        return [
            {"id": 1, "name": "Attack Analysis", "category": "DOMAIN_ENTITY"},
            {"id": 2, "name": "Data Breach", "category": "DOMAIN_ENTITY"},
            {"id": 3, "name": "Espionage", "category": "THREAT_ACTOR"},
            {"id": 4, "name": "Healthcare", "category": "SECTOR"},
        ]

    def list_long_term_runs(self, profile_id, *, limit=100):
        if profile_id != "healthcare_europe":
            return []
        return [
            {
                "id": "run-1",
                "profile_id": profile_id,
                "run_type": "long_term_pipeline",
                "mode": "shadow",
                "stage": "done",
                "status": "done",
                "started_at": 2_100_000,
                "finished_at": 2_100_100,
            },
            {
                "id": "run-2",
                "profile_id": profile_id,
                "run_type": "long_term_map",
                "mode": "shadow",
                "stage": "map",
                "status": "failed",
                "started_at": 2_000_000,
                "finished_at": 2_000_100,
                "error_type": "RuntimeError",
                "error": "model unavailable",
            },
        ][:limit]


class LongTermUiTests(unittest.TestCase):
    def setUp(self):
        self.original_cfg = webapp_viewer.APP_CFG
        self.original_store = webapp_viewer.APP_STORE
        self.store = _LongTermStore()
        webapp_viewer.APP_STORE = self.store
        webapp_viewer.APP_CFG = {
            "long_term_analysis": {
                "enabled": False,
                "mode": "shadow",
                "ui": {"enabled": True},
                "profiles": {
                    "healthcare_europe": {"enabled": True},
                    "public_sector": {"enabled": False},
                },
            }
        }
        webapp_viewer.app.config.update(TESTING=True)
        self.client = webapp_viewer.app.test_client()

    def tearDown(self):
        webapp_viewer.APP_CFG = self.original_cfg
        webapp_viewer.APP_STORE = self.original_store

    def test_feature_flag_hides_pages_and_api_without_reading_store(self):
        webapp_viewer.APP_CFG["long_term_analysis"]["ui"]["enabled"] = False

        for path in (
            "/long-term",
            "/long-term/clusters",
            "/long-term/report/report-2",
            "/api/v1/long-term/reports",
            "/api/v1/long-term/reports/report-2",
            "/api/v1/long-term/runs",
            "/api/v1/long-term/snapshots/snapshot-1",
            "/api/v1/long-term/clusters",
            "/api/v1/long-term/clusters/cluster-1",
            "/api/v1/long-term/reviews/cluster-review",
            "/long-term/snapshot/snapshot-1",
            "/long-term/review/cluster-review",
            "/long-term/cluster/cluster-1",
        ):
            with self.subTest(path=path):
                self.assertEqual(404, self.client.get(path).status_code)
        self.assertEqual([], self.store.list_calls)
        self.assertEqual([], self.store.cluster_list_calls)

    def test_list_api_has_versioned_compact_contract_and_profile_metadata(self):
        response = self.client.get(
            "/api/v1/long-term/reports?profile=healthcare_europe&limit=1"
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(1, payload["schema_version"])
        self.assertEqual("healthcare_europe", payload["active_profile"])
        self.assertEqual(
            ["healthcare_europe", "public_sector"], payload["available_profiles"]
        )
        self.assertEqual([("healthcare_europe", 1)], self.store.list_calls)
        self.assertEqual(3, payload["items"][0]["input_snapshot_count"])
        self.assertEqual(1, payload["items"][0]["forecast_count"])
        self.assertNotIn("metrics", payload["items"][0])
        self.assertNotIn("markdown", payload["items"][0])

    def test_unknown_profile_is_not_queried(self):
        response = self.client.get(
            "/api/v1/long-term/reports?profile=unknown_profile"
        )

        self.assertEqual(404, response.status_code)
        self.assertEqual([], self.store.list_calls)

    def test_detail_api_returns_authoritative_report_and_404_for_missing(self):
        response = self.client.get("/api/v1/long-term/reports/report-2")

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(1, payload["schema_version"])
        self.assertEqual("report-2", payload["item"]["id"])
        self.assertEqual({"windows": [{"days": 90}]}, payload["item"]["metrics"])
        self.assertEqual(
            404,
            self.client.get("/api/v1/long-term/reports/missing").status_code,
        )

    def test_list_page_renders_reports_warnings_and_navigation(self):
        response = self.client.get("/long-term?profile=healthcare_europe")

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn("Långtidsanalys av hotlandskapet", html)
        self.assertIn(">SHADOW</span>", html)
        self.assertIn("Tre oberoende händelser visar ett mönster.", html)
        self.assertEqual(1, html.count(">Senaste</span>"))
        self.assertIn("source_concentration", html)
        self.assertIn("/long-term/report/report-2", html)
        self.assertIn(">Långtidsanalys</a>", html)
        self.assertIn("Senaste pipelinekörningar", html)
        self.assertIn("long_term_pipeline", html)
        self.assertIn("model unavailable", html)

    def test_default_pages_show_each_profile_without_dropdown(self):
        profiles = webapp_viewer.APP_CFG["long_term_analysis"]["profiles"]
        profiles["public_sector"]["enabled"] = True
        profiles["all_profiles"] = {
            "enabled": True,
            "display_name": "Samlad analys",
            "aggregate_profiles": ["healthcare_europe", "public_sector"],
        }
        self.store.reports.insert(0, {
            **_report("report-aggregate"),
            "profile_id": "all_profiles",
            "analysis": {"executive_summary": "Samlad lägesbild."},
        })
        self.store.reports.append({
            **_report("report-public"),
            "profile_id": "public_sector",
            "analysis": {"executive_summary": "Offentlig sektor."},
        })
        self.store.extra_clusters.append({
            **self.store.other_cluster,
            "id": "cluster-public",
            "profile_id": "public_sector",
            "latest_snapshot_id": None,
        })

        reports = self.client.get("/long-term")
        clusters = self.client.get("/long-term/clusters")

        self.assertEqual(200, reports.status_code)
        self.assertEqual(200, clusters.status_code)
        report_html = reports.get_data(as_text=True)
        cluster_html = clusters.get_data(as_text=True)
        self.assertIn("Samlad lägesbild.", report_html)
        self.assertIn("Offentlig sektor.", report_html)
        self.assertIn("Tre oberoende händelser", report_html)
        self.assertLess(report_html.index("Samlad analys"), report_html.index("healthcare_europe"))
        self.assertNotIn('name="profile"', report_html)
        self.assertIn("cluster-public", cluster_html)
        self.assertIn("healthcare_europe", cluster_html)
        self.assertNotIn('<select class="form-select form-select-sm" id="profile"', cluster_html)

    def test_runs_api_exposes_compact_operational_status(self):
        response = self.client.get(
            "/api/v1/long-term/runs?profile=healthcare_europe&limit=1"
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(1, payload["schema_version"])
        self.assertEqual("run-1", payload["items"][0]["id"])
        self.assertEqual("shadow", payload["items"][0]["mode"])
        self.assertNotIn("results", payload["items"][0])

    def test_cluster_list_api_is_paginated_filtered_and_omits_vectors(self):
        response = self.client.get(
            "/api/v1/long-term/clusters"
            "?profile=healthcare_europe&status=all&page=1&per_page=1"
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(1, payload["schema_version"])
        self.assertEqual("healthcare_europe", payload["active_profile"])
        self.assertEqual("all", payload["active_status"])
        self.assertEqual(1, payload["page"])
        self.assertTrue(payload["has_next"])
        self.assertEqual(1, len(payload["items"]))
        self.assertEqual("cluster-1", payload["items"][0]["id"])
        self.assertEqual(
            "CVE-2026-65974 – ERPNext Server-Side Template Injection",
            payload["items"][0]["label"],
        )
        self.assertNotIn("centroid", payload["items"][0])
        self.assertEqual(
            {
                "profile_id": "healthcare_europe",
                "statuses": None,
                "limit": 2,
                "offset": 0,
                "include_vectors": False,
                "min_member_count": 1,
            },
            self.store.cluster_list_calls[-1],
        )

    def test_cluster_list_defaults_to_active_profile_after_rebuild(self):
        profiles = webapp_viewer.APP_CFG["long_term_analysis"]["profiles"]
        profiles["healthcare_europe"]["clustering_enabled"] = False
        profiles["healthcare_europe_tagged_v2"] = {
            "enabled": True,
            "clustering_enabled": True,
            "rebuild_source_profile_id": "healthcare_europe",
        }
        self.store.cluster["profile_id"] = "healthcare_europe_tagged_v2"

        response = self.client.get("/api/v1/long-term/clusters")
        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual("healthcare_europe_tagged_v2", payload["active_profile"])
        self.assertEqual("cluster-1", payload["items"][0]["id"])
        self.assertEqual(
            "healthcare_europe_tagged_v2", payload["available_profiles"][0]
        )
        self.assertEqual(
            "healthcare_europe_tagged_v2",
            self.store.cluster_list_calls[-1]["profile_id"],
        )

        page = self.client.get("/long-term/clusters")
        self.assertEqual(200, page.status_code)
        self.assertIn("cluster-1", page.get_data(as_text=True))

        archived_page = self.client.get(
            "/long-term/clusters?profile=healthcare_europe"
        )
        self.assertEqual(200, archived_page.status_code)
        self.assertIn("Den här profilen är arkiverad", archived_page.get_data(as_text=True))
        self.assertIn(
            "profile=healthcare_europe_tagged_v2",
            archived_page.get_data(as_text=True),
        )

    def test_cluster_list_page_links_readable_titles_and_next_page(self):
        response = self.client.get(
            "/long-term/clusters?profile=healthcare_europe&per_page=1"
        )

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn("Händelsekluster", html)
        self.assertIn(
            "CVE-2026-65974 – ERPNext Server-Side Template Injection", html
        )
        self.assertIn("/long-term/cluster/cluster-1", html)
        self.assertIn("/long-term/snapshot/snapshot-1", html)
        self.assertIn("Öppna senaste snapshot", html)
        self.assertNotIn(">cluster-1</a>", html)
        self.assertIn("Nästa →", html)
        self.assertIn("per_page=1", html)

    def test_cluster_list_can_only_show_clusters_with_snapshots(self):
        response = self.client.get(
            "/api/v1/long-term/clusters"
            "?profile=healthcare_europe&has_snapshot=1&per_page=1"
        )
        page = self.client.get(
            "/long-term/clusters"
            "?profile=healthcare_europe&has_snapshot=1&per_page=1"
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload["has_snapshot"])
        self.assertFalse(payload["has_next"])
        self.assertEqual(["cluster-1"], [row["id"] for row in payload["items"]])
        self.assertEqual(200, page.status_code)
        html = page.get_data(as_text=True)
        self.assertIn('id="has_snapshot"', html)
        self.assertIn("checked", html)
        self.assertNotIn("cluster-2", html)

    def test_cluster_list_snapshot_filter_is_preserved_across_pages(self):
        self.store.other_cluster["latest_snapshot_id"] = "snapshot-1"

        response = self.client.get(
            "/long-term/clusters"
            "?profile=healthcare_europe&has_snapshot=1&per_page=1"
        )
        next_page = self.client.get(
            "/api/v1/long-term/clusters"
            "?profile=healthcare_europe&has_snapshot=1&per_page=1&page=2"
        )

        self.assertEqual(200, response.status_code)
        self.assertIn("has_snapshot=1", response.get_data(as_text=True))
        self.assertEqual(["cluster-2"], [
            row["id"] for row in next_page.get_json()["items"]
        ])

    def test_cluster_list_tag_filter_requires_every_selected_tag(self):
        selected_tags = [
            "DOMAIN_ENTITY:attack analysis",
            "DOMAIN_ENTITY:data breach",
            "THREAT_ACTOR:espionage",
        ]
        self.store.cluster["tag_scope"] = "|".join([
            *selected_tags,
            "SECTOR:healthcare",
        ])
        self.store.other_cluster["tag_scope"] = "|".join(selected_tags[:2])
        query = {
            "profile": "healthcare_europe",
            "tag": selected_tags,
        }

        response = self.client.get(
            "/api/v1/long-term/clusters", query_string=query
        )
        page = self.client.get("/long-term/clusters", query_string=query)

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(selected_tags, payload["selected_tags"])
        self.assertEqual(["cluster-1"], [row["id"] for row in payload["items"]])
        self.assertEqual(200, page.status_code)
        html = page.get_data(as_text=True)
        self.assertIn("Ett kluster måste innehålla samtliga valda taggar", html)
        self.assertIn("Attack Analysis", html)
        self.assertEqual(3, html.count("checked"))
        self.assertNotIn("cluster-2", html)

    def test_cluster_list_keeps_same_label_clusters_separate(self):
        self.store.cluster["tag_scope"] = "VULNERABILITY"
        self.store.other_cluster["latest_snapshot_id"] = "snapshot-1"
        self.store.other_cluster["tag_scope"] = "MALWARE"

        response = self.client.get("/long-term/clusters?profile=healthcare_europe")

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        label = "CVE-2026-65974 – ERPNext Server-Side Template Injection"
        self.assertEqual(2, html.count(label))
        self.assertIn("VULNERABILITY", html)
        self.assertIn("MALWARE", html)
        self.assertIn("/long-term/cluster/cluster-1", html)
        self.assertIn("/long-term/cluster/cluster-2", html)

    def test_cluster_detail_shows_clusters_sharing_an_article(self):
        self.store.cluster["tag_scope"] = "VULNERABILITY"
        self.store.other_cluster["latest_snapshot_id"] = "snapshot-1"
        self.store.other_cluster["tag_scope"] = "MALWARE"
        self.store.other_membership["article_id"] = "article-1"

        response = self.client.get("/long-term/cluster/cluster-1")

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn("Andra kluster med gemensamma artiklar", html)
        self.assertIn("MALWARE", html)
        self.assertIn("/long-term/cluster/cluster-2", html)

    def test_article_page_lists_all_cluster_memberships(self):
        self.store.other_membership["article_id"] = "article-1"
        self.store.get_article = lambda article_id: {
            "id": article_id, "title": "Tillgänglig artikel", "text": "Text"
        }
        self.store.get_article_tags = lambda article_id: []

        response = self.client.get("/article/article-1")

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn("Artikeln kan ingå i flera kluster", html)
        self.assertIn("/long-term/cluster/cluster-1", html)
        self.assertIn("/long-term/cluster/cluster-2", html)

    def test_cluster_list_status_filter_and_validation(self):
        response = self.client.get(
            "/api/v1/long-term/clusters?profile=healthcare_europe&status=dormant"
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(["cluster-2"], [row["id"] for row in payload["items"]])
        self.assertEqual(["dormant"], self.store.cluster_list_calls[-1]["statuses"])
        self.assertNotIn("needs_review", payload["available_statuses"])
        self.assertEqual(
            400,
            self.client.get("/long-term/clusters?status=invalid").status_code,
        )
        self.assertEqual(
            400,
            self.client.get("/long-term/clusters?status=needs_review").status_code,
        )

    def test_cluster_list_filters_by_minimum_member_count(self):
        response = self.client.get(
            "/api/v1/long-term/clusters?profile=healthcare_europe&min_members=2"
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(2, payload["min_members"])
        self.assertEqual(["cluster-1"], [row["id"] for row in payload["items"]])
        self.assertEqual(2, self.store.cluster_list_calls[-1]["min_member_count"])

    def test_review_page_compares_persisted_candidates_without_vectors(self):
        api_response = self.client.get(
            "/api/v1/long-term/reviews/cluster-review"
        )
        page_response = self.client.get("/long-term/review/cluster-review")

        self.assertEqual(200, api_response.status_code)
        payload = api_response.get_json()
        self.assertEqual("persisted", payload["candidate_source"])
        self.assertEqual(
            ["cluster-1", "cluster-2"],
            [row["id"] for row in payload["candidates"]],
        )
        self.assertNotIn("centroid", payload["item"])
        self.assertEqual(200, page_response.status_code)
        html = page_response.get_data(as_text=True)
        self.assertIn("Ambiguous healthcare incident", html)
        self.assertIn("ERPNext Server-Side Template Injection", html)
        self.assertIn("Papercut incident", html)
        self.assertIn("Granskningsåtgärder är avstängda", html)

    def test_legacy_review_candidates_are_reconstructed(self):
        self.store.review_membership.pop("best_candidate_cluster_id")
        self.store.review_membership.pop("second_candidate_cluster_id")

        response = self.client.get(
            "/api/v1/long-term/reviews/cluster-review"
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual("reconstructed", payload["candidate_source"])
        self.assertEqual(2, len(payload["candidates"]))

    def test_review_keep_separate_action_is_audited(self):
        webapp_viewer.APP_CFG["long_term_analysis"]["enabled"] = True
        webapp_viewer.APP_CFG["long_term_analysis"]["ui"][
            "review_actions_enabled"
        ] = True

        response = self.client.post(
            "/api/v1/long-term/clusters/cluster-review/review",
            json={
                "decision": "keep_separate",
                "reviewer": "Analyst One",
                "comment": "Different victim and incident window.",
                "expected_membership_revision": 1,
            },
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual("active", response.get_json()["item"]["status"])
        resolution = self.store.review_resolutions[0]
        self.assertEqual("keep_separate", resolution["review_decision"])
        self.assertEqual("Analyst One", resolution["reviewed_by"])
        self.assertEqual(
            "Different victim and incident window.", resolution["review_comment"]
        )

    def test_review_merge_action_uses_selected_candidate_and_audit(self):
        webapp_viewer.APP_CFG["long_term_analysis"]["enabled"] = True
        webapp_viewer.APP_CFG["long_term_analysis"]["ui"][
            "review_actions_enabled"
        ] = True

        response = self.client.post(
            "/api/v1/long-term/clusters/cluster-review/review",
            json={
                "decision": "merge",
                "target_cluster_id": "cluster-1",
                "reviewer": "Analyst One",
                "comment": "Same concrete incident.",
                "expected_membership_revision": 1,
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()["item"]
        self.assertEqual("cluster-1", payload["target_cluster_id"])
        operation = self.store.applied_reconciliations[0]
        self.assertEqual("cluster-1", operation["primary_cluster_id"])
        self.assertEqual(
            "Analyst One", operation["review_resolution"]["reviewed_by"]
        )

    def test_review_writes_require_separate_feature_flag(self):
        response = self.client.post(
            "/api/v1/long-term/clusters/cluster-review/review",
            json={
                "decision": "exclude",
                "reviewer": "Analyst One",
                "expected_membership_revision": 1,
            },
        )

        self.assertEqual(403, response.status_code)
        self.assertEqual([], self.store.review_resolutions)

    def test_detail_page_distinguishes_hypotheses_and_shows_invalidation(self):
        response = self.client.get("/long-term/report/report-2")

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn("Prognoshypoteser", html)
        self.assertIn("Inte observerade fakta", html)
        self.assertIn("Liknande aktivitet kan fortsätta.", html)
        self.assertIn("Inga nya incidenter observeras", html)
        self.assertIn("snapshot-1", html)
        self.assertIn("cluster-1", html)
        self.assertIn("source_concentration", html)
        self.assertIn("/long-term/report/report-1", html)
        self.assertIn("/long-term/snapshot/snapshot-1", html)
        self.assertIn("/long-term/cluster/cluster-1", html)
        self.assertIn(
            "CVE-2026-65974 – ERPNext Server-Side Template Injection", html
        )
        self.assertNotIn(">cluster-1</a>", html)
        self.assertNotIn(">snapshot-1</a>", html)

    def test_snapshot_api_and_page_link_to_cluster_and_article_evidence(self):
        api_response = self.client.get(
            "/api/v1/long-term/snapshots/snapshot-1"
        )
        page_response = self.client.get("/long-term/snapshot/snapshot-1")

        self.assertEqual(200, api_response.status_code)
        self.assertEqual("snapshot-1", api_response.get_json()["item"]["id"])
        self.assertEqual(200, page_response.status_code)
        html = page_response.get_data(as_text=True)
        self.assertIn(
            "CVE-2026-65974 – ERPNext Server-Side Template Injection", html
        )
        self.assertIn("CVE-2026-65974 rapporterades.", html)
        self.assertIn("Omfattningen är ännu okänd.", html)
        self.assertIn("/long-term/cluster/cluster-1", html)
        self.assertIn("/article/article-1", html)
        self.assertIn(">Tillgänglig artikel</a>", html)
        self.assertIn(">Bortgallrad artikel</a>", html)
        self.assertNotIn(">article-1</a>", html)

    def test_cluster_api_omits_vectors_and_preserves_retained_evidence(self):
        response = self.client.get("/api/v1/long-term/clusters/cluster-1")

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual("cluster-1", payload["item"]["id"])
        self.assertNotIn("centroid", payload["item"])
        self.assertNotIn("vector_sum", payload["item"])
        self.assertEqual(2, len(payload["memberships"]))
        self.assertTrue(payload["memberships"][0]["article_available"])
        self.assertFalse(payload["memberships"][1]["article_available"])
        self.assertEqual(
            "Bortgallrad artikel",
            payload["memberships"][1]["evidence"]["title"],
        )
        self.assertEqual("cluster-2", payload["comparison_candidates"][0]["id"])
        self.assertIn("similarity", payload["comparison_candidates"][0])

    def test_cluster_page_shows_comparison_and_merge_controls(self):
        webapp_viewer.APP_CFG["long_term_analysis"]["enabled"] = True
        webapp_viewer.APP_CFG["long_term_analysis"]["ui"][
            "review_actions_enabled"
        ] = True

        response = self.client.get("/long-term/cluster/cluster-1")

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn("Jämför med andra kluster", html)
        self.assertIn("Papercut incident", html)
        self.assertIn("Slå ihop kluster", html)
        self.assertIn('name="merge_cluster_ids" value="cluster-2"', html)
        self.assertIn('form="merge_cluster_form"', html)
        self.assertNotIn('name="target_cluster_id"', html)

    def test_selected_comparison_clusters_merge_into_open_cluster_together(self):
        webapp_viewer.APP_CFG["long_term_analysis"]["enabled"] = True
        webapp_viewer.APP_CFG["long_term_analysis"]["ui"][
            "review_actions_enabled"
        ] = True
        self.store.extra_clusters.append({
            **self.store.other_cluster,
            "id": "cluster-3",
            "centroid": [0.99, 0.01],
            "vector_sum": [0.99, 0.01],
        })
        self.store.extra_memberships.append({
            **self.store.other_membership,
            "article_id": "article-3",
            "cluster_id": "cluster-3",
            "evidence": {"title": "Another incident"},
        })

        page = self.client.get("/long-term/cluster/cluster-1")
        self.assertEqual(200, page.status_code)
        html = page.get_data(as_text=True)
        self.assertIn('name="merge_cluster_ids" value="cluster-2"', html)
        self.assertIn('name="merge_cluster_ids" value="cluster-3"', html)

        response = self.client.post(
            "/api/v1/long-term/clusters/cluster-1/review",
            data={
                "decision": "merge",
                "merge_cluster_ids": ["cluster-2", "cluster-3"],
                "reviewer": "Analyst One",
                "comment": "Same incident.",
                "expected_membership_revision": "2",
            },
        )
        self.assertEqual(302, response.status_code)
        self.assertTrue(response.location.endswith("/long-term/cluster/cluster-1"))
        self.assertEqual(1, len(self.store.applied_reconciliations))
        operation = self.store.applied_reconciliations[0]
        validate_cluster_merge_operation(operation)
        self.assertEqual("cluster-1", operation["primary_cluster_id"])
        self.assertEqual(
            ["cluster-1", "cluster-2", "cluster-3"], operation["source_cluster_ids"]
        )
        self.assertEqual("Analyst One", operation["manual_merge"]["reviewed_by"])
        self.assertEqual(
            {"cluster-2", "cluster-3"},
            {row["id"] for row in operation["superseded_clusters"]},
        )

    def test_multi_merge_rejects_unselected_or_duplicate_clusters(self):
        webapp_viewer.APP_CFG["long_term_analysis"]["enabled"] = True
        webapp_viewer.APP_CFG["long_term_analysis"]["ui"][
            "review_actions_enabled"
        ] = True
        for selected_ids in ([], ["cluster-review"], ["cluster-2", "cluster-2"]):
            with self.subTest(selected_ids=selected_ids):
                response = self.client.post(
                    "/api/v1/long-term/clusters/cluster-1/review",
                    json={
                        "decision": "merge",
                        "merge_cluster_ids": selected_ids,
                        "reviewer": "Analyst One",
                        "expected_membership_revision": 2,
                    },
                )
                self.assertEqual(400, response.status_code)
        self.assertEqual([], self.store.applied_reconciliations)

    def test_active_cluster_can_be_merged_with_comparison_candidate(self):
        webapp_viewer.APP_CFG["long_term_analysis"]["enabled"] = True
        webapp_viewer.APP_CFG["long_term_analysis"]["ui"][
            "review_actions_enabled"
        ] = True

        response = self.client.post(
            "/api/v1/long-term/clusters/cluster-1/review",
            json={
                "decision": "merge",
                "target_cluster_id": "cluster-2",
                "reviewer": "Analyst One",
                "comment": "Same concrete incident.",
                "expected_membership_revision": 2,
            },
        )

        self.assertEqual(200, response.status_code)
        operation = self.store.applied_reconciliations[0]
        self.assertEqual("cluster-2", operation["primary_cluster_id"])
        self.assertEqual("merge", operation["review_resolution"]["review_decision"])
        self.assertEqual("Analyst One", operation["review_resolution"]["reviewed_by"])

    def test_cluster_page_links_available_article_and_labels_expired_article(self):
        response = self.client.get("/long-term/cluster/cluster-1")

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn(
            "CVE-2026-65974 – ERPNext Server-Side Template Injection", html
        )
        self.assertIn("/long-term/snapshot/snapshot-1", html)
        self.assertIn("/article/article-1", html)
        self.assertIn("Bortgallrad artikel", html)
        self.assertIn("Artikel bortgallrad", html)
        self.assertNotIn("[1.0, 0.0]", html)

    def test_cluster_page_uses_article_title_when_snapshot_is_missing(self):
        list_response = self.client.get(
            "/api/v1/long-term/clusters?profile=healthcare_europe&status=dormant"
        )
        page_response = self.client.get("/long-term/cluster/cluster-2")

        self.assertEqual(200, list_response.status_code)
        self.assertEqual("Papercut incident", list_response.get_json()["items"][0]["label"])
        self.assertEqual(200, page_response.status_code)
        self.assertIn(">Papercut incident</h1>", page_response.get_data(as_text=True))

    def test_membership_split_action_is_audited(self):
        webapp_viewer.APP_CFG["long_term_analysis"]["enabled"] = True
        webapp_viewer.APP_CFG["long_term_analysis"]["ui"][
            "review_actions_enabled"
        ] = True

        response = self.client.post(
            "/api/v1/long-term/clusters/cluster-1/members",
            json={
                "action": "split_out",
                "article_id": "article-1",
                "reviewer": "Analyst One",
                "comment": "Unrelated incident.",
                "expected_membership_revision": 2,
            },
        )

        self.assertEqual(200, response.status_code)
        item = response.get_json()["item"]
        self.assertEqual("split", item["action"])
        operation = self.store.applied_membership_edits[0]
        self.assertEqual("Analyst One", operation["edited_by"])
        self.assertEqual("Unrelated incident.", operation["comment"])

    def test_membership_actions_require_write_flag(self):
        response = self.client.post(
            "/api/v1/long-term/clusters/cluster-1/members",
            json={
                "action": "split_out",
                "article_id": "article-1",
                "reviewer": "Analyst One",
                "expected_membership_revision": 2,
            },
        )

        self.assertEqual(403, response.status_code)
        self.assertEqual([], self.store.applied_membership_edits)

    def test_superseded_cluster_page_links_to_merged_cluster(self):
        self.store.cluster.update(
            {
                "status": "closed",
                "member_count": 0,
                "superseded_by_cluster_id": "cluster-primary",
                "reconciliation_id": "reconciliation-1",
                "reconciled_at": 2_000_000,
            }
        )
        self.store.memberships = []

        response = self.client.get("/long-term/cluster/cluster-1")

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn("historiska kluster har slagits samman", html)
        self.assertIn("/long-term/cluster/cluster-primary", html)
        self.assertIn("Äldre snapshots bevaras", html)

    def test_missing_snapshot_and_cluster_return_404(self):
        for path in (
            "/api/v1/long-term/snapshots/missing",
            "/api/v1/long-term/clusters/missing",
            "/long-term/snapshot/missing",
            "/long-term/cluster/missing",
        ):
            with self.subTest(path=path):
                self.assertEqual(404, self.client.get(path).status_code)

    def test_list_page_has_explicit_empty_state(self):
        self.store.reports = []

        response = self.client.get("/long-term")

        self.assertEqual(200, response.status_code)
        self.assertIn(
            "Inga publicerade rapporter finns ännu",
            response.get_data(as_text=True),
        )


if __name__ == "__main__":
    unittest.main()
