"""Contract tests for the long-term TEF context endpoint."""

from __future__ import annotations

import os
import time
import unittest


_previous_auto_init = os.environ.get("FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT")
os.environ["FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT"] = "1"
try:
    from web_viewer import webapp_viewer, webapp_viewer_readonly
finally:
    if _previous_auto_init is None:
        os.environ.pop("FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT", None)
    else:
        os.environ["FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT"] = _previous_auto_init


class _Store:
    def __init__(self, now):
        ids = ["event-1", "event-2", "event-3"]
        self.reports = [{
            "id": "report-1",
            "profile_id": "healthcare_europe",
            "status": "published",
            "period_end_ts": now - 3600,
            "metrics": {
                "quality": {"input_truncated": False},
                "windows": [{
                    "days": 30,
                    "independent_event_count": 3,
                    "event_cluster_ids": ids,
                    "comparison": {"independent_event_count": 1},
                    "coverage": {"trend_eligible": True},
                }],
            },
        }]
        self.clusters = {
            item: {
                "id": item,
                "profile_id": "healthcare_europe",
                "status": "active",
                "last_seen_ts": now - 7200,
                "strong_indicators": ["cve:cve-2026-12345"] if item != "event-3" else
                                     ["cve:cve-2026-123456"],
                "latest_snapshot_id": f"snapshot-{item}",
            }
            for item in ids
        }
        self.snapshots = {
            "snapshot-event-1": {
                "id": "snapshot-event-1",
                "cluster_id": "event-1",
                "profile_id": "healthcare_europe",
                "created_at": now - 1800,
                "payload": {"summary": "CVE-2026-12345 was disclosed.",
                            "confidence": "high", "facts": []},
            }
        }
        self.reads = 0

    def list_threat_landscape_reports(self, profile_id, *, limit):
        self.reads += 1
        return self.reports[:limit]

    def list_threat_clusters(self, profile_id, *, statuses, min_last_seen_ts,
                             limit, offset, include_vectors):
        self.reads += 1
        rows = [row for row in self.clusters.values()
                if row["profile_id"] == profile_id and row["status"] in statuses
                and row["last_seen_ts"] >= min_last_seen_ts]
        rows.sort(key=lambda row: (-row["last_seen_ts"], row["id"]))
        return rows[offset:offset + limit]

    def get_cluster_snapshot(self, snapshot_id):
        self.reads += 1
        return self.snapshots.get(snapshot_id)

    def get_threat_cluster(self, cluster_id):
        self.reads += 1
        return self.clusters.get(cluster_id)


class AttackPressureTests(unittest.TestCase):
    def setUp(self):
        self.old_cfg = webapp_viewer.APP_CFG
        self.old_store = webapp_viewer.APP_STORE
        self.now = int(time.time())
        self.store = _Store(self.now)
        webapp_viewer.APP_STORE = self.store
        webapp_viewer.APP_CFG = {
            "long_term_analysis": {
                "ui": {"enabled": True},
                "profiles": {"healthcare_europe": {
                    "region": "Europe", "sector": "healthcare"
                }},
            }
        }
        webapp_viewer.app.config.update(TESTING=True)
        self.client = webapp_viewer.app.test_client()
        self.path = "/api/v1/long-term/attack-pressure?profile=healthcare_europe&cve=CVE-2026-12345"

    def tearDown(self):
        webapp_viewer.APP_CFG = self.old_cfg
        webapp_viewer.APP_STORE = self.old_store

    def test_assessment_and_exact_cve_link(self):
        response = self.client.get(self.path)
        self.assertEqual(200, response.status_code)
        body = response.get_json()
        self.assertEqual("assessed", body["status"])
        self.assertEqual("overlaps_cve_signal", body["region"]["level"])
        self.assertEqual("overlaps_cve_signal", body["sector"]["level"])
        self.assertEqual(["event-1", "event-2"], body["vulnerability"]["evidence_cluster_ids"])
        self.assertEqual(1.125, body["tef_adjustment"]["factor"])
        self.assertEqual("report-1", body["evidence"]["report_id"])
        self.assertEqual(1.1, body["region"]["factor"])
        self.assertEqual(1.1, body["sector"]["factor"])
        self.assertEqual(1.15, body["vulnerability"]["factor"])

    def test_report_cluster_keeps_cve_signal_when_cluster_is_no_longer_recent(self):
        self.store.clusters["event-1"]["last_seen_ts"] = self.now - 31 * 86400
        body = self.client.get(self.path).get_json()
        self.assertEqual("reported", body["vulnerability"]["level"])
        self.assertEqual(1.15, body["vulnerability"]["factor"])
        self.assertEqual(["event-1", "event-2"],
                 body["vulnerability"]["report_evidence_cluster_ids"])
        self.assertEqual(1.1, body["region"]["factor"])
        self.assertEqual(1.1, body["sector"]["factor"])

    def test_active_exploitation_snapshot_raises_cve_component(self):
        self.store.snapshots["snapshot-event-1"]["payload"]["summary"] = (
            "Attackers are exploiting CVE-2026-12345 in current campaigns. "
            "CVE-2026-123456 is also mentioned."
        )
        body = self.client.get(self.path).get_json()
        self.assertEqual("active_exploitation_reported", body["vulnerability"]["level"])
        self.assertEqual(1.4, body["vulnerability"]["factor"])
        self.assertEqual(["snapshot-event-1"], body["vulnerability"]["evidence_snapshot_ids"])
        self.assertEqual(1.25, body["tef_adjustment"]["factor"])
        self.assertEqual({"region": 0.25, "sector": 0.25, "cve": 0.5},
                         body["tef_adjustment"]["weights"])

    def test_negated_exploitation_does_not_raise_cve_component(self):
        self.store.snapshots["snapshot-event-1"]["payload"]["summary"] = (
            "CVE-2026-12345 is not exploited in the wild."
        )
        body = self.client.get(self.path).get_json()
        self.assertEqual("reported", body["vulnerability"]["level"])
        self.assertEqual(1.125, body["tef_adjustment"]["factor"])

    def test_low_confidence_snapshot_does_not_claim_active_exploitation(self):
        self.store.snapshots["snapshot-event-1"]["payload"].update({
            "summary": "Attackers are exploiting CVE-2026-12345.",
            "confidence": "low",
        })
        body = self.client.get(self.path).get_json()
        self.assertEqual("reported", body["vulnerability"]["level"])
        self.assertEqual(1.125, body["tef_adjustment"]["factor"])

    def test_sonicwall_chain_summary_and_explicit_timeline(self):
        self.store.clusters["event-1"]["strong_indicators"] = ["cve:cve-2026-83548"]
        payload = self.store.snapshots["snapshot-event-1"]["payload"]
        payload["summary"] = (
            "Threat actors are actively exploiting a chain of two zero-day "
            "vulnerabilities in SonicWall SMA1000 appliances. The chain consists "
            "of a pre-authentication SSRF (CVE-2026-83548) and a post-authentication "
            "injection (CVE-2026-83549)."
        )
        payload["timeline"] = [{
            "statement": "SonicWall confirms active exploitation of CVE-2026-83548 and CVE-2026-83549.",
            "evidence_article_ids": ["sonicwall-advisory"],
        }]
        body = self.client.get(self.path.replace("12345", "83548")).get_json()
        self.assertTrue(body["vulnerability"]["active_exploitation_reported"])
        self.assertEqual(["snapshot-event-1"], body["vulnerability"]["evidence_snapshot_ids"])
        self.assertIn("sonicwall-advisory", body["vulnerability"]["evidence_article_ids"])
        self.assertEqual(1.25, body["tef_adjustment"]["factor"])

    def test_chain_summary_can_link_immediately_following_cve_definition(self):
        payload = self.store.snapshots["snapshot-event-1"]["payload"]
        payload.update({
            "summary": "Threat actors are actively exploiting a vulnerability chain. "
                       "The chain consists of CVE-2026-12345 and CVE-2026-12346.",
            "timeline": [],
        })
        body = self.client.get(self.path).get_json()
        self.assertTrue(body["vulnerability"]["active_exploitation_reported"])

    def test_explicit_timeline_claim_works_without_summary_claim(self):
        payload = self.store.snapshots["snapshot-event-1"]["payload"]
        payload.update({
            "summary": "Two vulnerabilities affect the appliance.",
            "timeline": [{
                "statement": "The vendor confirms active exploitation of CVE-2026-12345.",
                "evidence_article_ids": ["vendor-advisory"],
            }],
        })
        body = self.client.get(self.path).get_json()
        self.assertTrue(body["vulnerability"]["active_exploitation_reported"])
        self.assertIn("vendor-advisory", body["vulnerability"]["evidence_article_ids"])

    def test_chain_definition_alone_does_not_claim_exploitation(self):
        self.store.snapshots["snapshot-event-1"]["payload"].update({
            "summary": "A vulnerability chain could be exploited. The chain consists "
                       "of CVE-2026-12345 and CVE-2026-12346.",
            "timeline": [],
        })
        body = self.client.get(self.path).get_json()
        self.assertFalse(body["vulnerability"]["active_exploitation_reported"])

    def test_large_report_and_newer_cluster_still_use_current_cve_snapshot(self):
        report = self.store.reports[0]
        report["period_end_ts"] = self.now - 10000
        window = report["metrics"]["windows"][0]
        window["independent_event_count"] = 2143
        window["comparison"]["independent_event_count"] = 0
        window["coverage"]["trend_eligible"] = False
        self.store.snapshots["snapshot-event-1"]["payload"]["summary"] = (
            "CVE-2026-12345 utnyttjas aktivt i pågående attacker."
        )
        body = self.client.get(self.path).get_json()
        self.assertEqual("assessed", body["status"])
        self.assertEqual(1.1, body["region"]["factor"])
        self.assertEqual(1.1, body["sector"]["factor"])
        self.assertEqual(1.4, body["vulnerability"]["factor"])
        self.assertEqual(1.25, body["tef_adjustment"]["factor"])

    def test_profile_uplift_requires_independent_non_cve_rise(self):
        window = self.store.reports[0]["metrics"]["windows"][0]
        window["event_cluster_ids"].append("event-4")
        window["independent_event_count"] = 4
        self.store.clusters["event-4"] = {
            "id": "event-4",
            "profile_id": "healthcare_europe",
            "status": "active",
            "last_seen_ts": self.now - 7200,
            "strong_indicators": [],
        }
        body = self.client.get(self.path).get_json()
        self.assertEqual("elevated", body["region"]["level"])
        self.assertEqual(1.1, body["region"]["factor"])
        self.assertEqual(1.1, body["sector"]["factor"])
        self.assertEqual(1.125, body["tef_adjustment"]["factor"])

    def test_unmatched_cve_and_ineligible_trend_are_neutral(self):
        self.store.reports[0]["metrics"]["windows"][0]["coverage"]["trend_eligible"] = False
        response = self.client.get(self.path.replace("12345", "99999"))
        body = response.get_json()
        self.assertEqual("unknown", body["region"]["level"])
        self.assertEqual("not_observed", body["vulnerability"]["level"])
        self.assertEqual(1.0, body["tef_adjustment"]["factor"])

    def test_stale_and_truncated_reports_fail_neutral(self):
        self.store.reports[0]["period_end_ts"] = self.now - 15 * 86400
        body = self.client.get(self.path).get_json()
        self.assertEqual("assessed", body["status"])
        self.assertEqual(1.05, body["tef_adjustment"]["factor"])
        self.assertEqual(1.0, body["region"]["factor"])
        self.store.reports[0]["period_end_ts"] = self.now - 3600
        self.store.reports[0]["metrics"]["quality"]["input_truncated"] = True
        body = self.client.get(self.path).get_json()
        self.assertEqual("truncated_report_inputs", body["evidence"]["reason"])

    def test_missing_cluster_indicator_data_does_not_fabricate_cve_signal(self):
        self.store.clusters["event-2"].pop("strong_indicators")
        body = self.client.get(self.path).get_json()
        self.assertEqual(1, body["vulnerability"]["independent_event_count"])
        self.assertEqual(["event-1"], body["vulnerability"]["evidence_cluster_ids"])

    def test_invalid_input_and_flag_do_not_read_store(self):
        self.assertEqual(400, self.client.get(
            self.path.replace("CVE-2026-12345", "CVE-xyz")
        ).status_code)
        self.assertEqual(404, self.client.get(
            self.path.replace("healthcare_europe", "unknown")
        ).status_code)
        webapp_viewer.APP_CFG["long_term_analysis"]["ui"]["enabled"] = False
        self.assertEqual(404, self.client.get(self.path).status_code)
        self.assertEqual(0, self.store.reads)

    def test_readonly_viewer_exposes_get_route(self):
        client = webapp_viewer_readonly.create_app().test_client()
        response = client.get(self.path)
        self.assertEqual(200, response.status_code)
        self.assertEqual("assessed", response.get_json()["status"])


if __name__ == "__main__":
    unittest.main()
