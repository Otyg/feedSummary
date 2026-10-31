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

import unittest

from web_viewer.long_term_cluster_graph import build_cluster_tag_network


def _cluster(cluster_id, tags, *, profile_id="profile-a", last_seen_ts=1):
    return {
        "id": cluster_id,
        "profile_id": profile_id,
        "status": "active",
        "tag_scope": "|".join(tags),
        "member_count": 1,
        "last_seen_ts": last_seen_ts,
    }


class LongTermClusterGraphTests(unittest.TestCase):
    def test_builds_reachable_one_tag_steps_in_both_directions(self):
        rows = [
            _cluster("base", ["SECTOR:Health", "REGION:EU", "ACTOR:Group"]),
            _cluster("up-1", ["SECTOR:health", "REGION:eu"]),
            _cluster("up-2", ["SECTOR:health"]),
            _cluster(
                "down-1",
                ["SECTOR:health", "REGION:eu", "ACTOR:group", "MALWARE:x"],
            ),
            _cluster(
                "down-2",
                [
                    "SECTOR:health",
                    "REGION:eu",
                    "ACTOR:group",
                    "MALWARE:x",
                    "CAMPAIGN:y",
                ],
            ),
            _cluster("unrelated", ["SECTOR:health", "MALWARE:x"]),
            _cluster("other-profile", ["SECTOR:health", "REGION:eu"], profile_id="profile-b"),
        ]

        graph = build_cluster_tag_network(
            rows,
            "base",
            up_levels=2,
            down_levels=2,
            max_nodes=20,
        )

        self.assertEqual(
            {"base": 0, "up-1": -1, "up-2": -2, "down-1": 1, "down-2": 2},
            {
                node["id"]: node["level"]
                for node in graph["nodes"]
            },
        )
        self.assertEqual([-2, -1, 0, 1, 2], graph["levels"])
        self.assertTrue(
            {
                ("up-2", "up-1", ("REGION:eu",)),
                ("up-1", "base", ("ACTOR:group",)),
                ("base", "down-1", ("MALWARE:x",)),
                ("down-1", "down-2", ("CAMPAIGN:y",)),
            }.issubset({
                (edge["source"], edge["target"], tuple(edge["added_tags"]))
                for edge in graph["edges"]
            }),
        )

    def test_uses_the_closest_existing_strict_subset_as_parent(self):
        rows = [
            _cluster("base", ["SECTOR:health", "REGION:eu", "ACTOR:group"]),
            _cluster("two-up", ["SECTOR:health"]),
            _cluster(
                "two-down",
                [
                    "SECTOR:health",
                    "REGION:eu",
                    "ACTOR:group",
                    "MALWARE:x",
                    "CAMPAIGN:y",
                ],
            ),
        ]

        graph = build_cluster_tag_network(
            rows,
            "base",
            up_levels=2,
            down_levels=2,
            max_nodes=20,
        )

        self.assertEqual(
            {"base": 0, "two-up": -1, "two-down": 1},
            {node["id"]: node["level"] for node in graph["nodes"]},
        )
        self.assertEqual(
            {
                ("two-up", "base", ("ACTOR:group", "REGION:eu")),
                ("base", "two-down", ("CAMPAIGN:y", "MALWARE:x")),
            },
            {
                (edge["source"], edge["target"], tuple(edge["added_tags"]))
                for edge in graph["edges"]
            },
        )

    def test_parent_relation_is_visible_from_both_endpoints(self):
        rows = [
            _cluster("parent", ["SECTOR:health"]),
            _cluster("child", ["SECTOR:health", "REGION:eu", "ACTOR:group"]),
        ]

        from_child = build_cluster_tag_network(
            rows,
            "child",
            up_levels=1,
            down_levels=0,
            max_nodes=20,
        )
        from_parent = build_cluster_tag_network(
            rows,
            "parent",
            up_levels=0,
            down_levels=1,
            max_nodes=20,
        )

        self.assertEqual(
            {("parent", "child")},
            {
                (edge["source"], edge["target"])
                for edge in from_child["edges"]
            },
        )
        self.assertEqual(
            {("parent", "child")},
            {
                (edge["source"], edge["target"])
                for edge in from_parent["edges"]
            },
        )

    def test_upward_level_uses_one_shared_tag_count(self):
        rows = [
            _cluster(
                "base",
                [
                    "SECTOR:health",
                    "REGION:eu",
                    "ACTOR:group",
                    "MALWARE:x",
                    "CAMPAIGN:y",
                ],
            ),
            _cluster(
                "parent-a",
                ["SECTOR:health", "REGION:eu", "ACTOR:group", "MALWARE:x"],
            ),
            _cluster(
                "parent-b",
                ["SECTOR:health", "REGION:eu", "ACTOR:group", "CAMPAIGN:y"],
            ),
            _cluster("ancestor", ["SECTOR:health", "REGION:eu", "ACTOR:group"]),
            _cluster("later", ["SECTOR:health", "REGION:eu"]),
        ]

        graph = build_cluster_tag_network(
            rows,
            "base",
            up_levels=2,
            down_levels=0,
            max_nodes=20,
        )

        self.assertEqual(
            {"base": 0, "parent-a": -1, "parent-b": -1, "ancestor": -2},
            {node["id"]: node["level"] for node in graph["nodes"]},
        )
        self.assertEqual(
            {4},
            {
                node["tag_count"]
                for node in graph["nodes"]
                if node["level"] == -1
            },
        )
        self.assertNotIn("later", {node["id"] for node in graph["nodes"]})
        edge_pairs = {
            (edge["source"], edge["target"])
            for edge in graph["edges"]
        }
        self.assertIn(("parent-a", "base"), edge_pairs)
        self.assertIn(("parent-b", "base"), edge_pairs)
        self.assertNotIn(("ancestor", "base"), edge_pairs)

    def test_marks_a_node_limited_network_as_truncated(self):
        rows = [
            _cluster("base", ["SECTOR:health", "REGION:eu"]),
            _cluster("up-a", ["SECTOR:health"], last_seen_ts=3),
            _cluster("up-b", ["REGION:eu"], last_seen_ts=2),
        ]

        graph = build_cluster_tag_network(
            rows,
            "base",
            up_levels=1,
            down_levels=0,
            max_nodes=2,
        )

        self.assertTrue(graph["truncated"])
        self.assertEqual(2, len(graph["nodes"]))

    def test_excluded_base_tag_is_removed_from_every_cluster_before_matching(self):
        rows = [
            _cluster("base", ["SECTOR:health", "REGION:eu", "ACTOR:group"]),
            _cluster("same-after-exclusion", ["SECTOR:health", "REGION:eu"]),
            _cluster("parent", ["SECTOR:health"]),
            _cluster(
                "child",
                [
                    "SECTOR:health",
                    "REGION:eu",
                    "ACTOR:group",
                    "MALWARE:x",
                ],
            ),
        ]

        graph = build_cluster_tag_network(
            rows,
            "base",
            up_levels=1,
            down_levels=1,
            max_nodes=20,
            excluded_tags=["actor:GROUP"],
        )

        self.assertEqual(["ACTOR:group"], graph["excluded_tags"])
        self.assertEqual(["REGION:eu", "SECTOR:health"], graph["base_tags"])
        self.assertEqual(2, next(node for node in graph["nodes"] if node["is_base"])["tag_count"])
        self.assertEqual(
            {"base": 0, "parent": -1, "child": 1},
            {node["id"]: node["level"] for node in graph["nodes"]},
        )
        self.assertTrue(
            all(
                "ACTOR:group" not in node["tags"]
                for node in graph["nodes"]
            )
        )


if __name__ == "__main__":
    unittest.main()
