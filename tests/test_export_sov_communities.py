"""Tests for `graphify export sov --level communities`.

Each test writes a minimal graph.json of its own: three communities, two distinct
cross-community pairs with a known, different edge count each. No network or LLM
calls; the CLI case runs with --no-layout.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from graphify.exporters.sov import (
    community_document,
    community_pairs,
    community_title,
    communities_to_sov,
    dumps_sov,
    graph_to_sov,
    to_sov_communities,
)

PYTHON = sys.executable


def _code(node_id: str, label: str, *, community: int, source_file: str) -> dict:
    return {
        "id": node_id,
        "label": label,
        "file_type": "code",
        "_callable": True,
        "source_file": source_file,
        "source_location": "L1",
        "community": community,
    }


def _link(source: str, target: str, relation: str, *, context: str | None = None) -> dict:
    link = {"source": source, "target": target, "relation": relation,
            "confidence": "EXTRACTED", "source_file": "pkg/mod/a.py", "source_location": "L1"}
    if context is not None:
        link["context"] = context
    return link


def _graph() -> dict:
    """Three communities: 0 and 1 share a 2+-segment module path; 1 ranks a top
    member by degree; 2 is a lone third community. Cross pairs: (0, 1) with 3
    qualifying links, (0, 2) with 1."""
    nodes = [
        _code("n0_a", "f0()", community=0, source_file="pkg/mod/a.py"),
        _code("n0_b", "f1()", community=0, source_file="pkg/mod/b.py"),
        _code("n1_a", "alpha()", community=1, source_file="pkg/one.py"),
        _code("n1_b", "beta()", community=1, source_file="lib/two.py"),
        _code("n2_a", "gamma()", community=2, source_file="solo/x.py"),
    ]
    links = [
        _link("n0_a", "n1_a", "calls", context="c1"),
        _link("n0_a", "n1_a", "calls", context="c2"),
        _link("n1_b", "n0_a", "calls", context="c3"),
        _link("n0_b", "n2_a", "calls", context="c4"),
        _link("n1_a", "n1_b", "calls", context="c5"),  # same community: no pair, counts for degree
    ]
    return {"directed": False, "multigraph": False, "graph": {}, "nodes": nodes, "links": links}


def test_community_pairs_counts_known_cross_community_edges():
    data = _graph()
    _cards, card_of, community_of, _skipped = _top_cards(data)
    pairs = community_pairs(data, card_of, community_of)
    assert pairs == [((0, 1), 3), ((0, 2), 1)]


def _top_cards(data: dict):
    from graphify.exporters.sov import _cards as __cards  # the module's own private helper
    return __cards(list(data["nodes"]))


def test_community_title_module_path_case():
    data = _graph()
    members = [n for n in data["nodes"] if n["community"] == 0]
    links = data["links"]
    assert community_title(members, links=links) == "pkg/mod"


def test_community_title_top_member_name_case():
    data = _graph()
    members = [n for n in data["nodes"] if n["community"] == 1]
    links = data["links"]
    # n1_a has degree 3 (two links to n0_a's community plus one to n1_b), n1_b has
    # degree 2; no shared 2+-segment directory prefix ("pkg" vs "lib"), so the
    # higher-degree member's own name wins.
    assert community_title(members, links=links) == "alpha"


def test_communities_to_sov_cards_wires_and_document_refs(tmp_path):
    data = _graph()
    doc = communities_to_sov(data, communities_dir_name="graph.communities")
    cards = [c for c in doc["components"] if c["symbolId"] == "group"]
    assert len(cards) == 3
    assert [c["id"] for c in cards] == ["community-0", "community-1", "community-2"]

    wires_by_pair = {(w["a"], w["b"]): w["config"]["label"] for w in doc["wires"]}
    assert wires_by_pair == {
        ("community-0", "community-1"): "3",
        ("community-0", "community-2"): "1",
    }

    full = graph_to_sov(data)
    full_ids = {c["id"] for c in full["components"] if c["symbolId"] != "group"}

    union_ids: list[str] = []
    for cid in (0, 1, 2):
        per_doc = community_document(data, cid)
        union_ids += [c["id"] for c in per_doc["components"] if c["symbolId"] != "group"]
    assert sorted(union_ids) == sorted(full_ids)
    assert len(union_ids) == len(set(union_ids))


def test_to_sov_communities_writes_output_and_document_refs(tmp_path):
    data = _graph()
    out = tmp_path / "graph.sov"
    res = to_sov_communities(data, out, layout=False)
    assert res["communities"] == 3
    assert res["wires"] == 2
    full = graph_to_sov(data)
    assert res["members"] == sum(1 for c in full["components"] if c["symbolId"] != "group")
    assert Path(res["output"]).is_file()
    assert Path(res["communities_dir"]).is_dir()

    top = json.loads(out.read_text(encoding="utf-8"))
    out_dir = out.parent
    for card in top["components"]:
        ref = card["config"]["documentRef"]
        resolved = out_dir / ref
        assert resolved.is_file(), f"{ref} not written by to_sov_communities"


def _cli_graph(reverse: bool) -> dict:
    data = _graph()
    nodes = list(data["nodes"])
    links = list(data["links"])
    if reverse:
        nodes.reverse()
        links.reverse()
    return {"directed": False, "multigraph": False, "graph": {}, "nodes": nodes, "links": links}


def _run(args: list[str], cwd: Path, env: dict[str, str] | None = None):
    import subprocess
    return subprocess.run(
        [PYTHON, "-m", "graphify"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )


def _write_graph(tmp_path: Path, data: dict) -> Path:
    out = tmp_path / "graphify-out"
    out.mkdir()
    (out / "graph.json").write_text(json.dumps(data), encoding="utf-8")
    return out / "graph.json"


def test_cli_level_communities_is_order_independent(tmp_path):
    env = dict(os.environ)
    env.pop("SCHEMATICALLY_DIR", None)

    results = []
    for i, reverse in enumerate((False, True)):
        run_dir = tmp_path / f"run{i}"
        run_dir.mkdir()
        graph = _write_graph(run_dir, _cli_graph(reverse))
        r = _run(["export", "sov", str(graph), "--no-layout", "--level", "communities"], run_dir, env=env)
        assert r.returncode == 0, r.stderr
        out = graph.parent / "graph.sov"
        comm_dir = graph.parent / "graph.communities"
        assert out.is_file()
        assert comm_dir.is_dir()
        per_community = {p.name: p.read_bytes() for p in sorted(comm_dir.iterdir())}
        results.append((out.read_bytes(), per_community))

    (top0, docs0), (top1, docs1) = results
    assert top0 == top1
    assert sorted(docs0) == sorted(docs1)
    for name in docs0:
        assert docs0[name] == docs1[name], name
