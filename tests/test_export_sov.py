"""Tests for `graphify export sov`: graphify's code graph as a Schematically document.

Each test writes a minimal graph.json of its own. The layout test runs only when
SCHEMATICALLY_DIR names a Schematically checkout and node is on PATH.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from graphify.exporters.sov import LABEL_LENGTH, dumps_sov, edge_label, graph_to_sov

PYTHON = sys.executable


def _run(args: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, "-m", "graphify"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )


def _env_without_schematically() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("SCHEMATICALLY_DIR", None)
    return env


def _code(node_id, label, *, community, callable_=None, callable_class=None,
          source_file="pkg/a.py", location="L1"):
    node = {"id": node_id, "label": label, "file_type": "code", "source_file": source_file,
            "source_location": location, "community": community}
    if callable_ is not None:
        node["_callable"] = callable_
    if callable_class is not None:
        node["_callable_class"] = callable_class
    return node


def _link(source, target, relation, *, context=None, confidence="EXTRACTED", location="L1"):
    link = {"source": source, "target": target, "relation": relation,
            "confidence": confidence, "source_file": "pkg/a.py", "source_location": location}
    if context is not None:
        link["context"] = context
    return link


def _graph() -> dict:
    nodes = [
        _code("pkg_a", "a.py", community=0, source_file="pkg\\a.py"),
        _code("pkg_a_f", "f()", community=0, callable_=True, location="L2"),
        _code("pkg_a_g", "g()", community=0, callable_=True, location="L9"),
        _code("pkg_a_c", "C", community=1, callable_=True, callable_class=True, location="L20"),
        _code("pkg_a_c_m", ".m()", community=1, callable_=True, location="L21"),
        _code("pkg/a.py::h", "h()", community=1, callable_=True, location="L30"),
        {"id": "pkg_a_rationale_2", "label": "Why f exists", "file_type": "rationale",
         "source_file": "pkg/a.py", "source_location": "L2", "community": 2},
        {"id": "idea", "label": "An idea", "file_type": "concept", "source_file": "",
         "community": 0},
        _code("os_path", "os.path", community=1, source_file=""),
    ]
    links = [
        _link("pkg_a", "pkg_a_f", "contains"),
        _link("pkg_a_c", "pkg_a_c_m", "method"),
        _link("pkg_a_rationale_2", "pkg_a_f", "rationale_for"),
        _link("pkg_a_f", "pkg_a_g", "calls", context="call", location="L3"),
        _link("pkg_a_f", "pkg_a_g", "calls", context="call", location="L4"),
        _link("pkg_a_g", "pkg_a_g", "calls", context="call", location="L10"),
        _link("pkg_a_c_m", "pkg_a_c", "references", context="parameter_type",
              confidence="INFERRED", location="L21"),
        _link("pkg_a_f", "os_path", "imports", context="import"),
        _link("pkg/a.py::h", "pkg_a_f", "calls", context="call", confidence="AMBIGUOUS", location="L31"),
    ]
    return {"directed": False, "multigraph": False, "graph": {}, "nodes": nodes, "links": links}


def _cards(doc: dict) -> dict[str, dict]:
    return {c["id"]: c for c in doc["components"] if c["symbolId"] != "group"}


def _groups(doc: dict) -> list[dict]:
    return [c for c in doc["components"] if c["symbolId"] == "group"]


def _write_graph(tmp_path: Path, data: dict, labels: dict | None = None) -> Path:
    out = tmp_path / "graphify-out"
    out.mkdir()
    (out / "graph.json").write_text(json.dumps(data), encoding="utf-8")
    if labels is not None:
        (out / ".graphify_labels.json").write_text(json.dumps(labels), encoding="utf-8")
    return out / "graph.json"


def test_cards_are_functions_classes_and_modules():
    doc = graph_to_sov(_graph())
    cards = _cards(doc)
    assert {cid: c["symbolId"] for cid, c in cards.items()} == {
        "pkg_a": "ground",
        "pkg_a_f": "act",
        "pkg_a_g": "act",
        "pkg_a_c": "hold",
        "pkg_a_c_m": "act",
        "pkg-a-py--h": "act",
    }
    assert cards["pkg_a_f"]["config"] == {"label": "f", "subtitle": "function, pkg/a.py L2"}
    assert cards["pkg_a_c"]["config"]["subtitle"] == "class, pkg/a.py L20"
    assert cards["pkg_a"]["config"]["subtitle"] == "module, pkg\\a.py L1"
    assert cards["pkg_a_c_m"]["config"]["label"] == ".m"
    for card in cards.values():
        assert "x" not in card and "y" not in card
    assert [c["id"] for c in doc["components"][: len(cards)]] == sorted(cards)


def test_card_id_collision_names_both():
    data = _graph()
    data["nodes"].append(_code("pkg:a_f", "f2()", community=0, callable_=True))
    data["nodes"].append(_code("pkg/a_f", "f3()", community=0, callable_=True))
    with pytest.raises(ValueError, match="pkg:a_f.*pkg/a_f"):
        graph_to_sov(data)


def test_one_wire_per_edge_except_structure():
    doc = graph_to_sov(_graph())
    pairs = [(w["a"], w["b"]) for w in doc["wires"]]
    assert ("pkg_a", "pkg_a_f") not in pairs  # contains
    assert ("pkg_a_c", "pkg_a_c_m") not in pairs  # method
    assert not any(w["id"].endswith(("-contains", "-method", "-rationale_for")) for w in doc["wires"])
    data = _graph()
    data["links"].append(_link("pkg_a_f", "pkg_a_g", "contains"))
    data["links"].append(_link("pkg_a_f", "pkg_a_g", "method"))
    data["links"].append(_link("pkg_a_f", "pkg_a_g", "rationale_for"))
    assert len(graph_to_sov(data)["wires"]) == len(doc["wires"])
    assert pairs.count(("pkg_a_f", "pkg_a_g")) == 2
    assert ("pkg_a_g", "pkg_a_g") in pairs
    assert len(doc["wires"]) == 5
    ids = [w["id"] for w in doc["wires"]]
    assert len(set(ids)) == len(ids)
    assert "w-pkg_a_f-pkg_a_g-calls" in ids and "w-pkg_a_f-pkg_a_g-calls-2" in ids
    for wire in doc["wires"]:
        assert wire["aSide"] == "out" and wire["bSide"] == "in"
        assert wire["canvasId"] == "canvas:global"


def test_basis_marks_extracted_and_inferred():
    doc = graph_to_sov(_graph())
    basis = {(w["a"], w["b"]): w["config"]["basis"] for w in doc["wires"]}
    assert basis[("pkg_a_f", "pkg_a_g")] == "EXTRACTED"
    assert basis[("pkg_a_c_m", "pkg_a_c")] == "INFERRED"
    assert basis[("pkg-a-py--h", "pkg_a_f")] == "AMBIGUOUS"
    for wire in doc["wires"]:
        assert not any("colo" in key.lower() for key in wire["config"]), wire
        assert not any("colo" in key.lower() for key in wire)


def test_label_rule_and_length():
    assert LABEL_LENGTH == 28
    assert edge_label("calls", "call") == "calls"
    assert edge_label("references", "parameter_type") == "references: parameter type"
    assert edge_label("imports_from", "import") == "imports from"
    long = edge_label("r" * 40, None)
    assert len(long) == 28 and long.endswith("…") and long[:27] == "r" * 27
    assert edge_label("references", "parameter_type", 10) == "reference…"
    doc = graph_to_sov(_graph(), label_length=10)
    labels = {(w["a"], w["b"]): w["config"]["label"] for w in doc["wires"]}
    assert labels[("pkg_a_c_m", "pkg_a_c")] == "reference…"
    assert labels[("pkg_a_f", "pkg_a_g")] == "calls"
    full = graph_to_sov(_graph())
    assert {(w["a"], w["b"]): w["config"]["label"] for w in full["wires"]}[
        ("pkg_a_c_m", "pkg_a_c")] == "references: parameter type"


def test_one_group_per_community_with_cards(tmp_path):
    doc = graph_to_sov(_graph())
    groups = _groups(doc)
    assert [g["id"] for g in groups] == ["community-0", "community-1"]
    assert groups[0]["config"] == {"label": "Community 0", "members": ["pkg_a", "pkg_a_f", "pkg_a_g"]}
    assert groups[1]["config"]["members"] == ["pkg-a-py--h", "pkg_a_c", "pkg_a_c_m"]
    for group in groups:
        assert group["canvasId"] == "canvas:global"

    graph = _write_graph(tmp_path, _graph(), labels={"0": "Core", "1": "Shapes", "2": "Notes"})
    r = _run(["export", "sov", str(graph), "--no-layout"], tmp_path, env=_env_without_schematically())
    assert r.returncode == 0, r.stderr
    written = json.loads((graph.parent / "graph.sov").read_text(encoding="utf-8"))
    assert [g["config"]["label"] for g in _groups(written)] == ["Core", "Shapes"]


def test_direction_follows_graph_json_links():
    data = _graph()
    data["links"] = [_link("pkg_a_g", "pkg_a_f", "calls", context="call"),
                     _link("pkg_a_c", "pkg_a", "uses")]
    doc = graph_to_sov(data)
    assert [(w["a"], w["b"]) for w in doc["wires"]] == [("pkg_a_c", "pkg_a"), ("pkg_a_g", "pkg_a_f")]


def test_output_is_deterministic(tmp_path):
    first = dumps_sov(graph_to_sov(_graph()))
    assert dumps_sov(graph_to_sov(_graph())) == first
    assert first.endswith("}\n") and not first.endswith("\n\n")
    shuffled = _graph()
    shuffled["nodes"].reverse()
    shuffled["links"].reverse()
    assert dumps_sov(graph_to_sov(shuffled)) == first

    graph = _write_graph(tmp_path, _graph())
    env = _env_without_schematically()
    outs = []
    for name in ("one.sov", "two.sov"):
        r = _run(["export", "sov", str(graph), "--no-layout", "--output", name], tmp_path, env=env)
        assert r.returncode == 0, r.stderr
        outs.append((tmp_path / name).read_bytes())
    assert outs[0] == outs[1]


def test_cli_refuses_without_schematically(tmp_path):
    graph = _write_graph(tmp_path, _graph())
    r = _run(["export", "sov", str(graph)], tmp_path, env=_env_without_schematically())
    assert r.returncode == 1
    assert "--schematically" in r.stderr
    assert not (graph.parent / "graph.sov").exists()
    assert sorted(p.name for p in graph.parent.iterdir()) == ["graph.json"]


def test_cli_refuses_a_checkout_without_the_layout_script(tmp_path):
    graph = _write_graph(tmp_path, _graph())
    empty = tmp_path / "not-schematically"
    empty.mkdir()
    r = _run(["export", "sov", str(graph), "--schematically", str(empty)], tmp_path,
             env=_env_without_schematically())
    assert r.returncode == 1
    assert "layout_sov.mjs" in r.stderr
    assert not (graph.parent / "graph.sov").exists()


def test_cli_no_layout_writes_the_document(tmp_path):
    graph = _write_graph(tmp_path, _graph())
    r = _run(["export", "sov", str(graph), "--no-layout"], tmp_path, env=_env_without_schematically())
    assert r.returncode == 0, r.stderr
    assert "graph.sov written: 6 cards, 5 wires (3 extracted, 1 inferred), 2 groups" in r.stdout
    doc = json.loads((graph.parent / "graph.sov").read_text(encoding="utf-8"))
    assert doc["schema"] == "soveraeign.schematic/document@0.1"
    assert doc["id"] == "schematify" and doc["revision"] == 0
    assert doc["meta"] == {"title": "schematify: 6 cards"}
    assert doc["references"] == []
    assert doc == graph_to_sov(_graph())


@pytest.mark.skipif(
    not os.environ.get("SCHEMATICALLY_DIR") or shutil.which("node") is None,
    reason="needs SCHEMATICALLY_DIR naming a Schematically checkout and node on PATH",
)
def test_layout_and_validate(tmp_path):
    checkout = Path(os.environ["SCHEMATICALLY_DIR"])
    graph = _write_graph(tmp_path, _graph())
    r = _run(["export", "sov", str(graph)], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    out = graph.parent / "graph.sov"
    v = subprocess.run(["node", str(checkout / "scripts" / "validate_sov.mjs"), str(out)],
                       capture_output=True, text=True, encoding="utf-8")
    assert v.returncode == 0, v.stdout + v.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    cards = _cards(doc)
    assert len(cards) == 6
    assert all("x" in c and "y" in c for c in cards.values())
    assert sorted(p.name for p in graph.parent.iterdir()) == ["graph.json", "graph.sov"]
