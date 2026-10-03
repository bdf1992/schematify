"""sov — graphify's code graph as a Schematically document (.sov).

``graph_to_sov`` turns a loaded graph.json into a ``soveraeign.schematic/document@0.1``
document: one card per function, class and module, one wire per non-structural edge,
one reading-only group per community. It reads the raw links, never a networkx Graph,
because graph.json is written undirected and only its links keep each edge's true
source and target (see ``to_json`` in graphify/export.py).

``to_sov`` writes that document and lays it out by running a Schematically checkout's
``scripts/layout_sov.mjs`` as a subprocess. The checkout is reached only through the
path passed in; nothing from it is copied here. Nothing in this module calls an LLM or
a network service.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from graphify.paths import write_text_atomic

LABEL_LENGTH = 28

SCHEMA = "soveraeign.schematic/document@0.1"
CANVAS = "canvas:global"
# Relations that say where a definition sits rather than what it does; the group and
# the card subtitle already show them.
STRUCTURAL_RELATIONS = frozenset({"contains", "method", "rationale_for"})
SYMBOL_BY_KIND = {"function": "act", "class": "hold", "module": "ground"}
BASES = ("EXTRACTED", "INFERRED", "AMBIGUOUS")
ELLIPSIS = "…"

_ID_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def card_id(node_id: str) -> str:
    """The graphify id with every character outside [A-Za-z0-9_-] replaced by a hyphen."""
    return _ID_UNSAFE.sub("-", str(node_id))


def node_kind(node: dict) -> str | None:
    """'function', 'class' or 'module' for a node that gets a card, else None."""
    if node.get("file_type") != "code":
        return None
    source_file = node.get("source_file") or ""
    if not source_file:
        return None  # an imported name from outside the corpus
    if node.get("_callable_class") is True:
        return "class"
    if node.get("_callable") is True:
        return "function"
    label = str(node.get("label") or "")
    if label and source_file.replace("\\", "/").endswith(label):
        return "module"
    return None


def edge_label(relation: str, context: str | None, label_length: int = LABEL_LENGTH) -> str:
    """Relation with underscores as spaces, plus ': <context>' when the context adds something."""
    relation = str(relation or "")
    label = relation.replace("_", " ")
    if context and not relation.startswith(str(context)):
        label = f"{label}: {str(context).replace('_', ' ')}"
    if len(label) > label_length:
        label = label[: label_length - 1] + ELLIPSIS
    return label


def _card_label(label: str) -> str:
    return label[:-2] if label.endswith("()") else label


def _community(node: dict) -> int | None:
    raw = node.get("community")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _cards(nodes: list[dict]) -> tuple[list[dict], dict[str, str], dict[str, int | None], int]:
    """Cards, graphify id -> card id, card id -> community, and the skipped count."""
    cards: list[dict] = []
    card_of: dict[str, str] = {}
    owner: dict[str, str] = {}
    community_of: dict[str, int | None] = {}
    skipped = 0
    for node in nodes:
        kind = node_kind(node)
        if kind is None:
            skipped += 1
            continue
        node_id = str(node["id"])
        cid = card_id(node_id)
        if cid in owner:
            raise ValueError(
                f"node ids {owner[cid]!r} and {node_id!r} both map to card id {cid!r}"
            )
        owner[cid] = node_id
        card_of[node_id] = cid
        community_of[cid] = _community(node)
        location = str(node.get("source_location") or "")
        subtitle = f"{kind}, {node.get('source_file')} {location}".rstrip()
        cards.append({
            "id": cid,
            "symbolId": SYMBOL_BY_KIND[kind],
            "config": {
                "label": _card_label(str(node.get("label") or node_id)),
                "subtitle": subtitle,
            },
        })
    cards.sort(key=lambda c: c["id"])
    return cards, card_of, community_of, skipped


def _wires(links: list[dict], card_of: dict[str, str], label_length: int) -> list[dict]:
    rows = []
    for link in links:
        relation = str(link.get("relation") or "")
        if relation in STRUCTURAL_RELATIONS:
            continue
        a = card_of.get(str(link.get("source")))
        b = card_of.get(str(link.get("target")))
        if a is None or b is None:
            continue
        context = link.get("context")
        key = (a, b, relation, str(link.get("source_location") or ""), str(context or ""))
        rows.append((key, link))
    rows.sort(key=lambda row: row[0])
    used: set[str] = set()
    wires = []
    for (a, b, relation, _loc, _ctx), link in rows:
        base = f"w-{a}-{b}-{relation}"
        wid, n = base, 1
        while wid in used:
            n += 1
            wid = f"{base}-{n}"
        used.add(wid)
        config = {"label": edge_label(relation, link.get("context"), label_length)}
        if link.get("confidence") is not None:
            config["basis"] = link["confidence"]
        wires.append({
            "id": wid,
            "a": a,
            "aSide": "out",
            "b": b,
            "bSide": "in",
            "canvasId": CANVAS,
            "config": config,
        })
    return wires


def _groups(community_of: dict[str, int | None], community_labels: dict | None) -> list[dict]:
    members: dict[int, list[str]] = {}
    for cid, community in community_of.items():
        if community is not None:
            members.setdefault(community, []).append(cid)
    labels = {}
    for key, value in (community_labels or {}).items():
        try:
            labels[int(key)] = value
        except (TypeError, ValueError):
            continue
    return [
        {
            "id": f"community-{n}",
            "symbolId": "group",
            "canvasId": CANVAS,
            "config": {
                "label": str(labels[n]) if n in labels else f"Community {n}",
                "members": sorted(members[n]),
            },
        }
        for n in sorted(members)
    ]


def _links(data: dict) -> list[dict]:
    links = data.get("links")
    if links is None:
        links = data.get("edges")
    return list(links or [])


def graph_to_sov(data: dict, *, community_labels: dict | None = None,
                 label_length: int = LABEL_LENGTH, title: str | None = None) -> dict:
    """The Schematically document for a loaded graph.json dict. Pure: no I/O."""
    if not isinstance(label_length, int) or label_length < 1:
        raise ValueError(f"label_length must be a positive integer, got {label_length!r}")
    cards, card_of, community_of, _skipped = _cards(list(data.get("nodes") or []))
    groups = _groups(community_of, community_labels)
    card_ids = {c["id"] for c in cards}
    for group in groups:
        if group["id"] in card_ids:
            raise ValueError(f"group id {group['id']!r} is also a card id")
    wires = _wires(_links(data), card_of, label_length)
    return {
        "schema": SCHEMA,
        "id": "schematify",
        "revision": 0,
        "meta": {"title": title or f"schematify: {len(cards)} cards"},
        "references": [],
        "components": cards + groups,
        "wires": wires,
    }


def dumps_sov(document: dict) -> str:
    """The document's bytes: indent 1, UTF-8 text as is, one final newline."""
    return json.dumps(document, indent=1, ensure_ascii=False) + "\n"


def sov_counts(document: dict, skipped: int) -> dict:
    kind_by_symbol = {v: k for k, v in SYMBOL_BY_KIND.items()}
    cards = {kind: 0 for kind in SYMBOL_BY_KIND}
    groups = 0
    for component in document["components"]:
        if component["symbolId"] == "group":
            groups += 1
        else:
            cards[kind_by_symbol[component["symbolId"]]] += 1
    wires = {basis: 0 for basis in BASES}
    for wire in document["wires"]:
        basis = wire["config"].get("basis")
        if basis in wires:
            wires[basis] += 1
    return {"cards": cards, "skipped": skipped, "wires": wires, "groups": groups}


def to_sov(data: dict, output_path, *, schematically_dir=None, community_labels: dict | None = None,
           label_length: int = LABEL_LENGTH, layout: bool = True, node: str = "node") -> dict:
    """Write the document to ``output_path``, laid out by a Schematically checkout.

    Refuses with ValueError, before writing anything, when the checkout, its
    ``scripts/layout_sov.mjs`` or ``node`` is missing while ``layout`` is true. A
    non-zero exit of the layout script raises RuntimeError with its output, and
    ``output_path`` is left unwritten.
    """
    output_path = Path(output_path)
    node_exe = None
    script = None
    if layout:
        if schematically_dir is None:
            raise ValueError(
                "no Schematically checkout: schematically_dir is None and layout is on"
            )
        script = Path(schematically_dir) / "scripts" / "layout_sov.mjs"
        if not script.is_file():
            raise ValueError(f"layout script not found: {script}")
        node_exe = shutil.which(node)
        if node_exe is None:
            raise ValueError(f"node not found: {node!r} is not on PATH")
    document = graph_to_sov(data, community_labels=community_labels, label_length=label_length)
    skipped = len(data.get("nodes") or []) - sum(
        1 for c in document["components"] if c["symbolId"] != "group"
    )
    text = dumps_sov(document)
    if not layout:
        write_text_atomic(output_path, text)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(output_path.parent), prefix=".schematify-", suffix=".sov")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            proc = subprocess.run(
                [node_exe, str(script), tmp, "--out", str(output_path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"layout_sov.mjs exited {proc.returncode}\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    counts = sov_counts(document, skipped)
    counts["output"] = str(output_path)
    return counts
