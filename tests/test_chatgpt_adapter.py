"""
ChatGPT export adapter (roadmap #15: "ChatGPT export adapter: streamed,
keeping every conversation branch").

  1. Every message on every branch becomes a chunk, not just the branch the
     UI showed, and the chunks' metadata rebuilds each branch.
  2. It streams: one conversation is read (and verified) at a time.
  3. Every content type the export uses renders to text.
  4. Deterministic output; malformed exports fail the transform.
  5. End to end: Wasm split → sealed record → Dispatcher → target.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stele.adapters import ChatGPTExportAdapter, MalformedExportError
from stele.archive import BlobStore, Source
from stele.contracts.adapter import LightRAGTarget, SealedBundle
from stele.contracts.dispatcher import Dispatcher
from stele.extractors import CHATGPT_EXPORT_SPLIT_SPEC
from stele.ledger.delivery import DeliveryStatus
from stele.ledger.models import ParserIdentity
from stele.ledger.transaction import record_run
from stele.replay.parsers import run_parser
from tests.ledger_helpers import open_ledger


def _msg(node_id, role, parts=None, *, content=None, name=None, t=None):
    return {
        "id": f"m-{node_id}",
        "author": {"role": role, "name": name, "metadata": {}},
        "create_time": t,
        "content": content or {"content_type": "text", "parts": parts or []},
        "recipient": "all",
    }


def _node(node_id, parent, children, message=None):
    return {"id": node_id, "message": message, "parent": parent, "children": children}


# An edited question and a regenerated answer:
#
#   root (system, empty)
#   ├── q1  "What is 2+2?"
#   │   ├── a1  "5"           (first answer)
#   │   └── a1r "4"           (regenerated; the branch the UI shows)
#   └── q1e "What is 2+3?"    (the user edited q1)
#       └── a2  "5"
BRANCHED = {
    "title": "Arithmetic, edited",
    "create_time": 1726000000.0,
    "update_time": 1726000900.0,
    "conversation_id": "conv-branched",
    "current_node": "a1r",
    "mapping": {
        "root": _node("root", None, ["q1", "q1e"], _msg("root", "system", [""])),
        "q1": _node("q1", "root", ["a1", "a1r"], _msg("q1", "user", ["What is 2+2?"], t=1.0)),
        "a1": _node("a1", "q1", [], _msg("a1", "assistant", ["5"], t=2.0)),
        "a1r": _node("a1r", "q1", [], _msg("a1r", "assistant", ["4"], t=3.0)),
        "q1e": _node("q1e", "root", ["a2"], _msg("q1e", "user", ["What is 2+3?"], t=4.0)),
        "a2": _node("a2", "q1e", [], _msg("a2", "assistant", ["5"], t=5.0)),
    },
}

CONTENT_TYPES = {
    "title": "Every content type",
    "id": "conv-types",  # older exports carry only "id"
    "current_node": "e",
    "mapping": {
        "a": _node("a", None, ["b"], _msg("a", "user", content={
            "content_type": "multimodal_text",
            "parts": [{"content_type": "image_asset_pointer", "asset_pointer": "file-service://f1"},
                      "What is in this picture?"],
        })),
        "b": _node("b", "a", ["c"], _msg("b", "assistant", content={
            "content_type": "code", "language": "python", "text": "print('cat')",
        }, name=None)),
        "c": _node("c", "b", ["d"], _msg("c", "tool", content={
            "content_type": "execution_output", "text": "cat",
        }, name="python")),
        "d": _node("d", "c", ["e"], _msg("d", "assistant", content={
            "content_type": "thoughts",
            "thoughts": [{"summary": "Looking", "content": "It is a cat."}],
        })),
        "e": _node("e", "d", [], _msg("e", "user", content={
            "content_type": "user_editable_context",
            "user_profile": "Likes cats", "user_instructions": "Be brief",
        })),
    },
}


def _export(*conversations) -> bytes:
    return json.dumps(list(conversations), ensure_ascii=False, indent=1).encode("utf-8")


def _bundle(tmp_path: Path, export: bytes, *, parser: str = "chatgpt-export-split") -> SealedBundle:
    """A bundle laid out exactly as the splitter writes it (no Wasm needed)."""
    archive = BlobStore(tmp_path / "archive")
    elements = json.loads(export)
    manifest, index = {}, []
    for i, element in enumerate(elements):
        data = json.dumps(element, ensure_ascii=False).encode()
        path = f"conversation-{i:06d}.json"
        manifest[path] = archive.put_bytes(data)
        index.append(json.dumps({"index": i, "path": path, "offset": 0, "length": len(data)}))
    manifest["index.jsonl"] = archive.put_bytes(("\n".join(index) + "\n").encode())
    return SealedBundle(
        record_id="rec", run_id="run", artifact_hash=archive.put_tree(manifest),
        manifest=manifest, parser=ParserIdentity(parser, "1"), parser_config={},
        source_hash=None, source_kind=None, source_path="conversations.json",
        _archive=archive,
    )


def _by_node(chunks):
    return {c.metadata["node_id"]: c for c in chunks}


# ---------------------------------------------------------------------------
# 1 — every branch
# ---------------------------------------------------------------------------

class TestEveryBranch:

    def test_every_message_on_every_branch_is_a_chunk(self, tmp_path) -> None:
        chunks = ChatGPTExportAdapter().transform(_bundle(tmp_path, _export(BRANCHED)))
        nodes = _by_node(chunks)
        # root has no content; every other node, on any branch, is kept.
        assert set(nodes) == {"q1", "a1", "a1r", "q1e", "a2"}
        assert nodes["a1"].content == "5" and nodes["a1r"].content == "4"
        assert nodes["q1e"].content == "What is 2+3?"
        assert {c.chunk_id for c in chunks} == {f"conv-branched:{n}" for n in nodes}

    def test_metadata_marks_the_shown_branch(self, tmp_path) -> None:
        nodes = _by_node(ChatGPTExportAdapter().transform(_bundle(tmp_path, _export(BRANCHED))))
        shown = {n for n, c in nodes.items() if c.metadata["on_current_branch"]}
        assert shown == {"q1", "a1r"}
        meta = nodes["a1"].metadata
        assert (meta["branch_count"], meta["node_count"]) == (3, 6)
        assert (meta["sibling_index"], meta["sibling_count"]) == (0, 2)
        assert nodes["a1r"].metadata["sibling_index"] == 1
        assert meta["depth"] == 2 and meta["is_leaf"] and meta["role"] == "assistant"
        assert meta["conversation_title"] == "Arithmetic, edited"

    def test_branches_rebuild_from_parent_links(self, tmp_path) -> None:
        nodes = _by_node(ChatGPTExportAdapter().transform(_bundle(tmp_path, _export(BRANCHED))))

        def path(leaf):
            texts, node = [], leaf
            while node is not None:
                texts.append(nodes[node].content)
                node = nodes[node].metadata["parent_node_id"]
            return list(reversed(texts))

        leaves = sorted(n for n, c in nodes.items() if c.metadata["is_leaf"])
        assert [path(leaf) for leaf in leaves] == [
            ["What is 2+2?", "5"],       # a1: the first answer
            ["What is 2+2?", "4"],       # a1r: the regenerated answer (shown)
            ["What is 2+3?", "5"],       # a2: after the edit
        ]
        # The empty system root has no chunk, so the chain skips it; the raw
        # parent is kept for audit.
        assert nodes["q1"].metadata["parent_node_id"] is None
        assert nodes["q1"].metadata["raw_parent_id"] == "root"


# ---------------------------------------------------------------------------
# 2 — streaming
# ---------------------------------------------------------------------------

def test_reads_one_conversation_at_a_time(tmp_path, monkeypatch) -> None:
    bundle = _bundle(tmp_path, _export(BRANCHED, CONTENT_TYPES))
    reads: list[str] = []
    real_read = SealedBundle.read
    monkeypatch.setattr(
        SealedBundle, "read", lambda self, path: reads.append(path) or real_read(self, path)
    )
    chunks = ChatGPTExportAdapter().iter_chunks(bundle)
    next(chunks)
    assert reads == ["index.jsonl", "conversation-000000.json"]
    rest = list(chunks)
    assert reads == ["index.jsonl", "conversation-000000.json", "conversation-000001.json"]
    assert len(rest) == 4 + 5


# ---------------------------------------------------------------------------
# 3 — content types
# ---------------------------------------------------------------------------

def test_every_content_type_renders(tmp_path) -> None:
    nodes = _by_node(ChatGPTExportAdapter().transform(_bundle(tmp_path, _export(CONTENT_TYPES))))
    assert nodes["a"].content == (
        "[image_asset_pointer: file-service://f1]\n\nWhat is in this picture?"
    )
    assert nodes["b"].content == "print('cat')"
    assert nodes["b"].metadata["content_type"] == "code"
    assert nodes["c"].content == "cat" and nodes["c"].metadata["author_name"] == "python"
    assert nodes["d"].content == "Looking\nIt is a cat."
    assert nodes["e"].content == "Likes cats\n\nBe brief"
    assert {c.metadata["conversation_id"] for c in nodes.values()} == {"conv-types"}


# ---------------------------------------------------------------------------
# 4 — determinism and malformed exports
# ---------------------------------------------------------------------------

def test_output_is_deterministic(tmp_path) -> None:
    bundle = _bundle(tmp_path, _export(BRANCHED, CONTENT_TYPES))
    adapter = ChatGPTExportAdapter()
    assert adapter.transform(bundle) == adapter.transform(bundle)


def test_duplicate_conversation_ids_stay_apart(tmp_path) -> None:
    chunks = ChatGPTExportAdapter().transform(_bundle(tmp_path, _export(BRANCHED, BRANCHED)))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)) == 10
    assert "conv-branched#2:a1r" in ids


def test_a_renamed_duplicate_never_collides_with_a_real_id(tmp_path) -> None:
    # Two copies of "conv-branched" plus a real conversation already called
    # "conv-branched#2": the copy's suffix must not land on the real id.
    real = json.loads(json.dumps(BRANCHED))
    real["conversation_id"] = "conv-branched#2"
    chunks = ChatGPTExportAdapter().transform(
        _bundle(tmp_path, _export(BRANCHED, real, BRANCHED))
    )
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)) == 15
    conversations = {c.metadata["conversation_id"] for c in chunks}
    assert conversations == {"conv-branched", "conv-branched%232", "conv-branched#2"}


def test_colons_in_ids_cannot_make_two_chunks_one(tmp_path) -> None:
    # Without encoding, conversation "a:b" node "c" and conversation "a" node
    # "b:c" would both be the chunk "a:b:c".
    first = {"conversation_id": "a:b", "mapping": {
        "c": _node("c", None, [], _msg("c", "user", ["one"])),
    }}
    second = {"conversation_id": "a", "mapping": {
        "b:c": _node("b:c", None, [], _msg("b:c", "user", ["two"])),
    }}
    chunks = ChatGPTExportAdapter().transform(_bundle(tmp_path, _export(first, second)))
    assert sorted(c.chunk_id for c in chunks) == ["a%3Ab:c", "a:b%3Ac"]
    # The node id in metadata stays raw: extraction anchors point into mapping by it.
    assert {c.metadata["node_id"] for c in chunks} == {"c", "b:c"}


def test_uuid_ids_are_unchanged_by_encoding(tmp_path) -> None:
    conversation = json.loads(json.dumps(BRANCHED))
    conversation["conversation_id"] = "6710a4c5-1b2e-8003-a4b9-2f6c1a7d9e10"
    chunks = ChatGPTExportAdapter().transform(_bundle(tmp_path, _export(conversation)))
    assert "6710a4c5-1b2e-8003-a4b9-2f6c1a7d9e10:a1r" in {c.chunk_id for c in chunks}


def test_deep_conversation_does_not_recurse(tmp_path) -> None:
    depth = 5000
    mapping = {
        f"n{i}": _node(f"n{i}", f"n{i-1}" if i else None,
                       [f"n{i+1}"] if i < depth - 1 else [], _msg(f"n{i}", "user", [f"turn {i}"]))
        for i in range(depth)
    }
    chunks = ChatGPTExportAdapter().transform(
        _bundle(tmp_path, _export({"id": "deep", "mapping": mapping}))
    )
    assert len(chunks) == depth and chunks[-1].metadata["depth"] == depth - 1


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda c: c["mapping"]["q1"].update(parent="a1", children=["a1"]) or
               c["mapping"]["root"].update(children=["q1e"]),  # q1 <-> a1 cycle
     "unreachable"),
    (lambda c: c["mapping"]["q1e"]["children"].append("a1"), "reached twice"),
    (lambda c: c.update(current_node="nope"), "current_node"),
    (lambda c: c.update(mapping=[]), "mapping is not an object"),
    (lambda c: c["mapping"]["q1"].update(children="a1"), "children that are not a list"),
    (lambda c: c.update(current_node=["a1r"]), "current_node"),
    (lambda c: c["mapping"]["a1"]["message"].update(author="assistant"),
     "author that is not an object"),
])
def test_malformed_conversations_fail(tmp_path, mutate, message) -> None:
    conversation = json.loads(json.dumps(BRANCHED))
    mutate(conversation)
    with pytest.raises(MalformedExportError, match=message):
        ChatGPTExportAdapter().transform(_bundle(tmp_path, _export(conversation)))


def test_non_object_nodes_and_ids_are_tolerated(tmp_path) -> None:
    conversation = json.loads(json.dumps(BRANCHED))
    conversation["mapping"]["stray"] = "x"                    # a node that is not an object
    conversation["mapping"]["a2"]["parent"] = ["q1e"]         # a parent id that is not a string
    conversation["mapping"]["q1e"]["children"] = [{"id": "a2"}]  # an id that is not a string
    chunks = ChatGPTExportAdapter().transform(_bundle(tmp_path, _export(conversation)))
    assert {c.metadata["node_id"] for c in chunks} >= {"q1", "a1", "a1r", "q1e", "a2"}


def test_refuses_other_parsers_bundles(tmp_path) -> None:
    with pytest.raises(MalformedExportError, match="expected a chatgpt-export-split"):
        ChatGPTExportAdapter().transform(_bundle(tmp_path, _export(BRANCHED), parser="marker"))


# ---------------------------------------------------------------------------
# 5 — end to end through the Wasm splitter, the ledger and the Dispatcher
# ---------------------------------------------------------------------------

class MemoryTarget:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], str] = {}

    def write_chunks(self, chunks, target, *, dispatch_id):
        for c in chunks:
            self.rows[(dispatch_id, c.chunk_id)] = c.content

    def remove_delivery(self, target, *, dispatch_id):
        keys = [k for k in self.rows if k[0] == dispatch_id]
        for k in keys:
            del self.rows[k]
        return len(keys)


@pytest.mark.usefixtures("wasm_backend")
class TestEndToEnd:

    def _sealed(self, tmp_path, export: bytes):
        ledger = open_ledger(tmp_path / "ledger.db")
        source = tmp_path / "export" / "conversations.json"
        source.parent.mkdir()
        source.write_bytes(export)
        run = run_parser(CHATGPT_EXPORT_SPLIT_SPEC, artifact_dir=tmp_path / "out",
                         parser_config={}, input_path=source, store=ledger.archive)
        assert run.succeeded, run.stderr
        record = record_run(ledger, run, parser=CHATGPT_EXPORT_SPLIT_SPEC.identity(),
                            parser_config={}, source=Source.from_path(source))
        return ledger, record

    def test_split_seal_dispatch(self, tmp_path) -> None:
        export = _export(BRANCHED, CONTENT_TYPES)
        ledger, record = self._sealed(tmp_path, export)
        target = MemoryTarget()
        dispatcher = Dispatcher(ledger)
        dispatcher.register_target(LightRAGTarget, target)

        result = dispatcher.dispatch(ChatGPTExportAdapter(), record, LightRAGTarget("chats"))

        assert result.status == "success" and result.chunks_written == 10
        assert sorted(chunk_id for _, chunk_id in target.rows) == sorted(
            [f"conv-branched:{n}" for n in ("q1", "a1", "a1r", "q1e", "a2")]
            + [f"conv-types:{n}" for n in "abcde"]
        )
        again = dispatcher.dispatch(ChatGPTExportAdapter(), record, LightRAGTarget("chats"))
        assert again.already_delivered and len(target.rows) == 10
        [delivery] = dispatcher.dispatch_log_for(record.record_id)
        assert delivery.status is DeliveryStatus.DELIVERED

    def test_export_offsets_point_at_the_source_bytes(self, tmp_path) -> None:
        export = _export(BRANCHED, CONTENT_TYPES)
        ledger, record = self._sealed(tmp_path, export)
        chunks = ChatGPTExportAdapter().transform(SealedBundle.from_record(record, ledger.archive))
        for chunk in chunks:
            meta = chunk.metadata
            element = export[meta["export_offset"]:meta["export_offset"] + meta["export_length"]]
            conversation = json.loads(element)
            assert meta["node_id"] in conversation["mapping"]
            assert meta["stele_record_id"] == record.record_id

    def test_malformed_export_fails_the_dispatch_and_writes_nothing(self, tmp_path) -> None:
        broken = json.loads(json.dumps(BRANCHED))
        broken["current_node"] = "missing"
        ledger, record = self._sealed(tmp_path, _export(broken))
        target = MemoryTarget()
        dispatcher = Dispatcher(ledger)
        dispatcher.register_target(LightRAGTarget, target)
        result = dispatcher.dispatch(ChatGPTExportAdapter(), record, LightRAGTarget("chats"))
        assert result.status == "failed" and "current_node" in result.error
        assert target.rows == {}
