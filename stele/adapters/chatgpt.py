"""
ChatGPT export adapter: every message on every branch, one conversation at a time.

Input is the sealed bundle of the ChatGPT export splitter (the Wasm extractor
chatgpt-export-split, stele/extractors): index.jsonl plus one
conversation-NNNNNN.json per conversation, each the exact bytes of one element
of the export's conversations.json. The adapter reads them through the
SealedBundle, so every byte it sees is re-verified against the sealed digest.
iter_chunks() is a generator that parses a conversation only when the
previous one's chunks are consumed, so it holds one parsed conversation at a
time. transform(), which the Dispatcher calls, returns the full chunk list
(the adapter contract), so it holds the text of every conversation at once.

A ChatGPT conversation is a tree, not a list. Its `mapping` holds every node
the user ever produced: when a message is edited or a response regenerated,
the old version stays as a sibling branch, and `current_node` only marks the
leaf the UI shows. Exporters that follow current_node back to the root drop
every other branch. This adapter keeps them all:

- one chunk per node whose message has content, on every branch;
- metadata that rebuilds the tree: the node's nearest ancestor with a chunk
  (`parent_node_id`, so skipped empty/system nodes do not break the chain),
  its raw `parent`, its position among its siblings, its depth, and whether it
  lies on the branch the UI showed (`on_current_branch`);
- per-conversation counts of nodes, leaves (= branches) and chunks, and the
  byte range of the conversation in the original export.

Chunk ids are "<conversation key>:<node key>", stable across re-exports, so a
re-delivered conversation updates rather than duplicates. Both parts are
percent-encoded (every character but unreserved ones), so neither contains
":" or "#": the "#<n>" given to the n-th copy of a conversation repeated in
one export can never collide with a real conversation id, and the ":"
between the parts is unambiguous. ChatGPT's ids are UUIDs, which encoding
leaves unchanged.

Traversal is iterative and order is deterministic (roots and siblings in
export order), so the same bundle always yields the same chunks, as #13
requires of adapters. A mapping with a cycle, a node reached twice, or a
current_node that does not exist is malformed and fails the transform.
"""
from __future__ import annotations

import json
from typing import Any, Iterator
from urllib.parse import quote

from ..contracts.adapter import SealedBundle, SteleChunk, make_chunk

SPLITTER = "chatgpt-export-split"
INDEX = "index.jsonl"


class MalformedExportError(ValueError):
    """The bundle is not a well-formed split ChatGPT export."""


class ChatGPTExportAdapter:
    """SteleAdapter for sealed chatgpt-export-split bundles."""

    def transform(self, bundle: SealedBundle) -> list[SteleChunk]:
        return list(self.iter_chunks(bundle))

    def iter_chunks(self, bundle: SealedBundle) -> Iterator[SteleChunk]:
        """Chunks for every conversation, reading one conversation at a time."""
        if bundle.parser is not None and bundle.parser.name != SPLITTER:
            raise MalformedExportError(
                f"expected a {SPLITTER} bundle, got one from {bundle.parser.name}"
            )
        if INDEX not in bundle.manifest:
            raise MalformedExportError(f"bundle has no {INDEX}")
        seen_ids: dict[str, int] = {}
        for entry in _index(bundle):
            raw = bundle.read(entry["path"])
            try:
                conversation = json.loads(raw)
            except ValueError as exc:
                raise MalformedExportError(f"{entry['path']} is not JSON: {exc}") from exc
            if not isinstance(conversation, dict):
                raise MalformedExportError(f"{entry['path']} is not a JSON object")
            conv_id = quote(str(
                conversation.get("conversation_id") or conversation.get("id")
                or f"conversation-{entry['index']:06d}"
            ), safe="")
            # The same conversation twice in one export keeps both copies apart.
            # An encoded id never contains "#", so the suffix cannot collide.
            seen_ids[conv_id] = seen_ids.get(conv_id, 0) + 1
            if seen_ids[conv_id] > 1:
                conv_id = f"{conv_id}#{seen_ids[conv_id]}"
            yield from _conversation_chunks(conversation, conv_id, entry, bundle.record_id)
            del conversation, raw  # only one conversation is ever held


def _index(bundle: SealedBundle) -> Iterator[dict[str, Any]]:
    lines = bundle.read_text(INDEX).splitlines()
    for number, line in enumerate(lines):
        try:
            entry = json.loads(line)
            path = entry["path"]
            entry["index"], entry["offset"], entry["length"] = (
                int(entry["index"]), int(entry["offset"]), int(entry["length"])
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise MalformedExportError(f"{INDEX} line {number + 1} is malformed: {exc}") from exc
        if path not in bundle.manifest:
            raise MalformedExportError(f"{INDEX} names {path!r}, which is not in the bundle")
        yield entry


def _conversation_chunks(
    conversation: dict[str, Any], conv_id: str, entry: dict[str, Any], record_id: str
) -> Iterator[SteleChunk]:
    mapping = conversation.get("mapping")
    if mapping is None:
        mapping = {}
    if not isinstance(mapping, dict):
        raise MalformedExportError(f"{conv_id}: mapping is not an object")
    order = {node_id: i for i, node_id in enumerate(mapping)}

    for node_id, node in mapping.items():
        kids = node.get("children") if isinstance(node, dict) else None
        if kids is not None and not isinstance(kids, list):
            raise MalformedExportError(f"{conv_id}: node {node_id!r} has children that are not a list")

    def children_of(node_id: str) -> list[str]:
        return _children(mapping, mapping.get(node_id))

    roots = [
        node_id for node_id, node in mapping.items()
        if not isinstance(node, dict) or not _in_mapping(mapping, node.get("parent"))
    ]
    current = conversation.get("current_node")
    if current is not None and not _in_mapping(mapping, current):
        raise MalformedExportError(f"{conv_id}: current_node {current!r} is not in the mapping")
    current_branch = _ancestors(mapping, current) if current is not None else set()
    leaves = [n for n in mapping if not children_of(n)]

    base = {
        "source": "chatgpt-export",
        "stele_record_id": record_id,
        "conversation_id": conv_id,
        "conversation_title": conversation.get("title"),
        "conversation_create_time": conversation.get("create_time"),
        "conversation_update_time": conversation.get("update_time"),
        "export_file": entry["path"],
        "export_offset": entry["offset"],
        "export_length": entry["length"],
        "node_count": len(mapping),
        "branch_count": len(leaves),
    }

    visited: set[str] = set()
    # (node_id, nearest ancestor that produced a chunk, depth, sibling index)
    stack: list[tuple[str, str | None, int, int]] = [
        (root, None, 0, i) for i, root in reversed(list(enumerate(
            sorted(roots, key=lambda n: order[n])
        )))
    ]
    while stack:
        node_id, chunk_parent, depth, sibling_index = stack.pop()
        if node_id in visited:
            raise MalformedExportError(f"{conv_id}: node {node_id!r} is reached twice")
        visited.add(node_id)
        node = mapping[node_id] if isinstance(mapping[node_id], dict) else {}
        message = node.get("message") if isinstance(node.get("message"), dict) else None
        text = _render(message) if message is not None else ""

        emitted_as = chunk_parent
        if text:
            author = message.get("author")
            if author is None:
                author = {}
            elif not isinstance(author, dict):
                raise MalformedExportError(
                    f"{conv_id}: node {node_id!r} has an author that is not an object"
                )
            yield make_chunk(
                f"{conv_id}:{quote(node_id, safe='')}",
                text,
                **base,
                node_id=node_id,
                message_id=message.get("id"),
                parent_node_id=chunk_parent,
                raw_parent_id=node.get("parent"),
                depth=depth,
                sibling_index=sibling_index,
                sibling_count=_sibling_count(mapping, node),
                child_count=len(children_of(node_id)),
                is_leaf=not children_of(node_id),
                on_current_branch=node_id in current_branch,
                role=author.get("role"),
                author_name=author.get("name"),
                recipient=message.get("recipient"),
                content_type=(message.get("content") or {}).get("content_type"),
                create_time=message.get("create_time"),
            )
            emitted_as = node_id

        kids = children_of(node_id)
        for i in reversed(range(len(kids))):
            stack.append((kids[i], emitted_as, depth + 1, i))

    unreached = set(mapping) - visited
    if unreached:
        raise MalformedExportError(
            f"{conv_id}: {len(unreached)} node(s) are unreachable (a cycle in the mapping)"
        )


def _ancestors(mapping: dict[str, Any], node_id: str) -> set[str]:
    """node_id and every ancestor, following parent links (cycle-safe)."""
    seen: set[str] = set()
    while _in_mapping(mapping, node_id) and node_id not in seen:
        seen.add(node_id)
        node = mapping[node_id]
        node_id = node.get("parent") if isinstance(node, dict) else None
    return seen


def _sibling_count(mapping: dict[str, Any], node: dict[str, Any]) -> int:
    parent = mapping.get(node.get("parent")) if _in_mapping(mapping, node.get("parent")) else None
    if not isinstance(parent, dict):
        return 1
    return len(_children(mapping, parent))


def _in_mapping(mapping: dict[str, Any], node_id: Any) -> bool:
    """True if node_id names a node (ids are strings; anything else never does)."""
    return isinstance(node_id, str) and node_id in mapping


def _children(mapping: dict[str, Any], node: Any) -> list[str]:
    """A node's children that the mapping contains.

    A node that is not an object has none. Children the mapping does not
    contain are dangling references and are skipped. A non-list children
    field is rejected up front by _conversation_chunks.
    """
    kids = node.get("children") if isinstance(node, dict) else None
    return [k for k in kids or [] if _in_mapping(mapping, k)]


def _render(message: dict[str, Any]) -> str:
    """The text of a message, for every content type the export uses.

    Non-text parts (images, audio, files) are kept as a bracketed reference so
    a branch that only differs by an attachment is still distinguishable.
    Content of an unknown type falls back to its string fields, never dropped
    silently: a node with nothing renderable simply has no chunk.
    """
    content = message.get("content")
    if not isinstance(content, dict):
        return ""
    kind = content.get("content_type")
    pieces: list[str] = []

    if isinstance(content.get("parts"), list):
        for part in content["parts"]:
            if isinstance(part, str):
                if part:
                    pieces.append(part)
            elif isinstance(part, dict):
                pieces.append(_part_reference(part))
    elif kind == "code":
        pieces.append(content.get("text") or "")
    elif kind == "user_editable_context":
        for key in ("user_profile", "user_instructions"):
            if content.get(key):
                pieces.append(str(content[key]))
    elif kind == "thoughts":
        for thought in content.get("thoughts") or []:
            if isinstance(thought, dict):
                pieces.append("\n".join(
                    str(thought[k]) for k in ("summary", "content") if thought.get(k)
                ))
    else:
        for key in ("text", "result", "content", "summary"):
            value = content.get(key)
            if isinstance(value, str) and value:
                pieces.append(value)
    return "\n\n".join(p for p in pieces if p).strip()


def _part_reference(part: dict[str, Any]) -> str:
    kind = part.get("content_type") or "attachment"
    if isinstance(part.get("text"), str) and part["text"]:
        return part["text"]
    pointer = part.get("asset_pointer") or part.get("name") or ""
    return f"[{kind}{': ' + str(pointer) if pointer else ''}]"
