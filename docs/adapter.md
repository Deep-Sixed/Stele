# Adapter Contract

## Goal

Define and enforce the typed interface that any downstream writer must
implement to write through Stele.  No parser-facing component
may write to a target store via any other path.

---

## Adapter contract (`contracts/adapter.py`)

```python
class SteleAdapter(Protocol):
    def transform(self, bundle: SealedBundle) -> list[SteleChunk]: ...
```

`transform()` is separated from writing. The adapter reads a sealed bundle and
returns typed chunks. It never receives a file path, DB handle, target
connection or writer reference. All writes, and their removal, are owned by
the Dispatcher.

**Adapters read verified bytes, not files.** A `SealedBundle` carries the
record's provenance (`record_id`, `run_id`, `artifact_hash`, `manifest`,
`parser`, `parser_config`, `source_hash`, `source_path`) and serves each
artifact by digest from the evidence archive: `bundle.read(path)` re-hashes
the bytes on every read, so a file edited after sealing, or a corrupted blob,
can never reach an adapter. There is no `artifact_dir`.

**Adapters must be deterministic.** A retried delivery re-runs `transform()`
and must get the same chunks; the Dispatcher refuses a retry whose chunks
differ from the ones its earlier attempt announced.

**Adapters are trusted code.** They run in the host process. "Must not write
to any external store" and "must be deterministic" are a contract, not an
enforced capability boundary: Python cannot stop an adapter from opening its
own connections (see the side-channel test in
`test_adapter_contract.py`). Only parser execution is sandboxed. An
adapter is reviewed and deployed like any other Stele code, never loaded from
a parser or a document.

Adapters have no invalidation hook (v1.0's `on_invalidation()`, which nothing
called, is gone). Removing delivered data is a target write, done by the
writer that wrote it.

`SteleChunk` fields: `chunk_id`, `content`, `content_hash` (sha256 verified),
`token_count` (a non-negative `int`), `metadata`. `metadata` must be a JSON
object with exactly one canonical encoding: it must survive a JSON round
trip unchanged (string keys only; no tuples, sets, NaN, infinities or other
non-JSON values). It is part of what a delivery fingerprints, like the
content.

`SteleTarget` union:
- `LightRAGTarget(workspace: str)`
- `HindsightTarget(instance: str)`

These are example target types; register a `TargetWriter` for each target
type you use.

---

## Target writers (`contracts/dispatcher.py`)

```python
class TargetWriter(Protocol):
    def write_chunks(self, chunks, target, *, dispatch_id: str) -> None: ...
    def remove_delivery(self, target, *, dispatch_id: str) -> int | None: ...
```

- `dispatch_id` is the **idempotency key**: writing the same chunks again
  under the same `dispatch_id` must not create duplicates (e.g. key rows by
  `(dispatch_id, chunk_id)`, or prefix ids with it).
- A writer that fails after writing some chunks raises
  `PartialWriteError(chunks_written=n)`. Any other exception is recorded as
  "unknown how many were written", never as zero.
- `remove_delivery` removes or tombstones everything written under a
  `dispatch_id`, idempotently, and returns how many chunks it removed (or
  `None` if unknown).

---

## Dispatcher and the delivery log

```
Dispatcher.dispatch(adapter, record_id, target)
      │
      ├── re-read the record from the ledger   →  DispatchRefusedError unless SEALED
      ├── re-verify the bundle in the archive  →  DispatchRefusedError if not intact
      ├── adapter.transform(SealedBundle)      →  list[SteleChunk]
      ├── validate_chunks(chunks)              →  logged failure if invalid
      ├── log  intent  (planned, chunks digest)          ── durable before the write;
      │                                                    binds the delivery's payload
      ├── TargetWriter.write_chunks(chunks, target, dispatch_id=…)
      └── log  receipt (done)  |  failure (done or unknown, error)
```

Refusals log nothing and write nothing. The state check is against the live
ledger, so a stale in-memory copy of a record that has since been
invalidated is refused.

A **delivery** is one record sent to one target, identified by its
`dispatch_id`. Its history is an append-only list of events in the ledger
database (`stele/ledger/delivery.py`; `UPDATE` and `DELETE` are refused by
triggers): `intent`, `receipt`, `failure`, `removal_intent`,
`removal_receipt`, `removal_failure`. Because each intent is committed before
its write and each outcome after it, a crash at any point leaves a log that
bounds what the target holds:

| Crash | Log shows | Target holds |
|-------|-----------|--------------|
| after the intent, before writing | `in_flight` | nothing |
| mid-write | `in_flight` | some of the planned chunks |
| after writing, before the receipt | `in_flight` | all of them |

Each write attempt gets its own `attempt_id` (ledger schema 8), carried by
its intent and by its receipt or failure. Two dispatchers can deliver the
same record to the same target at once, and the log keeps their attempts
apart: one attempt's receipt never closes another's intent, so an attempt
that never reported still counts as possibly written after the other one
finishes. Events from before schema 8 have no `attempt_id` and are read one
attempt at a time, as they were written.

`Delivery.explain()` states that bound ("between 0 and N chunks … under
dispatch_id …"). Dispatching the same record to the same target again reuses
the delivery's `dispatch_id`, so the retry completes it without duplicates;
a delivery that already has a receipt is not written again
(`already_delivered=True`).

A delivery writes **one payload**. Its first intent binds it to the chunks'
digest, which covers every chunk's `chunk_id`, `content_hash`, `token_count`
and `metadata`, in order. The intent is recorded in a write transaction that
first checks any earlier intent, and a database trigger refuses an intent
with a different digest. So a retry, or a concurrent dispatch of the same
record to the same target, that presents a different payload (even one that
differs only in metadata) is refused: `dispatch()` returns `status="failed"`,
its writer is never called, and the delivery's log is left unchanged.
Deliveries bound before ledger schema 6 recorded a digest of ids and
content hashes only, and are checked that far.

A failed dispatch never changes the record's ledger state; invalidating
after a failure is the caller's decision.

## Invalidation

```
Dispatcher.invalidate(record_id, reason)
      ├── ledger: SEALED → INVALIDATED
      └── for every delivery that may have written data (per its log):
            log removal_intent → writer.remove_delivery(target, dispatch_id=…)
            → log removal_receipt (done) | removal_failure (error)
```

- Deliveries the log proves hold nothing at the target are skipped: those
  that wrote nothing, and those removed after their last write concluded. A
  delivery with a write still in flight (an attempt's intent without that
  attempt's outcome) when
  it was removed may have landed since, e.g. if the dispatching process
  crashed after writing, so every `retract()` removes it again (removal is
  idempotent). A failed removal is retried by calling `invalidate()` or
  `retract()` again.
- A write that races an invalidation, and lands after the removal pass, is
  removed by the dispatching Dispatcher as soon as it records the outcome:
  a receipt, or a failure that may have written some chunks.
- A record invalidated directly in the ledger (`stele.replay.invalidation`)
  has its deliveries removed by `Dispatcher.retract_invalidated()`.

Ledger states are defined once, in [ledger.md](ledger.md#state-machine).
Delivery is never a ledger state.

---

## Adapters in this repo

### ChatGPT export (`stele/adapters/chatgpt.py`)

`ChatGPTExportAdapter` takes the sealed bundle of the Wasm splitter
`chatgpt-export-split` (`stele/extractors`): `index.jsonl` plus one file per
conversation, each holding the exact bytes of its element of
`conversations.json`.

- **Streamed.** `iter_chunks()` reads, verifies and parses one conversation
  at a time, so it holds one parsed conversation at once. `transform()`, which
  the Dispatcher calls, is `list(iter_chunks())`: it returns every chunk, so
  it holds the text of the whole export.
- **Every branch.** A conversation is a tree. Editing a message or
  regenerating a response keeps the old version as a sibling, and
  `current_node` only marks the leaf the UI shows. The adapter walks the whole
  `mapping` iteratively, so any depth works. Every node whose message has
  content becomes one chunk, `<conversation key>:<node key>`, stable across
  re-exports. Both parts are percent-encoded, so neither contains `:` or
  `#`. The `#<n>` given to the n-th copy of a conversation repeated in one
  export can't collide with a real conversation id, and two different
  (conversation, node) pairs never share a chunk id. ChatGPT's UUID ids are
  unchanged by the encoding. A message whose `author` isn't an object is
  malformed.
- **Rebuildable.** Chunk metadata rebuilds each branch:
  - `parent_node_id`, the nearest ancestor that has a chunk, so empty system
    nodes don't break the chain;
  - `raw_parent_id`, `depth`, `sibling_index` / `sibling_count`, `is_leaf`;
  - `on_current_branch`;
  - the conversation's `node_count` and `branch_count` (its leaves), and its
    byte range in the original export (`export_offset` / `export_length`).
- **Content types.** It renders `text`, `multimodal_text` (non-text parts
  become `[image_asset_pointer: …]` references), `code`, `execution_output`,
  `thoughts` and `user_editable_context`. Other types fall back to their
  string fields.
- **Deterministic.** Roots and siblings come in export order.
- **Malformed input fails the transform:** a cycle, a node reached twice, a
  `current_node` that doesn't exist, or a bundle from another parser. The
  Dispatcher records the failure and writes nothing.

### Canonical extraction (`stele/adapters/extraction.py`)

`ExtractionAdapter` delivers the `stele.extraction` v1 units of any bundle
that has a normalizer: Markdown from MinerU, Marker and Docling, and ChatGPT
conversations. It emits one chunk per unit, with its kind, order, section
parent and anchor. Before returning, it verifies every unit against the
sealed bytes with the trusted resolver. See [extraction.md](extraction.md).

## Invariants proven

| Proof | Test class |
|-------|-----------|
| Sealed artifact passed to adapter | `TestAdapterReceivesSealedRecord` |
| Adapter returns typed SteleChunk records | `TestAdapterReturnsTypedChunks` |
| Invalid adapter output rejected (empty, bad hash, dup ids) | `TestInvalidAdapterOutputRejected` |
| Adapter cannot bypass dispatcher write path | `TestAdapterCannotBypassDispatcher` |
| Target writes are represented as intents/results | `TestTargetWritesAsResults` |
| Adapter failure does not invalidate artifact automatically | `TestAdapterFailureIsolation` |
| Dispatcher records success/failure separately from ledger | `TestDispatchLog` |

Durable dispatch and invalidation (#13) are proved in
`tests/test_durable_dispatch.py`: refusal of non-sealed and stale records,
refusal of corrupted archived bytes, crash injection after the intent,
mid-write and before the receipt with duplicate-free retries, and removal
with receipts on invalidation.
