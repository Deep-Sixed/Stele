"""
The hash-chained event log (roadmap #15: "Append-only, hash-chained event log").

Every fact the ledger learns is appended here as an event, in the same
database transaction as the change it describes, so the two cannot diverge:

  record.created / record.sealed / record.failed / record.invalidated
  delivery.opened / delivery.event        (the #13 delivery log)
  replay.logged                           (the #14 replay log)
  *.imported                              (state found when a ledger was migrated)

Each event's hash is SHA-256 over the canonical JSON of its sequence number,
kind, subject, body, timestamp and the previous event's hash (64 zeros for
the first). Changing, removing or reordering any event therefore breaks every
hash after it, and the head hash (seq, hash) commits to the whole history.

The event table is append-only (UPDATE and DELETE are refused by triggers),
but the chain does not rely on that: verify() recomputes it. verify_ledger()
also replays the chain to rebuild what the ledger's tables must contain, and
compares. That catches edits to artifact_records, the one table whose rows
legitimately change (state transitions), e.g. an INVALIDATED record flipped
back to SEALED by hand.

Anchoring: publish the head somewhere Stele does not control (a commit, a
ticket, a timestamping service). verify(anchor=(seq, hash)) then also proves
the chain still contains that exact event, so truncating the log and
rewriting it from there is detected too.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterator

from ..archive.records import canonical_json

if TYPE_CHECKING:
    from .store import LedgerStore

GENESIS = "0" * 64

EVENTS_DDL = (
    """
    CREATE TABLE events (
        seq        INTEGER PRIMARY KEY,   -- 1, 2, 3, ... without gaps
        kind       TEXT NOT NULL,
        subject    TEXT NOT NULL,         -- record_id, dispatch_id or replay_id
        body       TEXT NOT NULL,         -- canonical JSON
        at         TEXT NOT NULL,
        prev_hash  TEXT NOT NULL,
        hash       TEXT NOT NULL UNIQUE
    )
    """,
    "CREATE INDEX idx_events_subject ON events(subject, seq)",
    """
    CREATE TRIGGER events_append_only_u BEFORE UPDATE ON events
    BEGIN SELECT RAISE(ABORT, 'the event log is append-only'); END
    """,
    """
    CREATE TRIGGER events_append_only_d BEFORE DELETE ON events
    BEGIN SELECT RAISE(ABORT, 'the event log is append-only'); END
    """,
)


class ChainError(Exception):
    """The event chain is broken, or does not contain a given anchor."""


@dataclass(frozen=True)
class Event:
    seq: int
    kind: str
    subject: str
    body: dict[str, Any]
    at: str
    prev_hash: str
    hash: str


def event_hash(seq: int, kind: str, subject: str, body: str, at: str, prev_hash: str) -> str:
    """The hash of one event; body is its canonical JSON text."""
    return hashlib.sha256(canonical_json({
        "seq": seq, "kind": kind, "subject": subject, "body": body,
        "at": at, "prev": prev_hash,
    })).hexdigest()


@contextmanager
def write_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT on an autocommit connection; ROLLBACK on error.

    The write lock is taken up front, so reading the chain head and appending
    after it cannot interleave with another writer.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def append_event(
    conn: sqlite3.Connection, kind: str, subject: str, body: dict[str, Any], *, at: str | None = None
) -> Event:
    """Append one event. The caller holds the write transaction."""
    last = conn.execute("SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
    seq, prev = (last[0] + 1, last[1]) if last else (1, GENESIS)
    body_json = canonical_json(body).decode("utf-8")
    at = at or datetime.now(timezone.utc).isoformat()
    digest = event_hash(seq, kind, subject, body_json, at, prev)
    conn.execute(
        "INSERT INTO events (seq, kind, subject, body, at, prev_hash, hash) VALUES (?,?,?,?,?,?,?)",
        (seq, kind, subject, body_json, at, prev, digest),
    )
    return Event(seq, kind, subject, body, at, prev, digest)


def _row_event(row: sqlite3.Row) -> Event:
    return Event(
        seq=row["seq"], kind=row["kind"], subject=row["subject"],
        body=json.loads(row["body"]), at=row["at"],
        prev_hash=row["prev_hash"], hash=row["hash"],
    )


# ---------------------------------------------------------------------------
# Reading and verifying
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainReport:
    length: int
    head: tuple[int, str] | None        # (seq, hash) of the last event
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems


class EventLog:
    """Read access to a ledger's event chain."""

    def __init__(self, ledger: LedgerStore) -> None:
        from .store import connect

        self._conn = connect(ledger.db_path)

    def close(self) -> None:
        self._conn.close()

    def events(self, *, after: int = 0) -> Iterator[Event]:
        cursor = self._conn.execute("SELECT * FROM events WHERE seq > ? ORDER BY seq", (after,))
        for row in cursor:
            yield _row_event(row)

    def for_subject(self, subject: str) -> list[Event]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE subject=? ORDER BY seq", (subject,)
        ).fetchall()
        return [_row_event(r) for r in rows]

    def head(self) -> tuple[int, str] | None:
        row = self._conn.execute("SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        return (row[0], row[1]) if row else None

    def verify(self, *, anchor: tuple[int, str] | None = None) -> ChainReport:
        """Recompute the chain; optionally require it to contain anchor."""
        problems: list[str] = []
        prev, expected_seq, head = GENESIS, 1, None
        anchor_seen = anchor is None
        rows = self._conn.execute(
            "SELECT seq, kind, subject, body, at, prev_hash, hash FROM events ORDER BY seq"
        )
        for seq, kind, subject, body, at, prev_hash, stored in rows:
            if seq != expected_seq:
                problems.append(f"event {expected_seq} is missing (next is {seq})")
                expected_seq = seq
            if prev_hash != prev:
                problems.append(f"event {seq} does not follow event {seq - 1}")
            if event_hash(seq, kind, subject, body, at, prev_hash) != stored:
                problems.append(f"event {seq} ({kind} {subject}) was altered")
            if anchor is not None and seq == anchor[0]:
                anchor_seen = stored == anchor[1]
                if not anchor_seen:
                    problems.append(f"event {seq} no longer has the anchored hash")
            prev, head, expected_seq = stored, (seq, stored), expected_seq + 1
        if anchor is not None and not anchor_seen and (head is None or head[0] < anchor[0]):
            problems.append(f"the chain ends before the anchored event {anchor[0]} (truncated)")
        return ChainReport(length=head[0] if head else 0, head=head, problems=tuple(problems))


# ---------------------------------------------------------------------------
# Ledger consistency
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LedgerReport:
    chain: ChainReport
    problems: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.chain.ok and not self.problems


_RECORD_STATE = {
    "record.created": "pending",
    "record.sealed": "sealed",
    "record.failed": "failed",
    "record.invalidated": "invalidated",
}


def verify_ledger(ledger: LedgerStore, *, anchor: tuple[int, str] | None = None) -> LedgerReport:
    """Verify the chain, then check the ledger's tables are what it says.

    Replays the chain to derive every record's state and identity, every
    delivery and its events, and every replay, and reports each row that is
    missing from the chain, missing from the tables, or different. Every
    field an event commits to is compared; a record's source_path is checked
    against the archived Source its source_id names. artifact_dir is only
    where the run's output was on disk, not evidence, and is not checked.
    """
    from ..archive.store import ArchiveError
    from .hashing import sha256_manifest

    log = EventLog(ledger)
    try:
        chain = log.verify(anchor=anchor)
        records: dict[str, dict[str, Any]] = {}
        deliveries: dict[str, dict[str, Any]] = {}
        delivery_events: dict[str, list[tuple]] = {}
        replays: dict[str, dict[str, Any]] = {}
        problems: list[str] = []
        # Parse each body here rather than through events(), so a body that
        # is not JSON is reported like any other tampered event.
        for row in log._conn.execute("SELECT * FROM events ORDER BY seq"):
            try:
                _apply(_row_event(row), records, deliveries, delivery_events, replays)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                # A tampered event must be reported, never crash verification.
                problems.append(f"event {row['seq']} ({row['kind']}) has a malformed body: {exc!r}")

        conn = log._conn
        rows = {r["record_id"]: r for r in conn.execute("SELECT * FROM artifact_records")}
        for record_id in sorted(set(rows) | set(records)):
            row, want = rows.get(record_id), records.get(record_id)
            if row is None:
                problems.append(f"record {record_id} is in the chain but not in the ledger")
                continue
            if want is None:
                problems.append(f"record {record_id} is in the ledger but was never logged")
                continue
            parser = None if row["parser_name"] is None else {
                "name": row["parser_name"], "version": row["parser_version"],
                "image_digest": row["parser_image_digest"],
                "module_sha256": row["parser_module_sha256"],
            }
            try:
                config = None if row["parser_config"] is None else json.loads(row["parser_config"])
                conditions = (
                    None if row["run_conditions"] is None else json.loads(row["run_conditions"])
                )
                manifest_hash = sha256_manifest(json.loads(row["artifact_manifest"]))
            except (ValueError, TypeError, AttributeError) as exc:
                # A hand-edited row is reported like any other difference.
                problems.append(f"record {record_id} has a malformed column: {exc!r}")
                continue
            actual = {
                "run_id": row["run_id"], "artifact_hash": row["artifact_hash"],
                "source_hash": row["source_hash"], "source_kind": row["source_kind"],
                "source_id": row["source_id"], "backend": row["backend"], "parser": parser,
                "parser_config": config, "run_conditions": conditions, "state": row["state"],
                "error": row["error"], "created_at": row["created_at"],
            }
            for key in sorted(want):
                if actual[key] != want[key]:
                    problems.append(
                        f"record {record_id}: {key} is {actual[key]!r}, the chain says {want[key]!r}"
                    )
            if manifest_hash != row["artifact_hash"]:
                problems.append(f"record {record_id}: artifact_manifest no longer matches artifact_hash")
            # source_path is the locator of the Source that source_id hashes;
            # the chain commits to source_id, so the archive vouches for it.
            if row["source_id"] is not None:
                try:
                    locator = ledger.archive.get_source(row["source_id"]).locator
                except ArchiveError as exc:
                    problems.append(f"record {record_id}: its Source is unreadable: {exc}")
                else:
                    if row["source_path"] != locator:
                        problems.append(
                            f"record {record_id}: source_path is {row['source_path']!r}, "
                            f"its Source says {locator!r}"
                        )

        rows = {r["dispatch_id"]: r for r in conn.execute("SELECT * FROM deliveries")}
        for dispatch_id in sorted(set(rows) | set(deliveries)):
            row, want = rows.get(dispatch_id), deliveries.get(dispatch_id)
            if row is None or want is None:
                where = "the chain" if row is None else "the ledger"
                problems.append(f"delivery {dispatch_id} is only in {where}")
                continue
            if {k: row[k] for k in want} != want:
                problems.append(f"delivery {dispatch_id} differs from the chain")
        actual_events: dict[str, list[tuple]] = {}
        for r in conn.execute("SELECT * FROM delivery_events ORDER BY event_id"):
            actual_events.setdefault(r["dispatch_id"], []).append(_event_tuple(r))
        for dispatch_id in sorted(set(actual_events) | set(delivery_events)):
            if actual_events.get(dispatch_id, []) != delivery_events.get(dispatch_id, []):
                problems.append(f"delivery {dispatch_id}: its events differ from the chain")

        rows = {r["replay_id"]: r for r in conn.execute("SELECT * FROM replays")}
        for replay_id in sorted(set(rows) | set(replays)):
            row, want = rows.get(replay_id), replays.get(replay_id)
            if row is None or want is None:
                where = "the chain" if row is None else "the ledger"
                problems.append(f"replay {replay_id} is only in {where}")
                continue
            try:
                actual = _replay_row(row)
            except (ValueError, TypeError) as exc:
                problems.append(f"replay {replay_id} has a malformed column: {exc!r}")
                continue
            for key in sorted(want):
                if actual[key] != want[key]:
                    problems.append(
                        f"replay {replay_id}: {key} is {actual[key]!r}, the chain says {want[key]!r}"
                    )
        return LedgerReport(chain=chain, problems=tuple(problems))
    finally:
        log.close()


def _apply(
    ev: Event,
    records: dict[str, dict[str, Any]],
    deliveries: dict[str, dict[str, Any]],
    delivery_events: dict[str, list[tuple]],
    replays: dict[str, dict[str, Any]],
) -> None:
    """Fold one event into the state the ledger's tables must hold."""
    b = ev.body
    if ev.kind in ("record.created", "record.imported"):
        want = {
            "run_id": b["run_id"], "artifact_hash": b["artifact_hash"],
            "source_hash": b.get("source_hash"),
            "source_kind": b.get("source_kind"), "source_id": b.get("source_id"),
            "backend": b.get("backend"),
            "parser": b.get("parser"), "parser_config": b.get("parser_config"),
            "run_conditions": b.get("run_conditions"),
            "state": b.get("state", "pending"),
            "error": b.get("error"),
        }
        # Chained since the event log first committed to it; older events
        # leave the record's creation time unchecked.
        if "created_at" in b:
            want["created_at"] = b["created_at"]
        records[ev.subject] = want
    elif ev.kind in _RECORD_STATE:
        if ev.subject in records:
            want = records[ev.subject]
            want["state"] = _RECORD_STATE[ev.kind]
            if ev.kind == "record.failed":
                want["error"] = b["error"]
            elif ev.kind == "record.invalidated":
                want["error"] = f"INVALIDATED: {b['reason']}"
    elif ev.kind in ("delivery.opened", "delivery.imported"):
        deliveries[ev.subject] = {k: b[k] for k in ("record_id", "target_kind", "target")}
    elif ev.kind in ("delivery.event", "delivery_event.imported"):
        delivery_events.setdefault(ev.subject, []).append(_event_tuple(b))
    elif ev.kind in ("replay.logged", "replay.imported"):
        # Every field the event carries (replay.imported carries fewer).
        replays[ev.subject] = {
            k: b[k] for k in _REPLAY_FIELDS if k in b
        }


_REPLAY_FIELDS = (
    "record_id", "outcome", "reason", "replay_artifact_hash", "policy", "platform",
)


def _replay_row(row: sqlite3.Row) -> dict[str, Any]:
    """A replays row in the form the chain commits to (policy as an object)."""
    out = {k: row[k] for k in _REPLAY_FIELDS}
    out["policy"] = None if row["policy"] is None else json.loads(row["policy"])
    return out


def _event_tuple(e: Any) -> tuple:
    # attempt_id (schema 8) is absent from events chained before it existed,
    # whose rows hold NULL.
    attempt_id = e["attempt_id"] if "attempt_id" in e.keys() else None
    return (e["event"], e["planned"], e["chunks_digest"], e["done"], e["error"], attempt_id)


# ---------------------------------------------------------------------------
# Bodies (shared by the writers and migration)
# ---------------------------------------------------------------------------

def record_body(row: sqlite3.Row | dict[str, Any], *, with_state: bool = False) -> dict[str, Any]:
    """What record.created (and record.imported) commits to."""
    parser = None if row["parser_name"] is None else {
        "name": row["parser_name"], "version": row["parser_version"],
        "image_digest": row["parser_image_digest"],
        "module_sha256": row["parser_module_sha256"],
    }
    body = {
        "run_id": row["run_id"],
        "artifact_hash": row["artifact_hash"],
        "source_hash": row["source_hash"],
        "source_kind": row["source_kind"],
        "source_id": row["source_id"],
        "parser": parser,
        "parser_config": None if row["parser_config"] is None else json.loads(row["parser_config"]),
        "backend": row["backend"],
        # The device and limits the run executed under (#30); replays use
        # them, so they are committed to like the parser identity.
        "run_conditions": (
            None if row["run_conditions"] is None else json.loads(row["run_conditions"])
        ),
        # Observations (stele.identity) name the run by when it was recorded.
        "created_at": row["created_at"],
    }
    if with_state:
        body["state"] = row["state"]
        body["error"] = row["error"]
    return body


def backfill(conn: sqlite3.Connection) -> int:
    """Log the state of an existing ledger as *.imported events.

    Used once, when a ledger older than the event log is migrated. The caller
    holds the write transaction. Returns the number of events appended.
    """
    count = 0
    for row in conn.execute("SELECT * FROM artifact_records ORDER BY created_at, record_id").fetchall():
        append_event(conn, "record.imported", row["record_id"], record_body(row, with_state=True))
        count += 1
    for row in conn.execute("SELECT * FROM deliveries ORDER BY created_at, dispatch_id").fetchall():
        append_event(conn, "delivery.imported", row["dispatch_id"], {
            "record_id": row["record_id"], "target_kind": row["target_kind"], "target": row["target"],
        })
        count += 1
    for row in conn.execute("SELECT * FROM delivery_events ORDER BY event_id").fetchall():
        append_event(conn, "delivery_event.imported", row["dispatch_id"], {
            k: row[k] for k in ("event", "planned", "chunks_digest", "done", "error", "attempt_id")
        })
        count += 1
    for row in conn.execute("SELECT * FROM replays ORDER BY replayed_at, replay_id").fetchall():
        append_event(conn, "replay.imported", row["replay_id"], _replay_row(row))
        count += 1
    return count


# ---------------------------------------------------------------------------
# CLI: python -m stele.ledger.events <ledger.db> <archive-dir> [--anchor SEQ:HASH]
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    from ..archive import BlobStore
    from .store import LedgerSchemaError, LedgerStore

    ap = argparse.ArgumentParser(
        prog="python -m stele.ledger.events",
        description="Verify a ledger's hash-chained event log and its tables.",
    )
    ap.add_argument("ledger", type=Path)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--anchor", default=None, metavar="SEQ:HASH",
                    help="a previously published head the chain must still contain")
    args = ap.parse_args(argv)
    anchor = None
    if args.anchor:
        seq, _, digest = args.anchor.partition(":")
        anchor = (int(seq), digest)
    try:
        # Read-only: migrating a ledger older than the event log would chain
        # its current tables and then report them as verified.
        ledger = LedgerStore(args.ledger, BlobStore(args.archive), migrate=False)
    except LedgerSchemaError as exc:
        print(json.dumps({"ok": False, "length": 0, "head": None, "problems": [str(exc)]}, indent=2))
        return 1
    report = verify_ledger(ledger, anchor=anchor)
    print(json.dumps({
        "ok": report.ok,
        "length": report.chain.length,
        "head": None if report.chain.head is None else f"{report.chain.head[0]}:{report.chain.head[1]}",
        "problems": [*report.chain.problems, *report.problems],
    }, indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(_main())
