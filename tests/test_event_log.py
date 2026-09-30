"""
Hash-chained event log (roadmap #15: "Append-only, hash-chained event log").

  1. Every ledger fact is an event, committed with the change it describes:
     records, deliveries and their events, replays.
  2. The chain verifies, and so does the ledger replayed from it.
  3. Tampering is detected: an altered, deleted or reordered event; a record
     row edited by hand; a delivery or replay row changed or removed; a
     truncated-and-rewritten chain against a published anchor.
  4. Migration logs an existing ledger as imported events.
  5. The CLI reports the head and any problem.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from stele.contracts.adapter import LightRAGTarget, make_chunk
from stele.contracts.dispatcher import Dispatcher
from stele.ledger.events import GENESIS, EventLog, event_hash, verify_ledger
from stele.ledger.store import SCHEMA_VERSION, LedgerStore
from stele.replay.engine import replay_record
from stele.replay.parsers import ParserCatalog
from tests.ledger_helpers import PROVENANCE, open_ledger


@pytest.fixture
def ledger(tmp_path: Path) -> LedgerStore:
    return open_ledger(tmp_path / "ledger.db")


def _pending(ledger: LedgerStore, tmp_path: Path, text: str = "hello"):
    out = tmp_path / f"out-{uuid.uuid4().hex[:6]}"
    out.mkdir()
    (out / "doc.txt").write_text(text)
    return ledger.create_pending(
        **PROVENANCE, run_id=str(uuid.uuid4()), artifact_dir=out,
        artifact_paths=[out / "doc.txt"],
    )


class Target:
    def __init__(self) -> None:
        self.rows: dict = {}

    def write_chunks(self, chunks, target, *, dispatch_id):
        for c in chunks:
            self.rows[(dispatch_id, c.chunk_id)] = c.content

    def remove_delivery(self, target, *, dispatch_id):
        keys = [k for k in self.rows if k[0] == dispatch_id]
        for k in keys:
            del self.rows[k]
        return len(keys)


class Adapter:
    def transform(self, bundle):
        return [make_chunk(p, bundle.read_text(p)) for p in bundle.paths()]


def _lifecycle(ledger: LedgerStore, tmp_path: Path) -> dict[str, str]:
    """Every kind of fact: sealed, failed, invalidated, delivered, removed, replayed."""
    sealed = ledger.seal(_pending(ledger, tmp_path, "keep").record_id)
    failed = ledger.fail(_pending(ledger, tmp_path, "bad").record_id, "parser crashed")
    withdrawn = ledger.seal(_pending(ledger, tmp_path, "stale").record_id)

    dispatcher = Dispatcher(ledger)
    dispatcher.register_target(LightRAGTarget, Target())
    dispatcher.dispatch(Adapter(), sealed, LightRAGTarget("ws"))
    dispatcher.dispatch(Adapter(), withdrawn, LightRAGTarget("ws"))
    dispatcher.invalidate(withdrawn.record_id, "superseded")
    replay = replay_record(ledger, ParserCatalog(), sealed)  # UNREPLAYABLE: no spec
    return {"sealed": sealed.record_id, "failed": failed.record_id,
            "withdrawn": withdrawn.record_id, "replay": replay.replay_id}


def _raw(ledger: LedgerStore) -> sqlite3.Connection:
    """A connection with the append-only triggers dropped, as an attacker would."""
    conn = sqlite3.connect(ledger.db_path, isolation_level=None)
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
        conn.execute(f"DROP TRIGGER {name}")
    return conn


# ---------------------------------------------------------------------------
# 1 + 2 — every fact is chained, and the chain verifies
# ---------------------------------------------------------------------------

class TestChain:

    def test_every_fact_is_an_event(self, ledger, tmp_path) -> None:
        ids = _lifecycle(ledger, tmp_path)
        log = EventLog(ledger)
        kinds = [e.kind for e in log.events()]
        assert kinds.count("record.created") == 3
        assert kinds.count("record.sealed") == 2
        assert kinds.count("record.failed") == 1
        assert kinds.count("record.invalidated") == 1
        assert kinds.count("delivery.opened") == 2
        # intent + receipt for each delivery, then removal intent + receipt.
        assert kinds.count("delivery.event") == 6
        assert kinds.count("replay.logged") == 1

        [created, sealed] = log.for_subject(ids["sealed"])
        assert created.kind == "record.created"
        assert created.body["artifact_hash"] == ledger.get(ids["sealed"]).artifact_hash
        assert created.body["parser"]["name"] == PROVENANCE["parser"].name
        assert sealed.body == {"artifact_hash": created.body["artifact_hash"]}
        assert log.for_subject(ids["withdrawn"])[-1].body == {"reason": "superseded"}

    def test_bound_intents_and_unwritten_failures_are_events(self, ledger, tmp_path) -> None:
        from stele.ledger.delivery import DeliveryLog, PayloadConflictError

        record = ledger.seal(_pending(ledger, tmp_path).record_id)
        log = DeliveryLog(ledger)
        dispatch_id = log.open_delivery(record.record_id, LightRAGTarget("docs"))
        log.append_intent(dispatch_id, [make_chunk("a", "one")])
        with pytest.raises(PayloadConflictError):
            log.append_intent(dispatch_id, [make_chunk("a", "two")])   # refused: logs nothing
        log.append_unwritten_failure(dispatch_id, "adapter crashed")
        log.close()
        bodies = [e.body for e in EventLog(ledger).for_subject(dispatch_id)
                  if e.kind == "delivery.event"]
        # The failure wrote nothing under its own attempt; the intent's attempt
        # stays open (the chain commits to both attempt ids).
        assert [(b["event"], b["done"]) for b in bodies] == [("intent", None), ("failure", 0)]
        assert bodies[0]["attempt_id"] and bodies[1]["attempt_id"]
        assert bodies[0]["attempt_id"] != bodies[1]["attempt_id"]
        log = DeliveryLog(ledger)
        assert log.get(dispatch_id).possibly_written
        log.close()
        assert verify_ledger(ledger).ok

    def test_chain_links_and_verifies(self, ledger, tmp_path) -> None:
        _lifecycle(ledger, tmp_path)
        log = EventLog(ledger)
        events = list(log.events())
        assert [e.seq for e in events] == list(range(1, len(events) + 1))
        assert events[0].prev_hash == GENESIS
        for before, after in zip(events, events[1:]):
            assert after.prev_hash == before.hash
        assert log.head() == (events[-1].seq, events[-1].hash)

        report = verify_ledger(ledger)
        assert report.ok, (report.chain.problems, report.problems)
        assert report.chain.length == len(events)

    def test_a_refused_transition_logs_nothing(self, ledger, tmp_path) -> None:
        record = ledger.fail(_pending(ledger, tmp_path).record_id, "boom")
        before = EventLog(ledger).head()
        with pytest.raises(Exception):
            ledger.seal(record.record_id)
        with pytest.raises(Exception):
            ledger.invalidate(record.record_id, "no")
        assert EventLog(ledger).head() == before
        assert verify_ledger(ledger).ok

    def test_duplicate_run_logs_nothing(self, ledger, tmp_path) -> None:
        record = _pending(ledger, tmp_path)
        head = EventLog(ledger).head()
        out = tmp_path / "again"
        out.mkdir()
        (out / "x").write_text("x")
        with pytest.raises(Exception):
            ledger.create_pending(**PROVENANCE, run_id=record.run_id, artifact_dir=out,
                                  artifact_paths=[out / "x"])
        assert EventLog(ledger).head() == head

    def test_the_event_table_is_append_only(self, ledger, tmp_path) -> None:
        _pending(ledger, tmp_path)
        conn = sqlite3.connect(ledger.db_path)
        for statement in ("UPDATE events SET body='{}'", "DELETE FROM events"):
            with pytest.raises(sqlite3.DatabaseError, match="append-only"):
                conn.execute(statement)
        conn.close()

    def test_hash_covers_every_field(self) -> None:
        base = event_hash(1, "k", "s", "{}", "t", GENESIS)
        for changed in (
            event_hash(2, "k", "s", "{}", "t", GENESIS), event_hash(1, "x", "s", "{}", "t", GENESIS),
            event_hash(1, "k", "x", "{}", "t", GENESIS), event_hash(1, "k", "s", "[]", "t", GENESIS),
            event_hash(1, "k", "s", "{}", "x", GENESIS), event_hash(1, "k", "s", "{}", "t", "1" * 64),
        ):
            assert changed != base


# ---------------------------------------------------------------------------
# 3 — tampering
# ---------------------------------------------------------------------------

class TestTampering:

    def test_altered_event(self, ledger, tmp_path) -> None:
        ids = _lifecycle(ledger, tmp_path)
        conn = _raw(ledger)
        conn.execute(
            "UPDATE events SET body=? WHERE subject=? AND kind='record.invalidated'",
            (json.dumps({"reason": "nothing to see"}), ids["withdrawn"]),
        )
        report = verify_ledger(ledger)
        assert not report.chain.ok
        assert any("was altered" in p for p in report.chain.problems)

    def test_deleted_event(self, ledger, tmp_path) -> None:
        ids = _lifecycle(ledger, tmp_path)
        _raw(ledger).execute(
            "DELETE FROM events WHERE subject=? AND kind='record.failed'", (ids["failed"],)
        )
        problems = verify_ledger(ledger).chain.problems
        assert any("is missing" in p for p in problems)
        assert any("does not follow" in p for p in problems)

    def test_reordered_events(self, ledger, tmp_path) -> None:
        _lifecycle(ledger, tmp_path)
        conn = _raw(ledger)
        a, b = conn.execute("SELECT seq, body FROM events WHERE seq IN (2, 3) ORDER BY seq").fetchall()
        conn.execute("UPDATE events SET body=? WHERE seq=?", (b[1], a[0]))
        conn.execute("UPDATE events SET body=? WHERE seq=?", (a[1], b[0]))
        assert not verify_ledger(ledger).chain.ok

    def test_record_flipped_back_by_hand(self, ledger, tmp_path) -> None:
        """The case triggers cannot stop: artifact_records rows legitimately change."""
        ids = _lifecycle(ledger, tmp_path)
        sqlite3.connect(ledger.db_path, isolation_level=None).execute(
            "UPDATE artifact_records SET state='sealed', error=NULL WHERE record_id=?",
            (ids["withdrawn"],),
        )
        report = verify_ledger(ledger)
        assert report.chain.ok and not report.ok
        # Both halves of the edit are reported: the state and the cleared reason.
        assert report.problems == (
            f"record {ids['withdrawn']}: error is None, the chain says 'INVALIDATED: superseded'",
            f"record {ids['withdrawn']}: state is 'sealed', the chain says 'invalidated'",
        )

    def test_record_identity_edited(self, ledger, tmp_path) -> None:
        ids = _lifecycle(ledger, tmp_path)
        sqlite3.connect(ledger.db_path, isolation_level=None).execute(
            "UPDATE artifact_records SET parser_version='9.9', artifact_manifest='{}' "
            "WHERE record_id=?", (ids["sealed"],),
        )
        problems = verify_ledger(ledger).problems
        assert any("parser is" in p for p in problems)
        assert any("artifact_manifest no longer matches" in p for p in problems)

    def test_unlogged_record(self, ledger, tmp_path) -> None:
        ids = _lifecycle(ledger, tmp_path)
        conn = _raw(ledger)
        row = conn.execute("SELECT * FROM artifact_records WHERE record_id=?", (ids["sealed"],))
        cols = [d[0] for d in row.description]
        values = dict(zip(cols, row.fetchone()))
        values.update(record_id="forged", run_id="forged-run")
        conn.execute(
            f"INSERT INTO artifact_records ({', '.join(values)}) VALUES ({', '.join('?' * len(values))})",
            tuple(values.values()),
        )
        assert "record forged is in the ledger but was never logged" in verify_ledger(ledger).problems

    def test_event_body_that_is_not_json_is_reported(self, ledger, tmp_path) -> None:
        """A tampered body is a finding, never a crash of the verification."""
        _lifecycle(ledger, tmp_path)
        _raw(ledger).execute("UPDATE events SET body='not json' WHERE seq=1")
        report = verify_ledger(ledger)
        assert any("event 1 " in p and "was altered" in p for p in report.chain.problems)
        assert any(p.startswith("event 1 ") and "malformed body" in p for p in report.problems)

    def test_record_columns_that_are_not_json_are_reported(self, ledger, tmp_path) -> None:
        ids = _lifecycle(ledger, tmp_path)
        sqlite3.connect(ledger.db_path, isolation_level=None).execute(
            "UPDATE artifact_records SET parser_config='{' WHERE record_id=?", (ids["sealed"],),
        )
        sqlite3.connect(ledger.db_path, isolation_level=None).execute(
            "UPDATE artifact_records SET artifact_manifest='[1, 2]' WHERE record_id=?",
            (ids["withdrawn"],),
        )
        problems = verify_ledger(ledger).problems
        assert any(p.startswith(f"record {ids['sealed']} ") and "malformed" in p for p in problems)
        assert any(p.startswith(f"record {ids['withdrawn']} ") and "malformed" in p for p in problems)

    def test_delivery_and_replay_rows_changed(self, ledger, tmp_path) -> None:
        ids = _lifecycle(ledger, tmp_path)
        conn = _raw(ledger)
        conn.execute("DELETE FROM delivery_events WHERE event='removal_receipt'")
        conn.execute("UPDATE replays SET outcome='reproduced' WHERE replay_id=?", (ids["replay"],))
        problems = verify_ledger(ledger).problems
        assert any("its events differ from the chain" in p for p in problems)
        assert (
            f"replay {ids['replay']}: outcome is 'reproduced', the chain says 'unreplayable'"
        ) in problems

    def test_truncation_is_caught_by_an_anchor(self, ledger, tmp_path) -> None:
        _lifecycle(ledger, tmp_path)
        anchor = EventLog(ledger).head()
        conn = _raw(ledger)
        conn.execute("DELETE FROM events WHERE seq > ?", (anchor[0] - 3,))
        conn.execute("DELETE FROM replays")  # keep the tables consistent with the cut
        assert EventLog(ledger).verify().ok  # a clean prefix on its own
        report = EventLog(ledger).verify(anchor=anchor)
        assert any("truncated" in p for p in report.problems)

    def test_rewritten_history_is_caught_by_an_anchor(self, ledger, tmp_path) -> None:
        record = ledger.seal(_pending(ledger, tmp_path).record_id)
        anchor = EventLog(ledger).head()
        conn = _raw(ledger)
        conn.execute("DELETE FROM events")
        # Rebuild a self-consistent chain with a different history.
        ledger2 = open_ledger(tmp_path / "ledger.db")
        from stele.ledger.events import append_event, write_transaction

        with write_transaction(ledger2._conn):
            append_event(ledger2._conn, "record.imported", record.record_id, {"forged": True})
            append_event(ledger2._conn, "record.imported", "x", {"forged": True})
        report = EventLog(ledger2).verify(anchor=anchor)
        assert report.problems == (f"event {anchor[0]} no longer has the anchored hash",)


# ---------------------------------------------------------------------------
# 4 — migration
# ---------------------------------------------------------------------------

def test_migration_logs_existing_state(tmp_path: Path) -> None:
    db = tmp_path / "ledger.db"
    ledger = open_ledger(db)
    ids = _lifecycle(ledger, tmp_path)
    ledger.close()

    conn = _raw(ledger)
    conn.execute("DROP TABLE events")
    conn.execute("ALTER TABLE delivery_events DROP COLUMN attempt_id")  # added in v8
    conn.execute("PRAGMA user_version = 6")  # v6: payload binding (#35), no event log yet
    conn.close()

    migrated = open_ledger(db)
    log = EventLog(migrated)
    kinds = [e.kind for e in log.events()]
    assert kinds[:3] == ["record.imported"] * 3
    assert kinds.count("delivery.imported") == 2
    assert kinds.count("delivery_event.imported") == 6
    assert kinds.count("replay.imported") == 1
    imported = {e.subject: e.body for e in log.events() if e.kind == "record.imported"}
    assert imported[ids["withdrawn"]]["state"] == "invalidated"
    assert imported[ids["failed"]]["error"] == "parser crashed"
    assert verify_ledger(migrated).ok

    # New facts chain on after the imported ones.
    migrated.invalidate(ids["sealed"], "later")
    assert verify_ledger(migrated).ok
    c = sqlite3.connect(db)
    assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 8
    c.close()


# ---------------------------------------------------------------------------
# 5 — CLI
# ---------------------------------------------------------------------------

def _cli(ledger: LedgerStore, *extra: str) -> tuple[int, dict]:
    done = subprocess.run(
        [sys.executable, "-m", "stele.ledger.events", str(ledger.db_path),
         str(ledger.archive.root), *extra],
        capture_output=True, text=True,
    )
    return done.returncode, json.loads(done.stdout)


def test_cli_reports_head_and_problems(ledger, tmp_path) -> None:
    ids = _lifecycle(ledger, tmp_path)
    code, report = _cli(ledger)
    assert code == 0 and report["ok"] and report["problems"] == []
    seq, digest = EventLog(ledger).head()
    assert report["head"] == f"{seq}:{digest}"

    sqlite3.connect(ledger.db_path, isolation_level=None).execute(
        "UPDATE artifact_records SET state='sealed' WHERE record_id=?", (ids["withdrawn"],)
    )
    code, report = _cli(ledger, "--anchor", f"{seq}:{digest}")
    assert code == 1 and not report["ok"]
    assert any("the chain says 'invalidated'" in p for p in report["problems"])


# ---------------------------------------------------------------------------
# Every chained field is checked
# ---------------------------------------------------------------------------

def _with_input(ledger: LedgerStore, tmp_path: Path):
    """A SEALED record over an archived input Snapshot and Source, run by bubblewrap."""
    from stele.archive import Source
    from stele.archive.records import Snapshot, SnapshotKind

    doc = b"%PDF-1.7 the input"
    digest = ledger.archive.put_bytes(doc)
    snapshot = ledger.archive.put_snapshot(Snapshot(SnapshotKind.FILE, digest, len(doc), 1))
    out = tmp_path / f"out-{uuid.uuid4().hex[:6]}"
    out.mkdir()
    (out / "doc.txt").write_text("hello")
    record = ledger.create_pending(
        **PROVENANCE, run_id=str(uuid.uuid4()), artifact_dir=out,
        artifact_paths=[out / "doc.txt"], input_snapshot=snapshot,
        source=Source(locator="/corpus/report.pdf"), backend="bubblewrap",
    )
    return ledger.seal(record.record_id)


class TestEveryChainedField:

    @pytest.mark.parametrize(("column", "value", "problem"), [
        ("source_kind", "tree", "source_kind is 'tree', the chain says 'file'"),
        ("source_id", "f" * 64, "source_id is 'ffff"),
        ("backend", "wasmtime", "backend is 'wasmtime', the chain says 'bubblewrap'"),
        ("created_at", "2001-01-01T00:00:00+00:00", "created_at is '2001-01-01T00:00:00+00:00'"),
        ("source_path", "/elsewhere/report.pdf",
         "source_path is '/elsewhere/report.pdf', its Source says '/corpus/report.pdf'"),
    ])
    def test_record_field_edited(self, ledger, tmp_path, column, value, problem) -> None:
        record = _with_input(ledger, tmp_path)
        assert verify_ledger(ledger).ok
        _raw(ledger).execute(
            f"UPDATE artifact_records SET {column}=? WHERE record_id=?", (value, record.record_id)
        )
        report = verify_ledger(ledger)
        assert report.chain.ok and not report.ok
        assert any(problem in p for p in report.problems), report.problems

    def test_invalidation_reason_edited(self, ledger, tmp_path) -> None:
        ids = _lifecycle(ledger, tmp_path)
        _raw(ledger).execute(
            "UPDATE artifact_records SET error='INVALIDATED: routine cleanup' WHERE record_id=?",
            (ids["withdrawn"],),
        )
        assert verify_ledger(ledger).problems == (
            f"record {ids['withdrawn']}: error is 'INVALIDATED: routine cleanup', "
            "the chain says 'INVALIDATED: superseded'",
        )

    def test_failure_reason_edited(self, ledger, tmp_path) -> None:
        ids = _lifecycle(ledger, tmp_path)
        _raw(ledger).execute(
            "UPDATE artifact_records SET error='out of disk' WHERE record_id=?", (ids["failed"],)
        )
        assert verify_ledger(ledger).problems == (
            f"record {ids['failed']}: error is 'out of disk', the chain says 'parser crashed'",
        )

    @pytest.mark.parametrize(("column", "value"), [
        ("reason", "looked fine"),
        ("platform", "plan9-mips"),
        ("policy", '{"name":"lenient"}'),
    ])
    def test_replay_field_edited(self, ledger, tmp_path, column, value) -> None:
        ids = _lifecycle(ledger, tmp_path)
        _raw(ledger).execute(f"UPDATE replays SET {column}=? WHERE replay_id=?", (value, ids["replay"]))
        report = verify_ledger(ledger)
        assert any(p.startswith(f"replay {ids['replay']}: {column} is") for p in report.problems)

    def test_an_intact_lifecycle_still_verifies(self, ledger, tmp_path) -> None:
        _lifecycle(ledger, tmp_path)
        _with_input(ledger, tmp_path)
        assert verify_ledger(ledger).ok


# ---------------------------------------------------------------------------
# Verifiers never migrate what they verify
# ---------------------------------------------------------------------------

def _pre_event_log(db: Path, tmp_path: Path) -> str:
    """A version-6 ledger (no event log) whose invalidated record was then un-invalidated."""
    ledger = open_ledger(db)
    record = ledger.seal(_pending(ledger, tmp_path).record_id)
    ledger.invalidate(record.record_id, "withdrawn")
    ledger.close()
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE events")
    conn.execute("ALTER TABLE delivery_events DROP COLUMN attempt_id")
    conn.execute("PRAGMA user_version = 6")
    conn.execute("UPDATE artifact_records SET state='sealed', error=NULL")
    conn.commit()
    conn.close()
    return record.record_id


def _version(db: Path) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


class TestVerifiersDoNotMigrate:

    def test_an_older_ledger_is_refused_and_left_alone(self, tmp_path) -> None:
        from stele.archive import BlobStore
        from stele.ledger.store import LedgerSchemaError

        db = tmp_path / "ledger.db"
        _pre_event_log(db, tmp_path)
        with pytest.raises(LedgerSchemaError, match="predates the event log"):
            LedgerStore(db, BlobStore(tmp_path / "archive"), migrate=False)
        assert _version(db) == 6

    def test_a_missing_ledger_is_not_created(self, tmp_path) -> None:
        from stele.archive import BlobStore
        from stele.ledger.store import LedgerSchemaError

        with pytest.raises(LedgerSchemaError, match="no ledger"):
            LedgerStore(tmp_path / "none" / "ledger.db", BlobStore(tmp_path / "archive"),
                        migrate=False)
        assert not (tmp_path / "none").exists()

    def test_a_current_ledger_opens(self, ledger, tmp_path) -> None:
        from stele.archive import BlobStore

        _lifecycle(ledger, tmp_path)
        again = LedgerStore(ledger.db_path, BlobStore(tmp_path / "archive"), migrate=False)
        assert verify_ledger(again).ok

    def test_the_cli_refuses_instead_of_certifying(self, tmp_path) -> None:
        db = tmp_path / "ledger.db"
        _pre_event_log(db, tmp_path)
        out = subprocess.run(
            [sys.executable, "-m", "stele.ledger.events", str(db), str(tmp_path / "archive")],
            capture_output=True, text=True,
        )
        assert out.returncode == 1
        report = json.loads(out.stdout)
        assert report["ok"] is False and "predates the event log" in report["problems"][0]
        assert _version(db) == 6  # untouched: nothing was imported into a new chain

    @pytest.mark.parametrize("argv", [
        lambda db, archive, rid: ["-m", "stele.identity", "refs", db, archive, rid],
        lambda db, archive, rid: ["-m", "stele.extraction", "extract", db, archive, rid],
        lambda db, archive, rid: ["-m", "stele.verification", "list",
                                  "--ledger", db, "--archive", archive],
    ], ids=["identity", "extraction", "verification"])
    def test_other_verifiers_refuse_too(self, tmp_path, argv) -> None:
        db = tmp_path / "ledger.db"
        record_id = _pre_event_log(db, tmp_path)
        out = subprocess.run(
            [sys.executable, *argv(str(db), str(tmp_path / "archive"), record_id)],
            capture_output=True, text=True,
        )
        assert out.returncode == 1 and "cannot open the ledger" in out.stderr
        assert _version(db) == 6
