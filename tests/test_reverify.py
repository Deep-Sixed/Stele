"""
Restoring records the migration invalidated for missing evidence (#39).

  1. The migration says *why* it invalidated a committed record: missing
     evidence (neither on disk nor archived) or mismatched evidence.
  2. reverify_record() seals a missing-evidence record again from files that
     hash as recorded, logs record.reverified, and the ledger still verifies.
  3. It refuses everything else and changes nothing: mismatched evidence,
     files that differ, records invalidated by an operator or a replay,
     records with deliveries, records that are not invalidated.
  4. The CLI does the same.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from pathlib import Path

import pytest

from stele.ledger.events import EventLog, verify_ledger
from stele.ledger.hashing import build_manifest, sha256_manifest
from stele.ledger.models import ArtifactState, ParserIdentity
from stele.ledger.reverify import ReverifyRefused, main as cli, reverify_record
from tests.ledger_helpers import open_ledger
from tests.test_ledger_provenance import _V0_DDL


@pytest.fixture
def v0(tmp_path: Path):
    """A pre-#12 ledger; add(...) returns (record_id, output directory)."""
    db = tmp_path / "ledger.db"
    conn = sqlite3.connect(db)
    conn.executescript(_V0_DDL)
    counter = iter(range(100))

    def add(state: str, *, keep_files: bool = True, error: str | None = None,
            content: str | None = None) -> tuple[str, Path]:
        n = next(counter)
        out = tmp_path / f"out-{n}"
        out.mkdir()
        (out / "r.json").write_text(content or f"output {n}")
        (out / "s.txt").write_text(f"second file {n}")
        manifest = build_manifest(out, [out / "r.json", out / "s.txt"])
        if not keep_files:
            shutil.rmtree(out)
        record_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO artifact_records VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (record_id, str(uuid.uuid4()), None, None, str(out), json.dumps(manifest),
             sha256_manifest(manifest), state, f"2026-06-26T00:00:{n:02d}+00:00",
             None if state == "pending" else "2026-06-26T01:00:00+00:00", error),
        )
        conn.commit()
        return record_id, out

    yield db, add
    conn.close()


def _restore(out: Path, n: int) -> None:
    """Put back the files the v0 fixture wrote for output n."""
    out.mkdir(exist_ok=True)
    (out / "r.json").write_text(f"output {n}")
    (out / "s.txt").write_text(f"second file {n}")


def _kinds(ledger, record_id: str) -> list[str]:
    log = EventLog(ledger)
    try:
        return [e.kind for e in log.for_subject(record_id)]
    finally:
        log.close()


# ---------------------------------------------------------------------------
# 1 — the migration says why
# ---------------------------------------------------------------------------

def test_migration_distinguishes_missing_from_mismatched_evidence(v0, tmp_path) -> None:
    db, add = v0
    missing, _ = add("committed", keep_files=False)
    mismatched, out = add("committed")
    (out / "r.json").write_text("not what was recorded")

    ledger = open_ledger(db)
    assert ledger.get(missing).error.startswith("INVALIDATED: migration — evidence missing:")
    assert ledger.get(mismatched).error.startswith("INVALIDATED: migration — evidence mismatched:")


# ---------------------------------------------------------------------------
# 2 — restoring
# ---------------------------------------------------------------------------

def test_files_that_hash_as_recorded_seal_the_record_again(v0, tmp_path) -> None:
    db, add = v0
    record_id, out = add("committed", keep_files=False)
    ledger = open_ledger(db)
    before = ledger.get(record_id)
    assert before.state is ArtifactState.INVALIDATED

    _restore(out, 0)
    record = reverify_record(ledger, record_id)

    assert record.state is ArtifactState.SEALED
    assert record.error is None
    assert record.artifact_hash == before.artifact_hash
    assert ledger.archive.read_tree(record.artifact_hash) == record.artifact_manifest
    assert _kinds(ledger, record_id) == ["record.imported", "record.reverified"]
    report = verify_ledger(ledger)
    assert report.ok, (report.chain.problems, report.problems)


def test_files_can_be_given_from_another_directory(v0, tmp_path) -> None:
    db, add = v0
    record_id, _ = add("committed", keep_files=False)
    elsewhere = tmp_path / "recovered"
    _restore(elsewhere, 0)
    ledger = open_ledger(db)

    assert reverify_record(ledger, record_id, elsewhere).state is ArtifactState.SEALED
    assert verify_ledger(ledger).ok


def test_a_restored_record_is_an_ordinary_sealed_record(v0, tmp_path) -> None:
    db, add = v0
    record_id, out = add("committed", keep_files=False)
    ledger = open_ledger(db)
    _restore(out, 0)
    reverify_record(ledger, record_id)
    shutil.rmtree(out)                       # the archive alone vouches for it now
    reopened = open_ledger(db)
    assert reopened.get(record_id).state is ArtifactState.SEALED
    assert verify_ledger(reopened).ok
    with pytest.raises(ReverifyRefused, match="not invalidated"):
        reverify_record(reopened, record_id)


# ---------------------------------------------------------------------------
# 3 — everything else is refused, and nothing changes
# ---------------------------------------------------------------------------

def _assert_untouched(ledger, record_id: str, events: list[str]) -> None:
    assert ledger.get(record_id).state is ArtifactState.INVALIDATED
    assert _kinds(ledger, record_id) == events
    assert verify_ledger(ledger).ok


def test_mismatched_evidence_is_never_restored(v0, tmp_path) -> None:
    db, add = v0
    record_id, out = add("committed")
    (out / "r.json").write_text("not what was recorded")
    ledger = open_ledger(db)
    _restore(out, 0)                          # even with the right files back
    with pytest.raises(ReverifyRefused, match="not invalidated by the migration for missing"):
        reverify_record(ledger, record_id)
    _assert_untouched(ledger, record_id, ["record.imported"])


@pytest.mark.parametrize("damage", ["changed", "missing_file", "extra_only"])
def test_files_that_differ_leave_the_record_invalidated(v0, tmp_path, damage) -> None:
    db, add = v0
    record_id, out = add("committed", keep_files=False)
    ledger = open_ledger(db)
    _restore(out, 0)
    if damage == "changed":
        (out / "s.txt").write_text("someone edited this")
    elif damage == "missing_file":
        (out / "s.txt").unlink()
    else:
        (out / "r.json").unlink()
        (out / "s.txt").unlink()
        (out / "other.txt").write_text("not in the manifest")
    with pytest.raises(ReverifyRefused, match="not the recorded bundle"):
        reverify_record(ledger, record_id)
    _assert_untouched(ledger, record_id, ["record.imported"])


def test_a_record_invalidated_by_an_operator_stays_invalidated(v0, tmp_path) -> None:
    """Even with the migration's own words as the note, the history gives it away."""
    db, add = v0
    ledger = open_ledger(db)
    out = tmp_path / "fresh"
    out.mkdir()
    (out / "a.txt").write_text("a")
    pending = ledger.create_pending(
        run_id=str(uuid.uuid4()), artifact_dir=out, artifact_paths=[out / "a.txt"],
        parser=ParserIdentity("p", "1"), parser_config={},
    )
    ledger.seal(pending.record_id)
    ledger.invalidate(pending.record_id, "migration — evidence missing: by hand")
    with pytest.raises(ReverifyRefused, match="invalidated for another reason"):
        reverify_record(ledger, pending.record_id, out)
    assert ledger.get(pending.record_id).state is ArtifactState.INVALIDATED


def test_the_pre_reverify_error_text_is_not_trusted(v0, tmp_path) -> None:
    """Invalidated by an earlier migration that did not say why: stays put."""
    db, add = v0
    record_id, out = add(
        "invalidated",
        error="INVALIDATED: migration — committed bundle could not be archived and verified: "
              "MissingArtifactError('x')",
    )
    ledger = open_ledger(db)
    _restore(out, 0)
    with pytest.raises(ReverifyRefused, match="not invalidated by the migration for missing"):
        reverify_record(ledger, record_id)
    _assert_untouched(ledger, record_id, ["record.imported"])


def test_a_record_with_deliveries_is_refused(v0, tmp_path) -> None:
    db, add = v0
    record_id, out = add("committed", keep_files=False)
    ledger = open_ledger(db)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO deliveries (dispatch_id, record_id, target_kind, target, created_at) "
        "VALUES ('d1', ?, 'T', '{}', '2026-06-26T02:00:00+00:00')", (record_id,),
    )
    conn.commit()
    conn.close()
    _restore(out, 0)
    with pytest.raises(ReverifyRefused, match="has deliveries"):
        reverify_record(ledger, record_id)
    assert ledger.get(record_id).state is ArtifactState.INVALIDATED


@pytest.mark.parametrize("state", ["committed", "pending", "failed"])
def test_only_invalidated_records_are_reverified(v0, tmp_path, state) -> None:
    db, add = v0
    record_id, _ = add(state, error="x" if state == "failed" else None)
    ledger = open_ledger(db)
    with pytest.raises(ReverifyRefused, match="not invalidated"):
        reverify_record(ledger, record_id)


def test_unknown_record(v0) -> None:
    ledger = open_ledger(v0[0])
    with pytest.raises(ReverifyRefused, match="no record"):
        reverify_record(ledger, str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# 4 — CLI
# ---------------------------------------------------------------------------

def test_cli(v0, tmp_path, capsys) -> None:
    db, add = v0
    record_id, out = add("committed", keep_files=False)
    ledger = open_ledger(db)
    archive = str(ledger.archive.root)
    ledger.close()

    assert cli([str(db), archive, record_id]) == 1            # files are still gone
    assert "not the recorded bundle" in capsys.readouterr().err
    _restore(out, 0)
    assert cli([str(db), archive, record_id, "--artifact-dir", str(out)]) == 0
    assert "sealed again" in capsys.readouterr().out
    assert cli([str(db), archive, record_id]) == 1            # already sealed
    assert "not invalidated" in capsys.readouterr().err
