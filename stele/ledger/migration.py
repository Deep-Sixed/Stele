"""
Migrating older ledgers to the current schema.

Version 2 (#12) → 3 (#13) adds the delivery log tables, 3 → 4 (#14) the
replay log, 4 → 5 (#30) the run_conditions column, NULL for existing
records (their device and limits were never recorded), 5 → 6 binds each
delivery to the payload of its first intent and admits the FAILED replay
outcome, 6 → 7 adds the hash-chained event log, and 7 → 8 gives delivery
events the attempt_id of their write attempt (NULL for existing events).
None of these changes existing rows; reaching 7 logs every existing record,
delivery and replay as an *.imported event, so the chain covers the ledger
from the migration on.

Version 0 is a pre-#12 ledger.

A version-0 ledger allowed one record per artifact_hash, meant "delivered
downstream" by COMMITTED, trusted a caller-supplied source_hash, and stored no
parser identity. Migration runs once, inside the write transaction that
LedgerStore opens, so it either completes or leaves the database untouched:

- pending, failed and invalidated records keep their state.
- committed records become SEALED if their bundle can be stored in the
  archive and verified against the recorded manifest (from disk, or from the
  archive if it already holds the bundle). A committed record whose bundle
  cannot be verified becomes INVALIDATED with the reason: the ledger can no
  longer vouch for that content. The reason says whether the evidence was
  missing (neither on disk nor archived) or mismatched; only a record with
  missing evidence can be restored later, by ledger.reverify, from files that
  still hash as recorded.
- source_hash is kept only when it is the digest of exactly one Snapshot in
  the archive. Otherwise it moves to legacy_source_hash, which is unverified
  audit data and never used as provenance.
- source_path is kept, and recorded as a Source in the archive.
- parser identity and config are unknown and stay NULL.
- Two records for one run_id cannot be represented and stop the migration.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..archive.records import SnapshotKind, Source
from ..archive.store import ArchiveError, BlobStore
from .events import EVENTS_DDL, backfill
from .hashing import UnsafeFileError
from .store import (
    DELIVERY_DDL,
    REPLAY_DDL,
    RUN_CONDITIONS_DDL,
    V6_DDL,
    V8_DDL,
    ArtifactDriftError,
    MissingArtifactError,
    archive_bundle,
    create_schema,
)


# Why a committed record was invalidated by migration (prefix of its error).
EVIDENCE_MISSING = "evidence missing"
EVIDENCE_MISMATCHED = "evidence mismatched"


class LedgerMigrationError(Exception):
    """Raised when an existing ledger cannot be migrated; nothing is changed."""


_V0_COLUMNS = {
    "record_id", "run_id", "source_path", "source_hash", "artifact_dir",
    "artifact_manifest", "artifact_hash", "state", "created_at", "finalized_at", "error",
}


def migrate_to_current(conn: sqlite3.Connection, version: int, archive: BlobStore) -> None:
    """Migrate in place. The caller holds a BEGIN IMMEDIATE transaction."""
    # version → DDL that reaches version + 1. Reaching 7 creates the event
    # log, which then records the state found as *.imported events.
    additive = {
        2: DELIVERY_DDL, 3: REPLAY_DDL, 4: RUN_CONDITIONS_DDL, 5: V6_DDL, 6: EVENTS_DDL,
        7: V8_DDL,
    }
    if version in additive:
        creates_event_log = version < 7
        while version in additive:
            for statement in additive[version]:
                conn.execute(statement)
            version += 1
        if creates_event_log:
            # Only once: a ledger that already has the event log is covered.
            backfill(conn)
        conn.execute(f"PRAGMA user_version = {version}")
        return
    if version != 0:
        raise LedgerMigrationError(f"no migration from ledger schema version {version}")

    columns = {row[1] for row in conn.execute("PRAGMA table_info(artifact_records)")}
    if columns != _V0_COLUMNS:
        raise LedgerMigrationError(
            f"artifact_records has unexpected columns {sorted(columns)}; not a v0 ledger"
        )

    rows = conn.execute("SELECT * FROM artifact_records ORDER BY created_at").fetchall()
    duplicates = conn.execute(
        "SELECT run_id FROM artifact_records GROUP BY run_id HAVING COUNT(*) > 1"
    ).fetchall()
    if duplicates:
        raise LedgerMigrationError(
            "these runs have more than one record, which the ledger no longer "
            f"allows: {sorted(r[0] for r in duplicates)}"
        )

    migrated = [_migrate_row(dict(zip(row.keys(), row)), archive) for row in rows]

    conn.execute("DROP INDEX IF EXISTS idx_artifact_hash")
    conn.execute("DROP INDEX IF EXISTS idx_run_id")
    conn.execute("ALTER TABLE artifact_records RENAME TO artifact_records_v0")
    create_schema(conn)
    for row in migrated:
        names = ", ".join(row)
        conn.execute(
            f"INSERT INTO artifact_records ({names}) VALUES ({', '.join('?' * len(row))})",
            tuple(row.values()),
        )
    conn.execute("DROP TABLE artifact_records_v0")
    backfill(conn)


def _migrate_row(old: dict, archive: BlobStore) -> dict:
    row = {
        "record_id": old["record_id"],
        "run_id": old["run_id"],
        "source_path": old["source_path"],
        "source_id": None,
        "source_hash": None,
        "source_kind": None,
        "legacy_source_hash": None,
        "artifact_dir": old["artifact_dir"],
        "artifact_manifest": old["artifact_manifest"],
        "artifact_hash": old["artifact_hash"],
        "state": old["state"],
        "created_at": old["created_at"],
        "finalized_at": old["finalized_at"],
        "error": old["error"],
    }

    if old["source_path"]:
        row["source_id"] = archive.put_source(Source(locator=old["source_path"]))

    if old["source_hash"]:
        kinds = [
            kind for kind in SnapshotKind
            if _is_snapshot(archive, old["source_hash"], kind)
        ]
        if len(kinds) == 1:
            row["source_hash"], row["source_kind"] = old["source_hash"], kinds[0].value
        else:
            row["legacy_source_hash"] = old["source_hash"]

    if old["state"] == "committed":
        try:
            archive_bundle(
                archive, Path(old["artifact_dir"]), json.loads(old["artifact_manifest"]),
                old["artifact_hash"],
            )
        except (
            ArchiveError, ArtifactDriftError, MissingArtifactError, UnsafeFileError,
            OSError, ValueError,
        ) as exc:
            row["state"] = "invalidated"
            row["finalized_at"] = datetime.now(timezone.utc).isoformat()
            # Missing evidence (neither on disk nor archived) can be restored
            # by an operator who still has the files (ledger.reverify);
            # anything else means the content is not what was recorded.
            kind = EVIDENCE_MISSING if isinstance(exc, MissingArtifactError) else EVIDENCE_MISMATCHED
            row["error"] = (
                f"INVALIDATED: migration — {kind}: committed bundle could not be "
                f"archived and verified: {exc!r}"
            )
        else:
            row["state"] = "sealed"
    elif old["state"] not in ("pending", "failed", "invalidated"):
        raise LedgerMigrationError(
            f"record {old['record_id']} has unknown state {old['state']!r}"
        )
    return row


def _is_snapshot(archive: BlobStore, digest: str, kind: SnapshotKind) -> bool:
    try:
        return archive.has_snapshot(digest, kind)
    except ArchiveError:  # not a well-formed digest
        return False
