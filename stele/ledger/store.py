"""
Stele — SQLite-backed artifact ledger store.

One record per sandbox run. A record points into the evidence archive
(stele.archive, roadmap #16): its input is a Snapshot digest there, and it is
SEALED only once its artifact bundle is stored there and verified. The state
machine is documented once, in docs/ledger.md.

Schema is intentionally portable: column names and types map 1:1 to what
a future Postgres migration would look like.  The backend can be swapped
without changing the public API.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from ..archive.ingest import ingest_artifacts
from ..archive.records import Snapshot, SnapshotKind, Source, canonical_json
from ..archive.store import BlobStore, MissingObjectError
from .events import EVENTS_DDL, append_event, record_body, write_transaction
from .hashing import UnsafeFileError, build_manifest, sha256_manifest
from .models import ArtifactRecord, ArtifactState, ParserIdentity, RunConditions

# PRAGMA user_version of the current schema. 0 is a pre-#12 ledger (or an
# empty database), 2 has records only, 3 adds the delivery log (#13), 4 the
# replay log (#14), 5 the run conditions column (#30), 6 binds each delivery
# to one payload and adds the FAILED replay outcome, 7 the hash-chained event
# log, 8 gives delivery events the write attempt they belong to; see
# stele/ledger/migration.py.
SCHEMA_VERSION = 8

_DDL = (
    """
    CREATE TABLE artifact_records (
        record_id            TEXT PRIMARY KEY,
        run_id               TEXT NOT NULL UNIQUE,
        source_path          TEXT,
        source_id            TEXT,
        source_hash          TEXT,     -- input Snapshot digest in the archive
        source_kind          TEXT CHECK (source_kind IN ('file', 'tree')),
        parser_name          TEXT,
        parser_version       TEXT,
        parser_image_digest  TEXT,
        parser_module_sha256 TEXT,
        parser_config        TEXT,     -- canonical JSON object
        backend              TEXT,
        artifact_dir         TEXT NOT NULL,
        artifact_manifest    TEXT NOT NULL,   -- JSON: {rel_path: sha256}
        artifact_hash        TEXT NOT NULL,   -- tree digest in the archive
        state                TEXT NOT NULL DEFAULT 'pending'
                             CHECK (state IN ('pending', 'sealed', 'failed', 'invalidated')),
        created_at           TEXT NOT NULL,   -- ISO-8601 UTC
        finalized_at         TEXT,
        error                TEXT,
        legacy_source_hash   TEXT,            -- unverified, pre-#12 ledgers only
        run_conditions       TEXT,            -- canonical JSON: device and limits (#30)
        CHECK ((source_hash IS NULL) = (source_kind IS NULL)),
        CHECK ((parser_name IS NULL) = (parser_version IS NULL)),
        CHECK ((parser_name IS NULL) = (parser_config IS NULL))
    )
    """,
    "CREATE INDEX idx_artifact_hash ON artifact_records(artifact_hash)",
    "CREATE INDEX idx_source_hash ON artifact_records(source_hash)",
    "CREATE INDEX idx_parser ON artifact_records(parser_name, parser_version)",
)

# Schema version 4 → 5 (roadmap #30): the device and limits a run executed
# under. Existing records keep NULL: their conditions were never recorded.
RUN_CONDITIONS_DDL = (
    "ALTER TABLE artifact_records ADD COLUMN run_conditions TEXT",
)

# The delivery log (roadmap #13, stele/ledger/delivery.py). A delivery is one
# record sent to one target, identified by its dispatch_id, which writers use
# as their idempotency key. What happened to it is an append-only sequence of
# events; nothing in either table is ever updated or deleted.
DELIVERY_DDL = (
    """
    CREATE TABLE deliveries (
        dispatch_id  TEXT PRIMARY KEY,
        record_id    TEXT NOT NULL REFERENCES artifact_records(record_id),
        target_kind  TEXT NOT NULL,   -- target type name, e.g. LightRAGTarget
        target       TEXT NOT NULL,   -- canonical JSON of the target's fields
        created_at   TEXT NOT NULL,
        UNIQUE (record_id, target_kind, target)
    )
    """,
    """
    CREATE TABLE delivery_events (
        event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        dispatch_id  TEXT NOT NULL REFERENCES deliveries(dispatch_id),
        event        TEXT NOT NULL CHECK (event IN (
                         'intent', 'receipt', 'failure',
                         'removal_intent', 'removal_receipt', 'removal_failure')),
        planned      INTEGER,   -- intent: chunks about to be written
        chunks_digest TEXT,     -- intent: digest of the payload (see delivery.chunks_digest)
        done         INTEGER,   -- outcome: chunks written or removed; NULL = unknown
        error        TEXT,
        at           TEXT NOT NULL
    )
    """,
    "CREATE INDEX idx_delivery_record ON deliveries(record_id)",
    "CREATE INDEX idx_delivery_events ON delivery_events(dispatch_id, event_id)",
    """
    CREATE TRIGGER deliveries_append_only_u BEFORE UPDATE ON deliveries
    BEGIN SELECT RAISE(ABORT, 'the delivery log is append-only'); END
    """,
    """
    CREATE TRIGGER deliveries_append_only_d BEFORE DELETE ON deliveries
    BEGIN SELECT RAISE(ABORT, 'the delivery log is append-only'); END
    """,
    """
    CREATE TRIGGER delivery_events_append_only_u BEFORE UPDATE ON delivery_events
    BEGIN SELECT RAISE(ABORT, 'the delivery log is append-only'); END
    """,
    """
    CREATE TRIGGER delivery_events_append_only_d BEFORE DELETE ON delivery_events
    BEGIN SELECT RAISE(ABORT, 'the delivery log is append-only'); END
    """,
)



class DuplicateRunError(Exception):
    """Raised when the ledger already holds a record for this run_id."""


class RecordNotFoundError(Exception):
    """Raised when a record_id does not exist in the ledger."""


class InvalidStateTransitionError(Exception):
    """Raised when attempting a state transition that is not permitted."""


class MissingArtifactError(Exception):
    """Raised when an artifact is neither on disk nor in the archive at seal time."""


class ArtifactDriftError(Exception):
    """Raised when a pending artifact changed or became unsafe before sealing."""


class ProvenanceError(ValueError):
    """Raised when a record's input or parser provenance cannot be verified."""


class LedgerSchemaError(Exception):
    """Raised when the ledger database has a schema this Stele cannot use."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_record(row: sqlite3.Row) -> ArtifactRecord:
    def _dt(s: str | None) -> datetime | None:
        return datetime.fromisoformat(s) if s else None

    parser = None
    if row["parser_name"] is not None:
        parser = ParserIdentity(
            name=row["parser_name"],
            version=row["parser_version"],
            image_digest=row["parser_image_digest"],
            module_sha256=row["parser_module_sha256"],
        )
    return ArtifactRecord(
        record_id=row["record_id"],
        run_id=row["run_id"],
        source_path=row["source_path"],
        source_hash=row["source_hash"],
        source_kind=SnapshotKind(row["source_kind"]) if row["source_kind"] else None,
        source_id=row["source_id"],
        parser=parser,
        parser_config=(
            json.loads(row["parser_config"]) if row["parser_config"] is not None else None
        ),
        backend=row["backend"],
        artifact_dir=row["artifact_dir"],
        artifact_manifest=json.loads(row["artifact_manifest"]),
        artifact_hash=row["artifact_hash"],
        state=ArtifactState(row["state"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        finalized_at=_dt(row["finalized_at"]),
        error=row["error"],
        legacy_source_hash=row["legacy_source_hash"],
        run_conditions=(
            RunConditions.from_dict(json.loads(row["run_conditions"]))
            if row["run_conditions"] is not None else None
        ),
    )


def canonical_parser_config(parser_config: Mapping[str, Any]) -> str:
    """The one stored form of a parser config: canonical JSON of an object."""
    if not isinstance(parser_config, Mapping):
        raise ValueError("parser_config must be a mapping (a JSON object)")
    try:
        encoded = canonical_json(dict(parser_config)).decode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"parser_config is not JSON-serializable: {exc}") from exc
    # JSON silently turns tuples into lists and int keys into strings; the
    # stored config must be exactly the one the parser was given.
    if json.loads(encoded) != dict(parser_config):
        raise ValueError("parser_config does not survive a JSON round trip unchanged")
    return encoded


# The replay log (roadmap #14, stele/replay/engine.py): one row per replay of
# a record, append-only like the delivery log.
REPLAY_DDL = (
    """
    CREATE TABLE replays (
        replay_id            TEXT PRIMARY KEY,
        record_id            TEXT NOT NULL REFERENCES artifact_records(record_id),
        outcome              TEXT NOT NULL CHECK (outcome IN (
                                 'reproduced', 'equivalent', 'diverged', 'unreplayable')),
        reason               TEXT NOT NULL,
        replay_run_id        TEXT,     -- NULL when nothing was run
        replay_artifact_hash TEXT,     -- tree digest of the replay's output in the archive
        differences          TEXT NOT NULL,  -- JSON list of differing paths / policy findings
        policy               TEXT,     -- canonical JSON of the comparison policy used
        backend              TEXT,
        platform             TEXT NOT NULL,
        replayed_at          TEXT NOT NULL
    )
    """,
    "CREATE INDEX idx_replay_record ON replays(record_id)",
    "CREATE INDEX idx_replay_outcome ON replays(outcome)",
    """
    CREATE TRIGGER replays_append_only_u BEFORE UPDATE ON replays
    BEGIN SELECT RAISE(ABORT, 'the replay log is append-only'); END
    """,
    """
    CREATE TRIGGER replays_append_only_d BEFORE DELETE ON replays
    BEGIN SELECT RAISE(ABORT, 'the replay log is append-only'); END
    """,
)


# Version 5 → 6.
#
# A delivery is bound to the payload of its first intent: every later intent
# of the same delivery (a retry, or a concurrent dispatch that lost the race)
# must carry the same chunks_digest and planned count, so one dispatch_id can
# never name two payloads. DeliveryLog.append_intent checks this in a write
# transaction; the trigger makes the database refuse it whatever the writer.
#
# The replay log gains the FAILED outcome (the replay run was attempted but
# produced no output to compare). SQLite cannot alter a CHECK constraint, so
# the table is rebuilt; its rows are copied unchanged.
V6_DDL = (
    """
    CREATE TRIGGER delivery_events_one_payload BEFORE INSERT ON delivery_events
    WHEN NEW.event = 'intent' AND (
        NEW.chunks_digest IS NULL OR NEW.planned IS NULL OR EXISTS (
            SELECT 1 FROM delivery_events
            WHERE dispatch_id = NEW.dispatch_id AND event = 'intent'
              AND (chunks_digest IS NOT NEW.chunks_digest OR planned IS NOT NEW.planned)))
    BEGIN SELECT RAISE(ABORT, 'a delivery is bound to the payload of its first intent'); END
    """,
    """
    CREATE TABLE replays_v6 (
        replay_id            TEXT PRIMARY KEY,
        record_id            TEXT NOT NULL REFERENCES artifact_records(record_id),
        outcome              TEXT NOT NULL CHECK (outcome IN (
                                 'reproduced', 'equivalent', 'diverged', 'unreplayable',
                                 'failed')),
        reason               TEXT NOT NULL,
        replay_run_id        TEXT,
        replay_artifact_hash TEXT,
        differences          TEXT NOT NULL,
        policy               TEXT,
        backend              TEXT,
        platform             TEXT NOT NULL,
        replayed_at          TEXT NOT NULL
    )
    """,
    """
    INSERT INTO replays_v6 (replay_id, record_id, outcome, reason, replay_run_id,
        replay_artifact_hash, differences, policy, backend, platform, replayed_at)
    SELECT replay_id, record_id, outcome, reason, replay_run_id,
        replay_artifact_hash, differences, policy, backend, platform, replayed_at
    FROM replays
    """,
    "DROP TABLE replays",  # drops its indexes and append-only triggers too
    "ALTER TABLE replays_v6 RENAME TO replays",
    *REPLAY_DDL[1:],       # the indexes and append-only triggers, recreated
)


# Version 7 → 8: each intent, receipt and failure names its write attempt,
# so overlapping attempts on one delivery are told apart (see
# Delivery.possibly_written). Existing events keep NULL and are read as
# before, one attempt at a time.
V8_DDL = (
    "ALTER TABLE delivery_events ADD COLUMN attempt_id TEXT",
)


def connect(db_path: Path) -> sqlite3.Connection:
    """A connection to a ledger database.

    Autocommit: single-statement writes commit on their own, and
    multi-statement work (schema creation, migration, delivery events)
    opens its own BEGIN IMMEDIATE. Foreign keys are enforced.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the current schema. The caller holds the write transaction."""
    for statement in (*_DDL, *DELIVERY_DDL, *REPLAY_DDL, *V6_DDL, *EVENTS_DDL, *V8_DDL):
        conn.execute(statement)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def verify_archived_bundle(
    archive: BlobStore, artifact_hash: str, manifest: dict[str, str]
) -> None:
    """Prove the archive holds exactly this bundle, re-hashing every blob.

    Raises MissingObjectError if the tree or a blob is absent and
    IntegrityError if stored bytes do not match their digest.
    """
    if archive.read_tree(artifact_hash) != manifest:
        raise ArtifactDriftError(
            f"archived tree {artifact_hash[:12]}… does not match the recorded manifest"
        )
    for digest in manifest.values():
        for _ in archive.iter_verified(digest):
            pass


def archive_bundle(
    archive: BlobStore, artifact_dir: Path, manifest: dict[str, str], artifact_hash: str
) -> None:
    """Store a recorded bundle in the archive and verify it.

    When every file is still on disk, the files are stored (writing content
    that is already stored re-verifies the stored copy) and must hash to the
    recorded manifest. When files are gone from disk, the bundle must already
    be in the archive, e.g. because run_in_sandbox was given the store.
    """
    artifact_dir = Path(artifact_dir)
    artifact_paths = [artifact_dir / rel_path for rel_path in manifest]
    try:
        stored = ingest_artifacts(archive, artifact_dir, artifact_paths)
    except FileNotFoundError as exc:
        try:
            verify_archived_bundle(archive, artifact_hash, manifest)
        except MissingObjectError:
            raise MissingArtifactError(
                f"artifact file missing at seal time: {exc.filename} — and the "
                "bundle is not in the archive; cannot seal"
            ) from exc
        return
    except (UnsafeFileError, ValueError) as exc:
        raise ArtifactDriftError(
            f"artifact bundle became unsafe before sealing: {exc}"
        ) from exc

    if stored != manifest:
        raise ArtifactDriftError(
            "artifact bundle changed after pending record was created — cannot seal"
        )
    if archive.put_tree(stored) != artifact_hash:
        raise ArtifactDriftError("archived tree digest differs from the recorded artifact_hash")


class LedgerStore:
    """Append-mostly SQLite ledger for Stele artifact records.

    One instance per database file, bound to the evidence archive its records
    point into. Opening a pre-#12 ledger migrates it (see migration.py).
    Thread-safe via SQLite WAL mode and the check_same_thread=False flag
    (callers serialize writes externally or rely on SQLite's own locking).
    """

    def __init__(self, db_path: Path, archive: BlobStore, *, migrate: bool = True) -> None:
        """Open (or create) the ledger at db_path.

        With migrate=False, as verifiers open it, the ledger must already
        exist at the current schema: nothing is created or migrated, and
        LedgerSchemaError says why a ledger cannot be opened. A verifier
        must never rewrite what it verifies; migrating a ledger older than
        the event log would chain its current tables as *.imported events
        and then vouch for them.
        """
        if not isinstance(archive, BlobStore):
            raise TypeError("LedgerStore needs the BlobStore its records point into")
        self.archive = archive
        self.db_path = Path(db_path)
        if not migrate:
            if not self.db_path.is_file():
                raise LedgerSchemaError(f"no ledger at {self.db_path}")
            self._conn = connect(self.db_path)
            try:
                self._check_schema()
            except BaseException:
                self._conn.close()
                raise
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = connect(self.db_path)
        self._open_schema()

    def _check_schema(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        has_table = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='artifact_records'"
        ).fetchone() is not None
        if not has_table:
            raise LedgerSchemaError(f"{self.db_path} is not a Stele ledger")
        if version > SCHEMA_VERSION:
            raise LedgerSchemaError(
                f"ledger schema version {version} is newer than this Stele ({SCHEMA_VERSION})"
            )
        if version < SCHEMA_VERSION:
            has_events = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone() is not None
            detail = (
                "" if has_events else
                "; it predates the event log, so there is no chain to verify its "
                "tables against"
            )
            raise LedgerSchemaError(
                f"ledger schema version {version} is older than this Stele ({SCHEMA_VERSION})"
                f"{detail}. Verification never migrates a ledger; open it with "
                "LedgerStore (e.g. by recording a run) to migrate it first"
            )

    def _open_schema(self) -> None:
        from .migration import migrate_to_current

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            has_table = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='artifact_records'"
            ).fetchone() is not None
            if version > SCHEMA_VERSION:
                raise LedgerSchemaError(
                    f"ledger schema version {version} is newer than this Stele "
                    f"({SCHEMA_VERSION})"
                )
            if not has_table:
                create_schema(self._conn)
            elif version < SCHEMA_VERSION:
                migrate_to_current(self._conn, version, self.archive)
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def create_pending(
        self,
        *,
        run_id: str,
        artifact_dir: Path,
        artifact_paths: list[Path],
        parser: ParserIdentity,
        parser_config: Mapping[str, Any],
        input_snapshot: Snapshot | None = None,
        source: Source | None = None,
        backend: str | None = None,
        run_conditions: RunConditions | None = None,
    ) -> ArtifactRecord:
        """Record one run's artifact bundle as PENDING.

        artifact_paths must be non-empty, all must exist, and all must be
        under artifact_dir (i.e. they came through /stele/output).

        input_snapshot is the Snapshot of the bytes the parser read; it must
        already be in the archive, and the record's source_hash is its digest.
        There is deliberately no way to pass a source_hash directly. source
        describes where that input came from and requires input_snapshot.

        Every run gets its own record, even when its output is identical to
        another run's: the archive stores the content once, and each record
        keeps its own provenance. A second record for the same run_id raises
        DuplicateRunError.
        """
        if not artifact_paths:
            raise ValueError("cannot create a pending record with no artifacts")
        if not isinstance(parser, ParserIdentity):
            raise TypeError("parser must be a ParserIdentity")
        config_json = canonical_parser_config(parser_config)
        if run_conditions is not None and not isinstance(run_conditions, RunConditions):
            raise TypeError("run_conditions must be a RunConditions")
        conditions_json = (
            canonical_json(run_conditions.to_dict()).decode("utf-8")
            if run_conditions is not None else None
        )

        source_hash = source_kind = source_id = source_path = None
        if input_snapshot is not None:
            if not self.archive.has_snapshot(input_snapshot.digest, input_snapshot.kind):
                raise ProvenanceError(
                    f"input snapshot {input_snapshot.kind.value}/{input_snapshot.digest[:12]}… "
                    "is not in the ledger's archive"
                )
            if self.archive.get_snapshot(input_snapshot.digest, input_snapshot.kind) != input_snapshot:
                raise ProvenanceError("input snapshot differs from the archived record")
            source_hash = input_snapshot.digest
            source_kind = input_snapshot.kind.value
        if source is not None:
            if input_snapshot is None:
                raise ProvenanceError("a Source is only recorded with the Snapshot taken of it")
            source_id = self.archive.put_source(source)
            source_path = source.locator

        manifest = build_manifest(artifact_dir, artifact_paths)
        artifact_hash = sha256_manifest(manifest)

        record_id = str(uuid4())
        row = {
            "record_id": record_id, "run_id": run_id, "source_path": source_path,
            "source_id": source_id, "source_hash": source_hash, "source_kind": source_kind,
            "parser_name": parser.name, "parser_version": parser.version,
            "parser_image_digest": parser.image_digest,
            "parser_module_sha256": parser.module_sha256,
            "parser_config": config_json, "backend": backend,
            "artifact_dir": str(artifact_dir), "artifact_manifest": json.dumps(manifest),
            "artifact_hash": artifact_hash, "created_at": _now_iso(),
            "run_conditions": conditions_json,
        }
        try:
            # The record and its record.created event commit together.
            with write_transaction(self._conn):
                self._conn.execute(
                    f"INSERT INTO artifact_records ({', '.join(row)}, state) "
                    f"VALUES ({', '.join('?' * len(row))}, 'pending')",
                    tuple(row.values()),
                )
                append_event(self._conn, "record.created", record_id, record_body(row))
        except sqlite3.IntegrityError as exc:
            existing = self.get_by_run_id(run_id)
            if existing is None:
                raise
            raise DuplicateRunError(
                f"run {run_id} already has record {existing.record_id} "
                f"(state={existing.state.value})"
            ) from exc
        return self.get(record_id)

    def seal(self, record_id: str) -> ArtifactRecord:
        """Transition PENDING → SEALED.

        Stores the bundle in the archive and verifies it against the recorded
        manifest first (see archive_bundle). Drift raises ArtifactDriftError
        and a bundle that is neither on disk nor archived raises
        MissingArtifactError; the record stays PENDING in both cases.
        Sealing says nothing about delivery to downstream targets.
        """
        record = self._require(record_id)

        if record.state is not ArtifactState.PENDING:
            raise InvalidStateTransitionError(
                f"cannot seal record {record_id}: state is {record.state.value!r} "
                f"(must be 'pending')"
            )

        archive_bundle(
            self.archive, Path(record.artifact_dir), record.artifact_manifest,
            record.artifact_hash,
        )

        self._transition(
            record_id,
            from_states=(ArtifactState.PENDING,),
            action="seal",
            set_sql="state='sealed', finalized_at=?",
            params=(_now_iso(),),
            event=("record.sealed", {"artifact_hash": record.artifact_hash}),
        )
        return self.get(record_id)

    def fail(self, record_id: str, error: str) -> ArtifactRecord:
        """Transition PENDING → FAILED.

        A failed record is never sealed.  Raises InvalidStateTransitionError
        if the record is already sealed or invalidated.
        """
        record = self._require(record_id)

        if record.state is ArtifactState.SEALED:
            raise InvalidStateTransitionError(
                f"cannot fail record {record_id}: already sealed"
            )
        if record.state is ArtifactState.INVALIDATED:
            raise InvalidStateTransitionError(
                f"cannot fail record {record_id}: already invalidated"
            )

        self._transition(
            record_id,
            from_states=(ArtifactState.PENDING, ArtifactState.FAILED),
            action="fail",
            set_sql="state='failed', finalized_at=?, error=?",
            params=(_now_iso(), error),
            event=("record.failed", {"error": error}),
        )
        return self.get(record_id)

    def invalidate(self, record_id: str, reason_note: str) -> ArtifactRecord:
        """Transition PENDING or SEALED → INVALIDATED.

        INVALIDATED is a terminal state — it cannot be reversed.
        Use this when a sealed artifact is found to be stale, incorrect, or
        superseded (see replay/).
        """
        record = self._require(record_id)

        if record.state is ArtifactState.INVALIDATED:
            raise InvalidStateTransitionError(
                f"record {record_id} is already invalidated"
            )
        if record.state is ArtifactState.FAILED:
            raise InvalidStateTransitionError(
                f"cannot invalidate a failed record {record_id} — already terminal"
            )

        self._transition(
            record_id,
            from_states=(ArtifactState.PENDING, ArtifactState.SEALED),
            action="invalidate",
            set_sql="state='invalidated', finalized_at=?, error=?",
            params=(_now_iso(), f"INVALIDATED: {reason_note}"),
            event=("record.invalidated", {"reason": reason_note}),
        )
        return self.get(record_id)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get(self, record_id: str) -> ArtifactRecord:
        row = self._conn.execute(
            "SELECT * FROM artifact_records WHERE record_id=?", (record_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(record_id)
        return _row_to_record(row)

    def get_by_run_id(self, run_id: str) -> ArtifactRecord | None:
        """The record of one run, or None. A run has at most one record."""
        row = self._conn.execute(
            "SELECT * FROM artifact_records WHERE run_id=?", (run_id,)
        ).fetchone()
        return _row_to_record(row) if row else None

    def find_by_artifact_hash(self, artifact_hash: str) -> list[ArtifactRecord]:
        """Every run whose output was exactly this bundle."""
        return self._select("artifact_hash=?", (artifact_hash,))

    def find_by_source_hash(self, source_hash: str) -> list[ArtifactRecord]:
        """Every run over the input Snapshot with this digest."""
        return self._select("source_hash=?", (source_hash,))

    def find_by_parser(
        self,
        name: str,
        version: str | None = None,
        *,
        parser_config: Mapping[str, Any] | None = None,
        device: str | None = None,
    ) -> list[ArtifactRecord]:
        """Runs of a parser, optionally of one version, one exact config, and
        one device ("cpu" or "gpu"; records without run conditions never match)."""
        clauses, params = ["parser_name=?"], [name]
        if version is not None:
            clauses.append("parser_version=?")
            params.append(version)
        if parser_config is not None:
            clauses.append("parser_config=?")
            params.append(canonical_parser_config(parser_config))
        if device is not None:
            clauses.append("json_extract(run_conditions, '$.device')=?")
            params.append(RunConditions(device=device).device)
        return self._select(" AND ".join(clauses), tuple(params))

    def list_by_states(self, states: list[ArtifactState]) -> list[ArtifactRecord]:
        """Return all records whose state is in the given list, ordered by created_at."""
        placeholders = ",".join("?" * len(states))
        return self._select(f"state IN ({placeholders})", tuple(s.value for s in states))

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _select(self, where: str, params: tuple) -> list[ArtifactRecord]:
        rows = self._conn.execute(
            f"SELECT * FROM artifact_records WHERE {where} ORDER BY created_at, record_id",
            params,
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def _transition(
        self,
        record_id: str,
        *,
        from_states: tuple[ArtifactState, ...],
        action: str,
        set_sql: str,
        params: tuple,
        event: tuple[str, dict[str, Any]],
    ) -> None:
        """Apply a state change only if the record is still in from_states.

        The state check and the write happen in one UPDATE, so a concurrent
        transition on another connection cannot be overwritten (e.g. a seal
        that raced an invalidation cannot resurrect the record). The change
        and its event commit in one transaction.
        """
        placeholders = ",".join("?" * len(from_states))
        with write_transaction(self._conn):
            cur = self._conn.execute(
                f"UPDATE artifact_records SET {set_sql} "
                f"WHERE record_id=? AND state IN ({placeholders})",
                (*params, record_id, *(s.value for s in from_states)),
            )
            if cur.rowcount == 1:
                append_event(self._conn, event[0], record_id, event[1])
        if cur.rowcount != 1:
            current = self._require(record_id).state.value
            raise InvalidStateTransitionError(
                f"cannot {action} record {record_id}: state changed to "
                f"{current!r} concurrently"
            )

    def _require(self, record_id: str) -> ArtifactRecord:
        try:
            return self.get(record_id)
        except RecordNotFoundError:
            raise RecordNotFoundError(f"no record with id={record_id}")
