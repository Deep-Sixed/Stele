"""
Restoring a record the migration invalidated for missing evidence (#39).

Migrating a pre-#12 ledger invalidates a committed record whose bundle can be
neither archived from disk nor found in the archive. Often the evidence was
never archived (the archive did not exist yet) and the files are still
somewhere: invalidating is safe, but it is permanent, so such a record could
never be sealed again.

reverify_record() is the explicit operator step that restores it. It never
happens during migration, and it is narrow:

- the record must be INVALIDATED *by the migration*, for *missing* evidence.
  Its error says so, and its event history is exactly the migration's import
  of it: an invalidation by an operator or a divergent replay has a
  record.invalidated event (or another error) and stays invalidated, as does
  a record whose evidence was found to be different (mismatched);
- it must have no deliveries, since invalidation withdraws delivered data;
- the files in the given directory are hashed against the recorded manifest
  and archived exactly as sealing does. Any difference, or a missing file,
  leaves the record INVALIDATED.

On success the record is SEALED again, its error cleared, and a
`record.reverified` event is appended to the hash chain in the same
transaction; the earlier events stay, so the history reads: imported as
invalidated, reverified.

    python -m stele.ledger.reverify <ledger.db> <archive-dir> <record_id> --artifact-dir DIR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..archive import BlobStore
from .events import EventLog
from .migration import EVIDENCE_MISSING
from .models import ArtifactRecord, ArtifactState
from .store import (
    ArtifactDriftError,
    InvalidStateTransitionError,
    LedgerStore,
    MissingArtifactError,
    RecordNotFoundError,
    archive_bundle,
)

_MIGRATION_PREFIX = "INVALIDATED: migration — "


class ReverifyRefused(Exception):
    """The record cannot be restored; nothing was changed."""


def reverifiable_problem(ledger: LedgerStore, record: ArtifactRecord) -> str | None:
    """Why the record may not be reverified; None if it may."""
    if record.state is not ArtifactState.INVALIDATED:
        return f"the record is {record.state.value}, not invalidated"
    error = record.error or ""
    if not error.startswith(f"{_MIGRATION_PREFIX}{EVIDENCE_MISSING}:"):
        return "it was not invalidated by the migration for missing evidence"
    log = EventLog(ledger)
    try:
        events = log.for_subject(record.record_id)
    finally:
        log.close()
    if [e.kind for e in events] != ["record.imported"] or events[0].body.get("state") != "invalidated":
        return "its history is not just the migration's import, so it was invalidated for another reason"
    deliveries = ledger._conn.execute(
        "SELECT COUNT(*) FROM deliveries WHERE record_id=?", (record.record_id,)
    ).fetchone()[0]
    if deliveries:
        return "it has deliveries, and invalidating withdrew the delivered data"
    return None


def reverify_record(
    ledger: LedgerStore, record_id: str, artifact_dir: Path | str | None = None
) -> ArtifactRecord:
    """Seal an invalidated record again from files that hash as recorded.

    artifact_dir defaults to the directory the record names. Raises
    ReverifyRefused (nothing changed) for a record that may not be restored
    or whose files do not match.
    """
    try:
        record = ledger.get(record_id)
    except RecordNotFoundError:
        raise ReverifyRefused(f"no record {record_id}") from None
    problem = reverifiable_problem(ledger, record)
    if problem is not None:
        raise ReverifyRefused(f"record {record_id} cannot be reverified: {problem}")
    directory = Path(artifact_dir) if artifact_dir is not None else Path(record.artifact_dir)
    try:
        archive_bundle(ledger.archive, directory, record.artifact_manifest, record.artifact_hash)
    except (MissingArtifactError, ArtifactDriftError) as exc:
        raise ReverifyRefused(
            f"record {record_id}: the files in {directory} are not the recorded bundle: {exc}"
        ) from exc
    try:
        ledger._transition(
            record_id,
            from_states=(ArtifactState.INVALIDATED,),
            action="reverify",
            set_sql="state='sealed', finalized_at=?, error=NULL",
            params=(_now_iso(),),
            event=("record.reverified", {
                "artifact_hash": record.artifact_hash,
                "previous_error": record.error,
            }),
        )
    except InvalidStateTransitionError as exc:
        raise ReverifyRefused(str(exc)) from exc
    return ledger.get(record_id)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m stele.ledger.reverify",
        description="Seal a migration-invalidated record again from files that hash as recorded.",
    )
    ap.add_argument("ledger", type=Path)
    ap.add_argument("archive", type=Path)
    ap.add_argument("record_id")
    ap.add_argument("--artifact-dir", type=Path, default=None,
                    help="where the record's files are now (default: the directory it names)")
    args = ap.parse_args(argv)
    ledger = LedgerStore(args.ledger, BlobStore(args.archive))
    try:
        record = reverify_record(ledger, args.record_id, args.artifact_dir)
    except ReverifyRefused as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        ledger.close()
    print(f"record {record.record_id} is sealed again (artifact_hash {record.artifact_hash})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
