"""
Resolve a reference against the evidence, never against the caller's word.

    SourceRef        the Source record is in the archive and hashes to source_id
    SnapshotRef      the Snapshot record is in the archive, and every blob it
                     covers re-hashes (the file, or the tree and each member)
    RecordRef        the record exists, is SEALED (INVALIDATED only when
                     allowed), has that artifact_hash, and the archived tree
                     object for it lists exactly the record's manifest
    ObservationRef   the Observation rebuilt from the record (its Source,
                     input Snapshot and creation time) has that digest, and its
                     Snapshot resolves. An observation stays true whatever
                     happened to the run afterwards, so any record state will do.
    AnchorRef        the record resolves as above, then the extraction
                     resolver reads the anchored span from archive-verified
                     bytes (stele.extraction.resolver)
    EventRef         the event is in the chain at that seq with that hash, and
                     the whole chain verifies (EventLog.verify with the event
                     as anchor), so the event still commits to all history
                     before it

resolve() returns what the reference names (a Source, Snapshot,
ArtifactRecord, Observation, the anchored text, or an Event) or raises
ResolveError; check() returns the problems as a list instead.
"""
from __future__ import annotations

from typing import Any

from ..archive.store import ArchiveError
from ..contracts.adapter import SealedBundle
from ..extraction.resolver import BundleResolver, ResolutionError
from ..ledger.events import EventLog
from ..ledger.models import ArtifactRecord, ArtifactState
from ..ledger.store import LedgerStore, RecordNotFoundError
from .refs import (
    AnchorRef, EventRef, ObservationRef, RecordRef, Ref, RefFormatError, SnapshotRef, SourceRef,
    observation_of, parse_ref,
)


class ResolveError(ValueError):
    """A reference that does not resolve against the evidence."""


class IdentityResolver:
    """Resolves Stele references against one ledger and its evidence archive."""

    def __init__(self, ledger: LedgerStore, *, allow_invalidated: bool = False) -> None:
        self.ledger = ledger
        self.archive = ledger.archive
        self.allow_invalidated = allow_invalidated

    def resolve(self, ref: Ref | str) -> Any:
        if isinstance(ref, str):
            try:
                ref = parse_ref(ref)
            except RefFormatError as exc:
                raise ResolveError(str(exc)) from exc
        method = getattr(self, f"_resolve_{type(ref).__name__}", None)
        if method is None:
            raise ResolveError(f"not a Stele reference: {ref!r}")
        return method(ref)

    def check(self, ref: Ref | str) -> list[str]:
        """[] when ref resolves, else the reason it does not."""
        try:
            self.resolve(ref)
        except ResolveError as exc:
            return [str(exc)]
        return []

    # -- one per reference type ----------------------------------------------

    def _resolve_SourceRef(self, ref: SourceRef) -> Any:
        try:
            return self.archive.get_source(ref.source_id)
        except ArchiveError as exc:
            raise ResolveError(f"{ref}: {exc}") from exc

    def _resolve_SnapshotRef(self, ref: SnapshotRef) -> Any:
        try:
            snapshot = self.archive.get_snapshot(ref.digest, ref.kind)
            self.archive.verify_snapshot(snapshot)
        except ArchiveError as exc:
            raise ResolveError(f"{ref}: {exc}") from exc
        return snapshot

    def _record(self, record_id: str, ref: Any) -> ArtifactRecord:
        try:
            return self.ledger.get(record_id)
        except RecordNotFoundError:
            raise ResolveError(f"{ref}: no record {record_id} in the ledger") from None

    def _resolve_RecordRef(self, ref: RecordRef) -> ArtifactRecord:
        record = self._record(ref.record_id, ref)
        allowed = {ArtifactState.SEALED}
        if self.allow_invalidated:
            allowed.add(ArtifactState.INVALIDATED)
        if record.state not in allowed:
            raise ResolveError(f"{ref}: record is {record.state.value}; only sealed evidence resolves")
        if record.artifact_hash != ref.artifact_hash:
            raise ResolveError(
                f"{ref}: the record sealed artifact_hash {record.artifact_hash}, not {ref.artifact_hash}"
            )
        try:
            tree = self.archive.read_tree(record.artifact_hash)
        except ArchiveError as exc:
            raise ResolveError(f"{ref}: sealed output tree unreadable: {exc}") from exc
        if tree != dict(record.artifact_manifest):
            raise ResolveError(f"{ref}: the archived output tree does not list the record's manifest")
        return record

    def _resolve_ObservationRef(self, ref: ObservationRef) -> Any:
        record = self._record(ref.record_id, ref)
        try:
            observation = observation_of(record)
        except ValueError as exc:
            raise ResolveError(f"{ref}: {exc}") from exc
        self._check_observed_facts(ref, record)
        if observation.digest != ref.digest:
            raise ResolveError(
                f"{ref}: the record's observation has digest {observation.digest}, not {ref.digest}"
            )
        self._resolve_SnapshotRef(observation.snapshot)
        if observation.source_id is not None:
            self._resolve_SourceRef(SourceRef(observation.source_id))
        return observation

    def _check_observed_facts(self, ref: ObservationRef, record: ArtifactRecord) -> None:
        """The record's row must say what its creation event in the chain says.

        An observation is rebuilt from the row, so it must not resolve from a
        row edited after the fact. source_hash, source_kind and source_id are
        chained with every record; created_at is chained for records created
        since the event log first committed to it (older records' creation
        time rests on the row alone).
        """
        log = EventLog(self.ledger)
        try:
            created = next(
                (e for e in log.for_subject(record.record_id)
                 if e.kind in ("record.created", "record.imported")),
                None,
            )
        finally:
            log.close()
        if created is None:
            raise ResolveError(f"{ref}: record {record.record_id} has no creation event in the chain")
        body = created.body
        facts = {
            "source_hash": record.source_hash,
            "source_kind": record.source_kind.value if record.source_kind else None,
            "source_id": record.source_id,
        }
        if "created_at" in body:
            facts["created_at"] = record.created_at.isoformat()
        for key, value in facts.items():
            if body.get(key) != value:
                raise ResolveError(
                    f"{ref}: the record's {key} is {value!r}, but its creation event "
                    f"(event {created.seq}) says {body.get(key)!r}"
                )

    def _resolve_AnchorRef(self, ref: AnchorRef) -> str:
        record = self._resolve_RecordRef(ref.record)
        try:
            return BundleResolver(SealedBundle.from_record(record, self.archive)).resolve(ref.anchor)
        except ResolutionError as exc:
            raise ResolveError(f"{ref}: {exc}") from exc

    def _resolve_EventRef(self, ref: EventRef) -> Any:
        log = EventLog(self.ledger)
        try:
            report = log.verify(anchor=(ref.seq, ref.hash))
            if not report.ok:
                raise ResolveError(f"{ref}: {'; '.join(report.problems[:3])}")
            event = next(log.events(after=ref.seq - 1), None)
        finally:
            log.close()
        if event is None or (event.seq, event.hash) != (ref.seq, ref.hash):
            raise ResolveError(f"{ref}: no such event in the chain")
        return event
