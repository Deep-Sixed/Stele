"""
python -m stele.identity resolve <ledger.db> <archive-dir> <ref>... [--allow-invalidated]
    Resolve each reference against the evidence and print one JSON result per
    reference; exit 1 if any does not resolve.
python -m stele.identity refs <ledger.db> <archive-dir> <record_id>
    Print the record, observation and event references of one record.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from ..archive import BlobStore, Snapshot, Source
from ..ledger.events import EventLog
from ..ledger.models import ArtifactRecord
from ..ledger.store import LedgerSchemaError, LedgerStore, RecordNotFoundError
from .refs import Observation, event_ref, observation_of, record_ref
from .resolver import IdentityResolver, ResolveError


def _summary(value: Any) -> Any:
    """A small JSON description of what a reference resolved to."""
    if isinstance(value, str):
        return {"text": value}
    if isinstance(value, ArtifactRecord):
        return {"record_id": value.record_id, "state": value.state.value,
                "artifact_hash": value.artifact_hash, "files": len(value.artifact_manifest)}
    if isinstance(value, Observation):
        return json.loads(value.to_canonical())
    if isinstance(value, (Source, Snapshot)):
        return json.loads(value.to_canonical())
    if dataclasses.is_dataclass(value):
        return {k: v for k, v in dataclasses.asdict(value).items()}
    return repr(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stele.identity",
                                 description="Stele references: resolve and list them.")
    sub = ap.add_subparsers(dest="command", required=True)
    re_ = sub.add_parser("resolve", help="resolve references against the evidence")
    re_.add_argument("ledger", type=Path)
    re_.add_argument("archive", type=Path)
    re_.add_argument("refs", nargs="+")
    re_.add_argument("--allow-invalidated", action="store_true")
    rf = sub.add_parser("refs", help="the references of one record")
    rf.add_argument("ledger", type=Path)
    rf.add_argument("archive", type=Path)
    rf.add_argument("record_id")
    args = ap.parse_args(argv)

    try:
        # Read-only: a verifier never migrates the ledger it reads.
        ledger = LedgerStore(args.ledger, BlobStore(args.archive), migrate=False)
    except LedgerSchemaError as exc:
        print(f"cannot open the ledger: {exc}", file=sys.stderr)
        return 1
    try:
        if args.command == "refs":
            try:
                record = ledger.get(args.record_id)
            except RecordNotFoundError:
                print(f"no record {args.record_id}", file=sys.stderr)
                return 1
            out: dict[str, Any] = {"record": str(record_ref(record))}
            if record.source_hash is not None and record.source_kind is not None:
                observation = observation_of(record)
                out["observation"] = str(observation.ref)
                out["snapshot"] = str(observation.snapshot)
            log = EventLog(ledger)
            try:
                out["events"] = [str(event_ref(e)) for e in log.for_subject(record.record_id)]
            finally:
                log.close()
            print(json.dumps(out, indent=2))
            return 0
        resolver = IdentityResolver(ledger, allow_invalidated=args.allow_invalidated)
        failed = False
        for ref in args.refs:
            try:
                value = resolver.resolve(ref)
            except ResolveError as exc:
                failed = True
                print(json.dumps({"ref": ref, "ok": False, "problem": str(exc)}))
            else:
                print(json.dumps({"ref": ref, "ok": True, "value": _summary(value)},
                                 ensure_ascii=False, default=str))
        return 1 if failed else 0
    finally:
        ledger.close()


if __name__ == "__main__":
    sys.exit(main())
