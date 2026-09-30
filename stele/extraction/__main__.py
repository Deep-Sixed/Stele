"""
python -m stele.extraction extract <ledger.db> <archive-dir> <record_id> [-o FILE]
    Normalize a sealed record and write its canonical Extraction (stdout by default).
python -m stele.extraction verify <ledger.db> <archive-dir> <extraction.json> [--allow-invalidated]
    Check every unit of an Extraction against the sealed evidence; exit 1 on any problem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from ..archive import BlobStore
from ..contracts.adapter import SealedBundle
from ..ledger.models import ArtifactState
from ..ledger.store import LedgerSchemaError, LedgerStore
from .contract import Extraction, ExtractionFormatError
from .normalizers import normalize
from .resolver import ResolutionError, Resolver


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stele.extraction",
                                 description="Canonical extractions of sealed ledger records.")
    sub = ap.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract", help="normalize a sealed record")
    ex.add_argument("ledger", type=Path)
    ex.add_argument("archive", type=Path)
    ex.add_argument("record_id")
    ex.add_argument("-o", "--output", type=Path, default=None)
    ve = sub.add_parser("verify", help="check an extraction against the evidence")
    ve.add_argument("ledger", type=Path)
    ve.add_argument("archive", type=Path)
    ve.add_argument("extraction", type=Path)
    ve.add_argument("--allow-invalidated", action="store_true")
    args = ap.parse_args(argv)

    try:
        # Read-only: a verifier never migrates the ledger it reads.
        ledger = LedgerStore(args.ledger, BlobStore(args.archive), migrate=False)
    except LedgerSchemaError as exc:
        print(f"cannot open the ledger: {exc}", file=sys.stderr)
        return 1
    try:
        if args.command == "extract":
            record = ledger.get(args.record_id)
            if record.state is not ArtifactState.SEALED:
                print(f"record {record.record_id} is {record.state.value}, not sealed", file=sys.stderr)
                return 1
            data = normalize(SealedBundle.from_record(record, ledger.archive)).to_canonical()
            if args.output is None:
                sys.stdout.buffer.write(data + b"\n")
            else:
                args.output.write_bytes(data)
            print(f"sha256:{hashlib.sha256(data).hexdigest()}", file=sys.stderr)
            return 0
        try:
            extraction = Extraction.from_canonical(args.extraction.read_bytes().rstrip(b"\n"))
        except ExtractionFormatError as exc:
            problems = [f"not a canonical extraction: {exc}"]
        else:
            problems = Resolver(ledger, allow_invalidated=args.allow_invalidated).verify(extraction)
        print(json.dumps({"ok": not problems, "problems": problems}, indent=2))
        return 0 if not problems else 1
    except ResolutionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        ledger.close()


if __name__ == "__main__":
    sys.exit(main())
