"""
python -m stele.verification corpus ROOT -o corpus.json
python -m stele.verification run --corpus corpus.json --root ROOT --ledger L --archive A \
        --journal J (--script P | --parser NAME | --extractor NAME) \
        --baseline bubblewrap [--candidate oci-runc ...]
python -m stele.verification report --corpus corpus.json --root ROOT --ledger L --archive A \
        --journal J [--waivers W.json] [--max-baseline-failure-rate R] [--max-slowdown X] -o report.json
python -m stele.verification sign-off --corpus corpus.json --root ROOT --ledger L --archive A \
        --journal J report.json --by NAME [--note TEXT]
python -m stele.verification check --ledger L --archive A REPORT_DIGEST
python -m stele.verification list --ledger L --archive A
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..archive import BlobStore
from ..ledger.store import LedgerSchemaError, LedgerStore
from .campaign import Campaign, Journal, Observation
from .corpus import Corpus
from .report import Gates, VerificationReport, Waiver, build_report
from .signoff import SignOffRefused, check_sign_off, sign_off, sign_offs


def _ledger(args: argparse.Namespace, *, migrate: bool = False) -> LedgerStore:
    """The ledger; only `run`, which records runs, may migrate it.

    report, sign-off, check and list verify the ledger, and a verifier never
    migrates what it verifies (LedgerSchemaError says why it cannot open it).
    """
    return LedgerStore(args.ledger, BlobStore(args.archive), migrate=migrate)


def _store_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)


def _campaign_args(p: argparse.ArgumentParser) -> None:
    _store_args(p)
    p.add_argument("--corpus", type=Path, required=True, help="stele.corpus manifest")
    p.add_argument("--root", type=Path, required=True, help="the corpus directory")
    p.add_argument("--journal", type=Path, required=True)


def _progress(obs: Observation) -> None:
    extra = (obs.failure or {}).get("reason") or obs.detail or obs.record_id or ""
    print(f"{obs.status:17} {obs.lane:14} {obs.path}  {extra}"[:240], file=sys.stderr)


def _cmd_corpus(args: argparse.Namespace) -> int:
    corpus = Corpus.build(args.root, name=args.name)
    corpus.save(args.output)
    print(json.dumps({
        "corpus_id": corpus.corpus_id, "documents": len(corpus), "bytes": corpus.total_bytes,
        "excluded": len(corpus.excluded),
    }, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from . import lanes as lane_mod

    backends = [args.baseline, *args.candidate]
    config = json.loads(args.config) if args.config else {}
    if args.script:
        lanes = lane_mod.script_lanes(
            args.script, backends, name=args.name, version=args.version,
            deterministic=args.deterministic, timeout_seconds=args.timeout, engine=args.engine,
            **({"image": args.image} if args.image else {}),
        )
    elif args.parser:
        lanes, defaults, _ = lane_mod.packaged_lanes(args.parser, backends, engine=args.engine)
        config = {**defaults, **config}
    else:
        lanes = lane_mod.extractor_lanes(args.extractor, backends)
    ledger = _ledger(args, migrate=True)
    try:
        campaign = Campaign(
            ledger, Corpus.load(args.corpus), args.root, lanes, baseline=args.baseline,
            journal=args.journal, parser_config=config, keep_outputs=args.keep_outputs,
        )
        done = campaign.run(limit=args.limit, progress=None if args.quiet else _progress)
        remaining = sum(1 for _ in campaign.pending())
    finally:
        ledger.close()
    print(json.dumps({"observed": len(done), "remaining": remaining}, indent=2))
    return 0


def _policy_for(parser: dict) -> object | None:
    from ..parsers.catalog import PARSERS

    image = PARSERS.get(parser["name"])
    if image is not None and image.version == parser["version"]:
        return image.comparison
    return None


def _cmd_report(args: argparse.Namespace) -> int:
    ledger = _ledger(args)
    try:
        journal = Journal.read(args.journal)
        report = build_report(
            ledger, Corpus.load(args.corpus), journal, root=args.root,
            policy=_policy_for(journal.header["parser"]),
            gates=Gates(
                max_baseline_failure_rate=args.max_baseline_failure_rate,
                max_slowdown=args.max_slowdown, check_extractions=not args.no_extractions,
            ),
            waivers=Waiver.load_all(args.waivers) if args.waivers else (),
        )
    finally:
        ledger.close()
    args.output.write_bytes(report.to_canonical())
    for gate in report.data["gates"]:
        mark = "PASS" if gate["passed"] else "FAIL"
        print(f"{mark}  {gate['name']:26} {gate['detail']}")
        for example in gate.get("examples", [])[:5] if not gate["passed"] else []:
            print(f"        {example}")
    print(f"report sha256:{report.digest} -> {args.output}  "
          f"({'passed' if report.passed else 'NOT passed'})")
    return 0 if report.passed else 1


def _cmd_sign_off(args: argparse.Namespace) -> int:
    report = VerificationReport.from_canonical(args.report.read_bytes())
    ledger = _ledger(args)
    try:
        journal = Journal.read(args.journal)
        done = sign_off(
            ledger, report, corpus=Corpus.load(args.corpus), journal=journal, root=args.root,
            policy=_policy_for(journal.header["parser"]), signed_off_by=args.by, note=args.note,
        )
    except SignOffRefused as exc:
        print("sign-off refused:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    finally:
        ledger.close()
    print(json.dumps({
        "report_digest": done.report_digest, "corpus_id": done.corpus_id,
        "signed_off_by": done.signed_off_by, "anchor": f"{done.seq}:{done.event_hash}",
    }, indent=2))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    ledger = _ledger(args)
    try:
        problems = check_sign_off(ledger, args.digest)
    finally:
        ledger.close()
    print(json.dumps({"ok": not problems, "problems": problems}, indent=2))
    return 0 if not problems else 1


def _cmd_list(args: argparse.Namespace) -> int:
    ledger = _ledger(args)
    try:
        print(json.dumps(sign_offs(ledger), indent=2))
    finally:
        ledger.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stele.verification",
                                 description="Full-corpus production verification and sign-off.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("corpus", help="pin a corpus directory by content")
    p.add_argument("root", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--name", default=None)
    p.set_defaults(func=_cmd_corpus)

    p = sub.add_parser("run", help="run (or resume) a campaign")
    _campaign_args(p)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--script", type=Path, help="a Python parser script")
    which.add_argument("--parser", help="a packaged parser: mineru, marker, docling")
    which.add_argument("--extractor", help="a Wasm extractor, e.g. chatgpt-export-split")
    p.add_argument("--baseline", default="bubblewrap")
    p.add_argument("--candidate", action="append", default=[])
    p.add_argument("--name", default=None, help="script parser name (default: file stem)")
    p.add_argument("--version", default=None, help="script parser version (default: its digest)")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--image", default=None, help="container image for a script parser")
    p.add_argument("--engine", default=None, choices=["podman", "docker"])
    p.add_argument("--config", default=None, help="parser config as JSON")
    p.add_argument("--limit", type=int, default=None, help="stop after this many runs")
    p.add_argument("--keep-outputs", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("report", help="build and gate the verification report")
    _campaign_args(p)
    p.add_argument("--waivers", type=Path, default=None)
    p.add_argument("--max-baseline-failure-rate", type=float, default=None)
    p.add_argument("--max-slowdown", type=float, default=None)
    p.add_argument("--no-extractions", action="store_true")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("sign-off", help="sign off a passed report")
    _campaign_args(p)
    p.add_argument("report", type=Path)
    p.add_argument("--by", required=True)
    p.add_argument("--note", default="")
    p.set_defaults(func=_cmd_sign_off)

    p = sub.add_parser("check", help="re-verify a sign-off")
    _store_args(p)
    p.add_argument("digest")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("list", help="list sign-offs")
    _store_args(p)
    p.set_defaults(func=_cmd_list)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except LedgerSchemaError as exc:
        print(f"cannot open the ledger: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
