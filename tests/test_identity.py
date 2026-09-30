"""
Identity contract (roadmap #42): typed, self-verifying references.

  1. Every reference type round-trips through exactly one canonical string;
     every other spelling is refused.
  2. Each type resolves against honest evidence to what it names.
  3. Each type stops resolving once the evidence it committed to changes:
     a wrong digest, a tampered blob, an invalidated record, a different
     observation, a misquoted anchor, a rewritten event.
  4. Extraction chunks carry anchor references that resolve to their text.
  5. The CLI lists and resolves references.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from stele.adapters import ExtractionAdapter
from stele.archive import Snapshot, SnapshotKind, Source
from stele.contracts.adapter import SealedBundle
from stele.extraction import Anchor, normalize
from stele.identity import (
    AnchorRef, EventRef, IdentityResolver, Observation, ObservationRef, RecordRef, RefFormatError,
    ResolveError, SnapshotRef, SourceRef, anchor_ref, event_ref, observation_of, parse_ref,
    record_ref,
)
from stele.identity.__main__ import main as cli
from stele.ledger.events import EventLog
from stele.ledger.models import ParserIdentity
from tests.ledger_helpers import open_ledger

D1, D2 = "a" * 64, "b" * 64
RID = "3f2c9a1e-0000-4000-8000-000000000001"

MARKDOWN = "# Title\n\nFirst paragraph, naïve.\n\n## Part\n\n- item\n".encode("utf-8")


def _sealed(tmp_path: Path, *, with_source: bool = True):
    """A SEALED Marker-shaped record over an archived input Snapshot."""
    ledger = open_ledger(tmp_path / "ledger.db")
    archive = ledger.archive
    doc = b"%PDF-1.7 the input document"
    digest = archive.put_bytes(doc)
    snapshot = archive.put_snapshot(Snapshot(SnapshotKind.FILE, digest, len(doc), 1))
    source = Source(locator="/corpus/report.pdf") if with_source else None
    if source is not None:
        archive.put_source(source)
    out = tmp_path / f"out-{uuid.uuid4().hex[:6]}"
    out.mkdir()
    (out / "document.md").write_bytes(MARKDOWN)
    (out / "stele-parser.json").write_text(json.dumps(
        {"parser": "marker", "version": "1.0", "outputs": {"markdown": "document.md"}}
    ))
    record = ledger.create_pending(
        run_id=str(uuid.uuid4()), artifact_dir=out, artifact_paths=sorted(out.iterdir()),
        parser=ParserIdentity("marker", "1.0"), parser_config={},
        input_snapshot=snapshot, source=source,
    )
    return ledger, ledger.seal(record.record_id)


def _extraction(ledger, record):
    return normalize(SealedBundle.from_record(ledger.get(record.record_id), ledger.archive))


# ---------------------------------------------------------------------------
# 1 — canonical strings
# ---------------------------------------------------------------------------

class TestFormat:

    @pytest.mark.parametrize("ref", [
        SourceRef(D1),
        SnapshotRef(SnapshotKind.FILE, D1),
        SnapshotRef(SnapshotKind.TREE, D2),
        RecordRef(RID, D1),
        ObservationRef(RID, D2),
        EventRef(1, D1),
        EventRef(12345, D2),
        AnchorRef(RecordRef(RID, D1), Anchor(artifact="document.md", digest=D2, range=(0, 12))),
        AnchorRef(RecordRef(RID, D1), Anchor(artifact="dir/a b@#%.md", digest=D2, range=(3, 3))),
        AnchorRef(RecordRef(RID, D1), Anchor(
            artifact="conversations/c1.json", digest=D2,
            pointer="/mapping/a~1b c/message", render="chatgpt-message")),
        AnchorRef(RecordRef(RID, D1), Anchor(
            artifact="x.json", digest=D2, pointer="", render="json-string")),
        AnchorRef(RecordRef(RID, D1), Anchor(artifact="文書.md", digest=D2, range=(0, 1))),
    ], ids=str)
    def test_round_trip(self, ref) -> None:
        text = str(ref)
        assert text.startswith("stele:")
        assert parse_ref(text) == ref
        assert str(parse_ref(text)) == text

    def test_expected_spellings(self) -> None:
        assert str(SnapshotRef("file", D1)) == f"stele:snapshot:file:{D1}"
        assert str(RecordRef(RID, D1)) == f"stele:record:{RID}@{D1}"
        assert str(EventRef(7, D1)) == f"stele:event:7@{D1}"
        ref = AnchorRef(RecordRef(RID, D1), Anchor(artifact="a b/c.md", digest=D2, range=(4, 9)))
        assert str(ref) == f"stele:anchor:{RID}@{D1}/a%20b/c.md@{D2}#bytes=4-9"
        ref = AnchorRef(RecordRef(RID, D1), Anchor(
            artifact="c.json", digest=D2, pointer="/mapping/n 1/message", render="chatgpt-message"))
        assert str(ref).endswith("#pointer=/mapping/n%201/message&render=chatgpt-message")

    @pytest.mark.parametrize("text", [
        "",
        f"stele:thing:{D1}",
        f"source:{D1}",
        f"stele:source:{D1.upper()}",
        f"stele:source:{D1[:-1]}",
        f"stele:snapshot:blob:{D1}",
        f"stele:snapshot:FILE:{D1}",
        f"stele:record:{RID}",
        f"stele:record:{RID}@{D1}@{D2}",
        f"stele:record:bad/id@{D1}",
        f"stele:event:0@{D1}",
        f"stele:event:07@{D1}",
        f"stele:event:-1@{D1}",
        f"stele:anchor:{RID}@{D1}/doc.md@{D2}",                     # no fragment
        f"stele:anchor:{RID}@{D1}/doc.md@{D2}#bytes=9-4",           # backwards range
        f"stele:anchor:{RID}@{D1}/doc.md@{D2}#bytes=01-4",          # leading zero
        f"stele:anchor:{RID}@{D1}/doc.md@{D2}#bytes=4",
        f"stele:anchor:{RID}@{D1}/doc.md@{D2}#lines=1-2",
        f"stele:anchor:{RID}@{D1}/doc.md@{D2}#pointer=/x",          # no renderer
        f"stele:anchor:{RID}@{D1}/a%2fb.md@{D2}#bytes=0-1",         # lower-case / needless escape
        f"stele:anchor:{RID}@{D1}/a%2Fb.md@{D2}#bytes=0-1",         # "/" must stay literal
        f"stele:anchor:{RID}@{D1}/a b.md@{D2}#bytes=0-1",           # space must be escaped
        f"stele:anchor:{RID}@{D1}/@{D2}#bytes=0-1",                 # empty artifact
    ])
    def test_non_canonical_is_refused(self, text) -> None:
        with pytest.raises(RefFormatError):
            parse_ref(text)

    def test_anchor_ref_needs_typed_parts(self) -> None:
        with pytest.raises(RefFormatError):
            AnchorRef(RecordRef(RID, D1), {"artifact": "x", "digest": D2, "range": [0, 1]})

    def test_observation_digest_covers_every_field(self) -> None:
        base = Observation(RID, D1, SnapshotRef("file", D2), "2026-09-25T12:00:00+00:00")
        variants = [
            Observation(RID.replace("1", "2"), D1, base.snapshot, base.observed_at),
            Observation(RID, None, base.snapshot, base.observed_at),
            Observation(RID, D1, SnapshotRef("tree", D2), base.observed_at),
            Observation(RID, D1, base.snapshot, "2026-09-25T12:00:01+00:00"),
        ]
        assert len({base.digest, *(v.digest for v in variants)}) == 5
        assert base.ref == ObservationRef(RID, base.digest)
        with pytest.raises(RefFormatError, match="ISO 8601"):
            Observation(RID, D1, base.snapshot, "yesterday")


# ---------------------------------------------------------------------------
# 2, 3 — resolution against the evidence
# ---------------------------------------------------------------------------

class TestResolve:

    def test_every_type_resolves_to_what_it_names(self, tmp_path) -> None:
        ledger, record = _sealed(tmp_path)
        resolver = IdentityResolver(ledger)
        observation = observation_of(record)

        assert resolver.resolve(SourceRef(record.source_id)).locator == "/corpus/report.pdf"
        assert resolver.resolve(observation.snapshot).digest == record.source_hash
        assert resolver.resolve(record_ref(record)).record_id == record.record_id
        assert resolver.resolve(str(observation.ref)) == observation

        extraction = _extraction(ledger, record)
        for unit in extraction.units:
            assert resolver.resolve(str(anchor_ref(extraction, unit))) == unit.text

        log = EventLog(ledger)
        events = log.for_subject(record.record_id)
        log.close()
        assert [e.kind for e in events] == ["record.created", "record.sealed"]
        for event in events:
            assert resolver.resolve(str(event_ref(event))) == event

    def test_malformed_string_is_a_resolve_error(self, tmp_path) -> None:
        ledger, _ = _sealed(tmp_path)
        with pytest.raises(ResolveError, match="not a Stele reference"):
            IdentityResolver(ledger).resolve("https://example.com")

    def test_wrong_digests_are_refused(self, tmp_path) -> None:
        ledger, record = _sealed(tmp_path)
        resolver = IdentityResolver(ledger)
        extraction = _extraction(ledger, record)
        unit = extraction.units[1]
        forged = [
            SourceRef(D1),
            SnapshotRef("file", D1),
            SnapshotRef("tree", record.source_hash),        # right digest, wrong kind
            RecordRef(record.record_id, D1),
            RecordRef(str(uuid.uuid4()), record.artifact_hash),
            ObservationRef(record.record_id, D1),
            AnchorRef(RecordRef(record.record_id, D1), unit.anchor),
            AnchorRef(record_ref(record), Anchor(
                artifact=unit.anchor.artifact, digest=D1, range=unit.anchor.range)),
            AnchorRef(record_ref(record), Anchor(
                artifact="elsewhere.md", digest=unit.anchor.digest, range=unit.anchor.range)),
            AnchorRef(record_ref(record), Anchor(
                artifact=unit.anchor.artifact, digest=unit.anchor.digest, range=(0, 10_000))),
            EventRef(1, D1),
            EventRef(999, D1),
        ]
        for ref in forged:
            assert resolver.check(ref), f"{ref} resolved"

    def test_a_different_observation_is_refused(self, tmp_path) -> None:
        ledger, record = _sealed(tmp_path)
        honest = observation_of(record)
        for other in (
            Observation(record.record_id, None, honest.snapshot, honest.observed_at),
            Observation(record.record_id, honest.source_id, honest.snapshot,
                        "2020-01-01T00:00:00+00:00"),
        ):
            problems = IdentityResolver(ledger).check(other.ref)
            assert problems and "observation has digest" in problems[0]

    @pytest.mark.parametrize("column", ["created_at", "source_id"])
    def test_an_observation_rebuilt_from_an_edited_row_is_refused(self, tmp_path, column) -> None:
        # Edit the row, then present the observation rebuilt from it. Its
        # digest matches the row, so only the record's creation event in the
        # chain can show that the row is not what was recorded.
        ledger, record = _sealed(tmp_path)
        other_source = ledger.archive.put_source(Source(locator="/somewhere/else.pdf"))
        value = {"created_at": "2001-01-01T00:00:00+00:00", "source_id": other_source}[column]
        conn = sqlite3.connect(ledger.db_path, isolation_level=None)
        conn.execute(f"UPDATE artifact_records SET {column}=? WHERE record_id=?",
                     (value, record.record_id))
        conn.close()
        forged = observation_of(ledger.get(record.record_id))
        problems = IdentityResolver(ledger).check(forged.ref)
        assert problems and f"the record's {column} is" in problems[0]
        assert "creation event" in problems[0]

    def test_record_without_input_has_no_observation(self, tmp_path) -> None:
        ledger = open_ledger(tmp_path / "ledger.db")
        out = tmp_path / "out"
        out.mkdir()
        (out / "x.txt").write_text("x")
        record = ledger.seal(ledger.create_pending(
            run_id=str(uuid.uuid4()), artifact_dir=out, artifact_paths=[out / "x.txt"],
            parser=ParserIdentity("p", "1"), parser_config={},
        ).record_id)
        with pytest.raises(ValueError, match="no input Snapshot"):
            observation_of(record)
        assert "no input Snapshot" in IdentityResolver(ledger).check(
            ObservationRef(record.record_id, D1))[0]

    def test_observation_without_source(self, tmp_path) -> None:
        ledger, record = _sealed(tmp_path, with_source=False)
        observation = observation_of(record)
        assert observation.source_id is None
        assert IdentityResolver(ledger).resolve(observation.ref) == observation

    def test_invalidated_record_needs_permission(self, tmp_path) -> None:
        ledger, record = _sealed(tmp_path)
        extraction = _extraction(ledger, record)
        anchor = anchor_ref(extraction, extraction.units[0])
        observation = observation_of(record).ref
        ledger.invalidate(record.record_id, "withdrawn")

        strict = IdentityResolver(ledger)
        for ref in (record_ref(record), anchor):
            assert "invalidated" in strict.check(ref)[0]
        # What was observed stays observed.
        assert strict.check(observation) == []

        lenient = IdentityResolver(ledger, allow_invalidated=True)
        assert lenient.check(record_ref(record)) == []
        assert lenient.resolve(anchor) == extraction.units[0].text

    def test_tampered_evidence_is_refused(self, tmp_path) -> None:
        ledger, record = _sealed(tmp_path)
        resolver = IdentityResolver(ledger)
        extraction = _extraction(ledger, record)
        anchor = anchor_ref(extraction, extraction.units[0])
        observation = observation_of(record)

        def blob(digest: str) -> Path:
            path = ledger.archive._blob_path(digest)
            path.chmod(0o644)
            return path

        # Output blob: the anchor reads verified bytes, so it is refused.
        md = blob(record.artifact_manifest["document.md"])
        md.write_bytes(md.read_bytes().replace(b"Title", b"Tit1e"))
        assert "unreadable" in resolver.check(anchor)[0]

        # Input blob: the Snapshot, and so the Observation, no longer resolve.
        src = blob(record.source_hash)
        src.write_bytes(b"%PDF-1.7 another document!!")
        assert resolver.check(observation.snapshot)
        assert resolver.check(observation.ref)

    def test_rewritten_event_is_refused(self, tmp_path) -> None:
        ledger, record = _sealed(tmp_path)
        log = EventLog(ledger)
        first = log.for_subject(record.record_id)[0]
        log.close()
        ref = event_ref(first)
        assert IdentityResolver(ledger).check(ref) == []

        conn = sqlite3.connect(ledger.db_path)
        conn.execute("DROP TRIGGER events_append_only_u")
        conn.execute("UPDATE events SET body=? WHERE seq=?", ('{"forged":true}', first.seq))
        conn.commit()
        conn.close()
        problems = IdentityResolver(ledger).check(ref)
        assert problems and "altered" in problems[0]


# ---------------------------------------------------------------------------
# 4 — extraction chunks cite their evidence
# ---------------------------------------------------------------------------

def test_extraction_chunks_carry_resolvable_anchor_refs(tmp_path) -> None:
    ledger, record = _sealed(tmp_path)
    bundle = SealedBundle.from_record(ledger.get(record.record_id), ledger.archive)
    chunks = ExtractionAdapter().transform(bundle)
    resolver = IdentityResolver(ledger)
    assert chunks
    for chunk in chunks:
        ref = parse_ref(chunk.metadata["stele_anchor"])
        assert isinstance(ref, AnchorRef)
        assert ref.record == record_ref(record)
        assert resolver.resolve(ref) == chunk.content


# ---------------------------------------------------------------------------
# 5 — CLI
# ---------------------------------------------------------------------------

def test_cli_refs_and_resolve(tmp_path, capsys) -> None:
    ledger, record = _sealed(tmp_path)
    db, archive = str(ledger.db_path), str(ledger.archive.root)
    ledger.close()

    assert cli(["refs", db, archive, record.record_id]) == 0
    refs = json.loads(capsys.readouterr().out)
    assert refs["record"] == str(record_ref(record))
    assert refs["observation"].startswith(f"stele:observation:{record.record_id}@")
    assert len(refs["events"]) == 2

    good = [refs["record"], refs["observation"], refs["snapshot"], *refs["events"]]
    assert cli(["resolve", db, archive, *good]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["ok"] for line in lines] == [True] * len(good)

    assert cli(["resolve", db, archive, refs["record"], f"stele:record:{record.record_id}@{D1}"]) == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["ok"] for line in lines] == [True, False]

    assert cli(["refs", db, archive, "no-such-record"]) == 1
