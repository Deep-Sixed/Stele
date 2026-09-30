"""
Resource limits, telemetry, fault diagnostics and cleanup (roadmap #11).

  1. Every run result carries normalized telemetry and, on failure, exactly
     one structured reason, whatever backend ran it.
  2. Fault injection per backend: timeout, out of memory, blocked syscall,
     Wasm trap / fuel / epoch, crash, exit status, engine error, and unsafe
     output each produce the right reason.
  3. After every fault, nothing is left behind: no output, no staging
     directory, no process and no container.

The process-backend proofs run on every registered host-process backend
(conftest.containment_backend); the Wasm proofs on every Wasm backend.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

from stele.containment.backend import (
    Capability,
    ExecutionOutcome,
    ParserRequirements,
    SandboxBackend,
)
from stele.containment.oci import OciBackend
from stele.containment.runner import run_in_sandbox
from stele.containment.sandbox import SandboxConfig
from stele.containment.telemetry import FailureReason, RunTelemetry
from stele.containment.wasm import WasmtimeBackend
from tests.conftest import _PROVED, _proof_id

PROBE = Path(__file__).parent / "fixtures" / "wasm" / "probe.wat"
WASM = ParserRequirements(wasm_module=True)
MiB = 1024 * 1024


def _staging_left(result) -> list[Path]:
    return list(Path(tempfile.gettempdir()).glob(f"stele-{result.run_id}-*"))


def _processes_with(marker: str) -> list[str]:
    """Host processes whose command line carries marker (Linux /proc)."""
    found = []
    for entry in Path("/proc").iterdir() if Path("/proc").is_dir() else []:
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if marker.encode() in cmdline and str(os.getpid()) != entry.name:
            found.append(cmdline.replace(b"\0", b" ").decode(errors="replace"))
    return found


def _containers_left(backend) -> list[str]:
    if not isinstance(backend, OciBackend):
        return []
    engine = backend.probe().engine
    listed = subprocess.run(
        [engine, "ps", "-a", "--filter", "name=stele-", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    return [n for n in listed.stdout.split() if n]


def _assert_clean(result, backend, marker: str | None = None) -> None:
    assert not result.artifact_dir.exists(), "a failed run's output must be removed"
    assert result.artifact_paths == []
    assert _staging_left(result) == []
    if marker is not None:
        assert _processes_with(marker) == []
    assert _containers_left(backend) == []


def _run(tmp_path: Path, python: str, code: str, *, timeout: int = 60, backend=None,
         with_input: bool = True):
    source = None
    if with_input:
        source = tmp_path / "in" / "doc.txt"
        source.parent.mkdir(exist_ok=True)
        source.write_text("input")
    return run_in_sandbox(
        SandboxConfig(
            command=[python, "-c", code],
            artifact_dir=tmp_path / f"out-{uuid.uuid4().hex[:6]}",
            input_path=source,
            timeout_seconds=timeout,
        ),
        backend=backend,
    )


# Every fault writes a partial artifact first, so cleanup is observable.
PARTIAL = (
    "import os\n"
    "open(os.path.join(os.environ['STELE_OUTPUT_DIR'], 'partial.txt'), 'w').write('half')\n"
)


# ---------------------------------------------------------------------------
# 1 + 2 + 3 — host-process backends (bubblewrap, OCI runc / gVisor)
# ---------------------------------------------------------------------------

class TestProcessBackendFaults:

    def test_success_carries_normalized_telemetry(
        self, tmp_path, containment_backend, sandbox_python
    ) -> None:
        result = _run(tmp_path, sandbox_python, PARTIAL + "sum(range(10**6))\n")
        assert result.succeeded and result.failure is None
        t = result.telemetry
        assert isinstance(t, RunTelemetry)
        assert t.backend == containment_backend.name == result.backend
        assert t.runtime
        assert t.exit_code == 0 and t.wall_time_seconds > 0
        assert t.limits["timeout_seconds"] == 60
        if isinstance(containment_backend, OciBackend):
            assert t.limits["memory"] == containment_backend.memory
            assert t.runtime.endswith("/" + containment_backend.runtime)
            # --rm containers take their accounting with them: unknown, not zero.
            assert t.cpu_time_seconds is None and t.peak_memory_bytes is None
        else:
            assert t.cpu_time_seconds is not None and t.cpu_time_seconds > 0
            assert t.peak_memory_bytes is not None and t.peak_memory_bytes > MiB
        assert result.artifact_dir.is_dir() and [p.name for p in result.artifact_paths] == [
            "partial.txt"
        ]

    def test_timeout_kills_the_whole_tree(
        self, tmp_path, containment_backend, sandbox_python
    ) -> None:
        marker = f"stele-fault-{uuid.uuid4().hex}"
        code = PARTIAL + (
            "import subprocess, sys, time\n"
            # A grandchild in its own session, as a parser's helper might be.
            f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3600)', "
            f"'{marker}'], start_new_session=True)\n"
            "time.sleep(3600)\n"
        )
        result = _run(tmp_path, sandbox_python, code, timeout=3)
        assert result.timed_out and not result.succeeded
        assert result.failure.reason is FailureReason.TIMEOUT
        assert "3s" in result.failure.detail
        assert result.telemetry.wall_time_seconds < 60
        _assert_clean(result, containment_backend, marker)

    def test_crash_is_reported_with_its_signal(
        self, tmp_path, containment_backend, sandbox_python
    ) -> None:
        # A real memory fault: a container runs the parser as PID 1, which
        # ignores signals sent with kill() but not faults the kernel raises.
        code = PARTIAL + "import ctypes\nctypes.string_at(0)\n"
        result = _run(tmp_path, sandbox_python, code)
        assert result.failure.reason is FailureReason.CRASHED
        assert result.failure.signal_name == "SIGSEGV"
        assert result.telemetry.exit_code == 128 + 11
        _assert_clean(result, containment_backend)

    def test_exit_status_is_reported(self, tmp_path, containment_backend, sandbox_python) -> None:
        result = _run(tmp_path, sandbox_python, PARTIAL + "raise SystemExit(3)\n")
        assert result.failure.reason is FailureReason.EXIT_STATUS
        assert result.failure.exit_code == 3 and result.failure.signal is None
        _assert_clean(result, containment_backend)

    @pytest.mark.parametrize("kind", ["symlink", "fifo"])
    def test_unsafe_output_is_a_failure_not_an_exception(
        self, tmp_path, containment_backend, sandbox_python, kind
    ) -> None:
        make = (
            "os.symlink('/etc/hostname', os.path.join(out, 'link'))\n" if kind == "symlink"
            else "os.mkfifo(os.path.join(out, 'pipe'))\n"
        )
        code = PARTIAL + "out = os.environ['STELE_OUTPUT_DIR']\n" + make
        result = _run(tmp_path, sandbox_python, code)
        if kind == "fifo" and getattr(containment_backend, "runtime", None) == "runsc":
            # gVisor keeps the FIFO inside the sandbox: it never exists on the
            # host, so there is nothing unsafe to refuse.
            assert result.succeeded
            assert [p.name for p in result.artifact_paths] == ["partial.txt"]
            return
        assert result.exit_code == 0 and not result.succeeded
        assert result.failure.reason is FailureReason.UNSAFE_ARTIFACT
        assert "symlink" in result.failure.detail or "non-regular" in result.failure.detail
        _assert_clean(result, containment_backend)


_LIMITED = [b for b in _PROVED if isinstance(b, OciBackend) and not b.gpu]


@pytest.mark.parametrize("backend", _LIMITED, ids=[_proof_id(b) for b in _LIMITED])
def test_container_out_of_memory(tmp_path: Path, backend: OciBackend) -> None:
    if not backend.available():
        pytest.skip(f"{backend.name} backend unavailable here: {backend.unavailable_reason()}")
    if Capability.RESOURCE_LIMITS not in backend.capabilities():
        pytest.skip(f"{backend.name} does not enforce resource limits here")
    limited = OciBackend(engine=backend.engine, runtime=backend.runtime, memory="96m")
    code = PARTIAL + "blob = b'x' * (512 * 1024 * 1024)\n"
    result = _run(tmp_path, "python3", code, backend=limited)
    assert result.failure.reason is FailureReason.OUT_OF_MEMORY, result.failure
    assert "96m" in result.failure.detail
    assert result.telemetry.limits["memory"] == "96m"
    _assert_clean(result, limited)


@pytest.mark.parametrize("backend", _LIMITED, ids=[_proof_id(b) for b in _LIMITED])
def test_container_engine_error(tmp_path: Path, backend: OciBackend) -> None:
    if not backend.available():
        pytest.skip(f"{backend.name} backend unavailable here: {backend.unavailable_reason()}")
    result = run_in_sandbox(
        SandboxConfig(command=["/no/such/parser"], artifact_dir=tmp_path / "out"),
        backend=backend,
    )
    assert result.failure.reason is FailureReason.ENGINE_ERROR
    assert result.failure.exit_code in (125, 126, 127)
    _assert_clean(result, backend)


@pytest.mark.parametrize("backend", _LIMITED, ids=[_proof_id(b) for b in _LIMITED])
@pytest.mark.parametrize("code", [125, 126, 127])
def test_parser_exit_125_to_127_is_its_own(tmp_path: Path, backend: OciBackend, code: int) -> None:
    """Issue #38: a parser exiting 125-127 is not an engine error."""
    if not backend.available():
        pytest.skip(f"{backend.name} backend unavailable here: {backend.unavailable_reason()}")
    # Neutral wording: a parser that copies the engine's own "Error: ..." line at
    # exit 125 cannot be told apart from the engine (see oci._engine_failed).
    result = _run(tmp_path, "python3", f"import sys; print('parser gave up', file=sys.stderr); "
                                       f"sys.exit({code})", backend=backend)
    assert result.failure.reason is FailureReason.EXIT_STATUS, result.failure
    assert result.failure.exit_code == code
    _assert_clean(result, backend)


@pytest.mark.parametrize("backend", _LIMITED, ids=[_proof_id(b) for b in _LIMITED])
def test_no_writable_scratch_in_a_container(tmp_path: Path, backend: OciBackend) -> None:
    """Issue #38: with writable_scratch=False only /stele/output is writable."""
    if not backend.available():
        pytest.skip(f"{backend.name} backend unavailable here: {backend.unavailable_reason()}")
    code = (
        "import os, sys\n"
        "blocked = []\n"
        "for d in ('/tmp', '/var', '/run', '/home', '/root', '/mnt', '/etc', '/'):\n"
        "    try:\n"
        "        open(os.path.join(d, 'stele-probe'), 'w').close()\n"
        "        print('WROTE', d, file=sys.stderr)\n"
        "    except OSError:\n"
        "        blocked.append(d)\n"
        "open('/stele/output/ok.txt', 'w').write(' '.join(blocked))\n"
    )
    config = SandboxConfig(command=["python3", "-c", code], artifact_dir=tmp_path / "out",
                           writable_scratch=False)
    result = run_in_sandbox(config, backend=backend)
    assert result.exit_code == 0, result.stderr
    assert "WROTE" not in result.stderr, result.stderr
    assert (tmp_path / "out" / "ok.txt").read_text().split() == [
        "/tmp", "/var", "/run", "/home", "/root", "/mnt", "/etc", "/",
    ]


def test_blocked_syscall_is_reported(tmp_path: Path) -> None:
    from stele.containment import seccomp
    from stele.containment.backend import BubblewrapBackend

    backend = BubblewrapBackend()
    if not backend.available() or backend.seccomp_program() is None:
        pytest.skip("seccomp proof requires Linux bubblewrap with a supported architecture")
    nr = seccomp._ARCHES[seccomp.host_machine()][1]["ptrace"]
    code = PARTIAL + f"import ctypes\nctypes.CDLL(None).syscall({nr}, 0, 0, 0, 0)\n"
    result = _run(tmp_path, str(Path(sys.executable).resolve()), code, backend=backend)
    assert result.failure.reason is FailureReason.SYSCALL_BLOCKED
    assert result.failure.signal_name == "SIGSYS"
    _assert_clean(result, backend)


# ---------------------------------------------------------------------------
# 1 + 2 + 3 — Wasm backends
# ---------------------------------------------------------------------------

TRAP_WAT = '(module (memory (export "memory") 1) (func (export "_start") unreachable))'


def _wasm(tmp_path: Path, command: list[str], backend, *, timeout: int = 30):
    source = tmp_path / "in" / "sample.txt"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(b"hello input\n")
    return run_in_sandbox(
        SandboxConfig(
            command=command, artifact_dir=tmp_path / f"out-{uuid.uuid4().hex[:6]}",
            input_path=source, timeout_seconds=timeout,
        ),
        requirements=WASM, backend=backend,
    )


class TestWasmFaults:

    def test_success_carries_normalized_telemetry(self, tmp_path, wasm_backend) -> None:
        result = _wasm(tmp_path, [str(PROBE), "a"], wasm_backend)
        assert result.succeeded
        t = result.telemetry
        assert t.backend == "wasmtime" and t.runtime.startswith("wasmtime")
        assert t.cpu_time_seconds is not None and t.cpu_time_seconds >= 0
        assert t.peak_memory_bytes is not None and t.peak_memory_bytes % 65536 == 0
        assert t.counters["fuel_consumed"] > 0
        assert t.limits == {
            "timeout_seconds": 30, "memory_bytes": wasm_backend.memory_limit_bytes,
            "fuel": wasm_backend.fuel, "output_bytes": wasm_backend.output_limit_bytes,
        }

    def test_epoch_timeout(self, tmp_path, wasm_backend) -> None:
        result = _wasm(tmp_path, [str(PROBE), "l"], wasm_backend, timeout=1)
        assert result.failure.reason is FailureReason.TIMEOUT and result.timed_out
        assert result.telemetry.cpu_time_seconds > 0.1
        _assert_clean(result, wasm_backend)

    def test_fuel_exhaustion_is_a_cpu_limit(self, tmp_path, wasm_backend) -> None:
        result = _wasm(tmp_path, [str(PROBE), "l"], type(wasm_backend)(fuel=2_000_000))
        assert result.failure.reason is FailureReason.CPU_LIMIT
        assert result.telemetry.counters["fuel_consumed"] == 2_000_000
        _assert_clean(result, wasm_backend)

    def test_memory_cap_is_out_of_memory(self, tmp_path, wasm_backend) -> None:
        result = _wasm(tmp_path, [str(PROBE), "g"], type(wasm_backend)(memory_limit_bytes=4 * MiB))
        assert result.failure.reason is FailureReason.OUT_OF_MEMORY
        assert result.telemetry.peak_memory_bytes == 4 * MiB
        _assert_clean(result, wasm_backend)

    def test_trap(self, tmp_path, wasm_backend) -> None:
        module = tmp_path / "trap.wat"
        module.write_text(TRAP_WAT)
        result = _wasm(tmp_path, [str(module)], wasm_backend)
        assert result.failure.reason is FailureReason.WASM_TRAP
        assert "unreachable" in result.failure.detail
        _assert_clean(result, wasm_backend)

    def test_exit_status(self, tmp_path, wasm_backend) -> None:
        result = _wasm(tmp_path, [str(PROBE), "x3"], wasm_backend)
        assert result.failure.reason is FailureReason.EXIT_STATUS
        assert result.failure.exit_code == 3
        _assert_clean(result, wasm_backend)

    def test_unloadable_module_is_an_engine_error(self, tmp_path, wasm_backend) -> None:
        module = tmp_path / "garbage.wasm"
        module.write_bytes(b"\0asm not really")
        result = _wasm(tmp_path, [str(module)], wasm_backend)
        assert result.failure.reason is FailureReason.ENGINE_ERROR
        _assert_clean(result, wasm_backend)


# ---------------------------------------------------------------------------
# Generic reasons for backends that do not classify
# ---------------------------------------------------------------------------

class _Plain(SandboxBackend):
    name = "plain"

    def __init__(self, outcome: ExecutionOutcome) -> None:
        self.outcome = outcome

    def capabilities(self):
        return frozenset({
            Capability.FILESYSTEM_ISOLATION, Capability.NETWORK_ISOLATION, Capability.HOST_PROCESS,
        })

    def available(self):
        return True

    def unavailable_reason(self):
        return ""

    def execute(self, config):
        config.artifact_dir.mkdir(parents=True, exist_ok=True)
        (config.artifact_dir / "partial.txt").write_text("half")
        return self.outcome


@pytest.mark.parametrize(("outcome", "reason"), [
    (ExecutionOutcome(0, "", "", 0.1), None),
    (ExecutionOutcome(5, "", "", 0.1), FailureReason.EXIT_STATUS),
    (ExecutionOutcome(-1, "", "", 9.9, timed_out=True), FailureReason.TIMEOUT),
    (ExecutionOutcome(159, "", "", 0.1, violation="blocked"), FailureReason.SYSCALL_BLOCKED),
], ids=["ok", "exit", "timeout", "violation"])
def test_generic_reasons_and_cleanup(tmp_path: Path, outcome, reason) -> None:
    existing = tmp_path / "out"
    existing.mkdir()  # the caller made it: emptied on failure, not removed
    result = run_in_sandbox(
        SandboxConfig(command=["x"], artifact_dir=existing), backend=_Plain(outcome)
    )
    assert (result.failure.reason if result.failure else None) is reason
    assert result.telemetry.backend == "plain"
    assert result.telemetry.cpu_time_seconds is None  # not measured is never zero
    assert result.telemetry.limits == {"timeout_seconds": 300}
    if reason is None:
        assert [p.name for p in result.artifact_paths] == ["partial.txt"]
    else:
        assert existing.is_dir() and list(existing.iterdir()) == []


def test_backend_exception_removes_output_it_created(tmp_path: Path) -> None:
    class Exploding(_Plain):
        def execute(self, config):
            super().execute(config)
            raise RuntimeError("engine vanished")

    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        run_in_sandbox(SandboxConfig(command=["x"], artifact_dir=out),
                       backend=Exploding(ExecutionOutcome(0, "", "", 0)))
    assert not out.exists()


def test_cleanup_error_does_not_hide_the_backend_error(tmp_path: Path, monkeypatch) -> None:
    import stele.containment.runner as runner

    class Exploding(_Plain):
        def execute(self, config):
            raise RuntimeError("engine vanished")

    def failing_cleanup(artifact_dir, created):
        raise PermissionError("cannot remove output")

    monkeypatch.setattr(runner, "_remove_output", failing_cleanup)
    with pytest.raises(RuntimeError, match="engine vanished"):
        run_in_sandbox(SandboxConfig(command=["x"], artifact_dir=tmp_path / "out"),
                       backend=Exploding(ExecutionOutcome(0, "", "", 0)))


class _Symlinking(_Plain):
    """Exits 0 after writing a symlink: an unsafe-output failure."""

    def execute(self, config):
        config.artifact_dir.mkdir(parents=True, exist_ok=True)
        os.symlink("/etc/hostname", config.artifact_dir / "link")
        return self.outcome


@pytest.mark.skipif(not hasattr(os, "symlink") or os.name == "nt", reason="needs POSIX symlinks")
def test_cli_exits_nonzero_for_a_failed_run_with_exit_code_0(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import stele.containment.runner as runner

    real = runner.run_in_sandbox
    monkeypatch.setattr(runner, "run_in_sandbox", lambda config, **kw: real(
        config, backend=_Symlinking(ExecutionOutcome(0, "", "", 0.1))
    ))
    monkeypatch.setattr(sys, "argv", ["runner", "--artifact-dir", str(tmp_path / "o"), "--", "x"])
    with pytest.raises(SystemExit) as info:
        runner._main()

    report = json.loads(capsys.readouterr().out)
    assert report["exit_code"] == 0 and report["failure"]["reason"] == "unsafe_artifact"
    assert info.value.code == 1


def test_cli_reports_failure_and_telemetry(tmp_path: Path) -> None:
    if WasmtimeBackend().available() is False:
        pytest.skip("wasmtime not installed")
    out = subprocess.run(
        [sys.executable, "-m", "stele.containment.runner", "--artifact-dir",
         str(tmp_path / "o"), "--wasm", "--", str(PROBE), "x3"],
        capture_output=True, text=True,
    )

    report = json.loads(out.stdout)
    assert report["failure"]["reason"] == "exit_status"
    assert report["telemetry"]["backend"] == "wasmtime"


# ---------------------------------------------------------------------------
# Output limits: what a run may leave in /stele/output
# ---------------------------------------------------------------------------

class _Writes(_Plain):
    """Leaves the given {relative path: size} files in the output."""

    def __init__(self, files: dict[str, int]) -> None:
        super().__init__(ExecutionOutcome(0, "", "", 0.1))
        self.files = files

    def execute(self, config):
        config.artifact_dir.mkdir(parents=True, exist_ok=True)
        for rel, size in self.files.items():
            path = config.artifact_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as fh:
                fh.truncate(size)  # sparse: the size is what the limit counts
        return self.outcome


@pytest.mark.parametrize(
    ("files", "limits", "message"),
    [
        ({"a.bin": 600, "b/c.bin": 600}, {"max_output_bytes": 1000}, "more than 1000 bytes"),
        ({f"f{i}.txt": 1 for i in range(6)}, {"max_output_files": 5}, "more than 5 files"),
    ],
    ids=["bytes", "files"],
)
def test_output_over_its_limit_fails_the_run_and_is_removed(
    tmp_path: Path, files, limits, message
) -> None:
    from stele.archive.store import BlobStore

    store = BlobStore(tmp_path / "archive")
    out = tmp_path / "out"
    result = run_in_sandbox(
        SandboxConfig(command=["x"], artifact_dir=out, **limits), backend=_Writes(files),
        store=store,
    )
    assert result.failure is not None and result.failure.reason is FailureReason.OUTPUT_LIMIT
    assert message in result.failure.detail
    assert not result.succeeded and result.artifact_paths == []
    assert not out.exists()  # the runner created it, so it is gone
    assert result.artifact_bundle_digest is None and result.artifact_digests == {}
    assert not any((store.root / "blobs").rglob("*"))  # nothing reached the archive


def test_output_at_its_limit_is_kept(tmp_path: Path) -> None:
    result = run_in_sandbox(
        SandboxConfig(command=["x"], artifact_dir=tmp_path / "out",
                      max_output_bytes=1000, max_output_files=2),
        backend=_Writes({"a.bin": 500, "b.bin": 500}),
    )
    assert result.failure is None
    assert sorted(p.name for p in result.artifact_paths) == ["a.bin", "b.bin"]


def test_default_output_limits() -> None:
    from stele.containment.sandbox import DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_MAX_OUTPUT_FILES

    config = SandboxConfig(command=["x"], artifact_dir=Path("out"))
    assert config.max_output_bytes == DEFAULT_MAX_OUTPUT_BYTES == 4 * 1024 ** 3
    assert config.max_output_files == DEFAULT_MAX_OUTPUT_FILES
