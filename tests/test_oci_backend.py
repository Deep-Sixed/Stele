"""
Roadmap #8 — OCI container backend (Podman / Docker; runc, opt-in gVisor, GPU).

Structural tests use a faked engine, so they run everywhere:

  - Every run is hardened: no network, read-only root, all capabilities
    dropped, no-new-privileges, non-root user, memory/CPU/PID limits, tmpfs
    scratch, never pulls, runs the inspected image by id.
  - The in-container layout and environment match bubblewrap.
  - runsc is never silently replaced by runc; GPU is never silently dropped,
    and is never combined with gVisor.
  - SYSCALL_FILTER is claimed only when the engine's seccomp profile is active.
  - A timed-out run force-removes its container.

Live tests run over every registered backend that is available here.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from stele.containment import backend as backend_module
from stele.containment import oci as oci_module
from stele.containment.backend import (
    Capability,
    ContainmentCleanupError,
    ParserRequirements,
    SandboxUnavailableError,
    select_backend,
)
from stele.containment.oci import (
    DEFAULT_OCI_IMAGE,
    OciBackend,
    parse_engine_info,
)
from stele.containment.capture import BoundedRun
from stele.containment.telemetry import FailureReason
from stele.containment.runner import run_in_sandbox
from stele.containment.sandbox import SandboxConfig

DIGEST = "sha256:" + "ab" * 32
IMAGE_ID = "sha256:" + "cd" * 32

DOCKER_INFO = {
    "OSType": "linux",
    "SecurityOptions": ["name=apparmor", "name=seccomp,profile=builtin", "name=cgroupns"],
    "Runtimes": {"runc": {"path": "runc"}, "io.containerd.runc.v2": {"path": "runc"}},
    "DefaultRuntime": "runc",
    "MemoryLimit": True,
    "PidsLimit": True,
    "CpuCfsQuota": True,
}
PODMAN_INFO = {
    "host": {
        "os": "linux",
        "security": {"rootless": True, "seccompEnabled": True},
        "ociRuntime": {"name": "crun"},
        "cgroupControllers": ["cpuset", "cpu", "io", "memory", "pids"],
    },
}
IMAGE_INSPECT = [{"Id": IMAGE_ID, "RepoDigests": [f"docker.io/library/python@{DIGEST}"]}]


def _fake_engine(monkeypatch, info=None, image=None, *, engine="/usr/bin/docker"):
    """Fake `<engine> info` and `<engine> image inspect` results."""
    info = DOCKER_INFO if info is None else info
    image = IMAGE_INSPECT if image is None else image
    calls: list[list[str]] = []

    def run_json(argv):
        calls.append(argv)
        if argv[1] == "info":
            return info
        if argv[1:3] == ["image", "inspect"]:
            return image
        raise AssertionError(f"unexpected probe {argv}")

    monkeypatch.setattr(oci_module, "_run_json", run_json)
    monkeypatch.setattr(OciBackend, "_find_engines", lambda self: [engine])
    return calls


def _config(tmp_path: Path, **kwargs) -> SandboxConfig:
    inp = tmp_path / "staged" / "report.pdf"
    inp.parent.mkdir()
    inp.write_bytes(b"%PDF")
    script = tmp_path / "parser.py"
    script.write_text("pass")
    return SandboxConfig(
        command=["python3", "/stele/parser", "--flag"],
        artifact_dir=tmp_path / "out",
        input_path=inp,
        script_path=script,
        **kwargs,
    )


def _argv(backend: OciBackend, config: SandboxConfig, user=(1000, 1000)) -> list[str]:
    return backend.build_argv(config, container_name="stele-test", user=user)


def _fake_subprocess(monkeypatch, fake_run) -> None:
    """Serve both the engine's probe/cleanup calls and the parser run from fake_run.

    The run goes through run_bounded(); a TimeoutExpired from fake_run stands
    for the run hitting its timeout.
    """
    monkeypatch.setattr(subprocess, "run", fake_run)

    def bounded(argv, *, timeout, limit, cwd=None, pass_fds=()):
        try:
            proc = fake_run(argv, timeout=timeout, cwd=cwd)
        except subprocess.TimeoutExpired as exc:
            return BoundedRun(-1, _text(exc.output), _text(exc.stderr), timed_out=True)
        return BoundedRun(proc.returncode, proc.stdout or "", proc.stderr or "", timed_out=False)

    monkeypatch.setattr(oci_module, "run_bounded", bounded)


def _text(raw) -> str:
    return raw.decode() if isinstance(raw, bytes) else (raw or "")


def _pairs(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


# ---------------------------------------------------------------------------
# Engine info
# ---------------------------------------------------------------------------

class TestEngineInfo:

    def test_docker_defaults(self) -> None:
        info = parse_engine_info(DOCKER_INFO)
        assert info.kind == "docker"
        assert info.os_type == "linux"
        assert info.seccomp and not info.rootless
        assert "runc" in info.runtimes
        assert info.gpu_flags == ()

    def test_unconfined_seccomp_is_not_seccomp(self) -> None:
        raw = dict(DOCKER_INFO, SecurityOptions=["name=seccomp,profile=unconfined"])
        assert not parse_engine_info(raw).seccomp
        assert not parse_engine_info(dict(DOCKER_INFO, SecurityOptions=[])).seccomp

    def test_docker_rootless_detected(self) -> None:
        raw = dict(DOCKER_INFO, SecurityOptions=["name=seccomp,profile=builtin", "name=rootless"])
        assert parse_engine_info(raw).rootless

    def test_docker_gpu_via_cdi_or_nvidia_runtime(self) -> None:
        cdi = dict(DOCKER_INFO, DiscoveredDevices=[{"Source": "cdi", "ID": "nvidia.com/gpu=0"}])
        assert parse_engine_info(cdi).gpu_flags == ("--device", "nvidia.com/gpu=all")
        runtime = dict(DOCKER_INFO, Runtimes={"runc": {}, "nvidia": {}})
        assert parse_engine_info(runtime).gpu_flags == ("--gpus", "all")

    def test_podman(self, monkeypatch) -> None:
        monkeypatch.setattr(oci_module, "_cdi_has_nvidia_gpu", lambda: True)
        info = parse_engine_info(PODMAN_INFO)
        assert info.kind == "podman"
        assert info.rootless and info.seccomp
        assert info.runtimes == frozenset({"crun"})
        assert info.resource_limits
        assert info.gpu_flags == ("--device", "nvidia.com/gpu=all")


# ---------------------------------------------------------------------------
# Hardened argv and sandbox layout
# ---------------------------------------------------------------------------

class TestArgv:

    def test_isolation_flags(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        argv = _argv(OciBackend(), _config(tmp_path))

        assert argv[:2] == ["/usr/bin/docker", "run"]
        for flag in ("--rm", "--read-only"):
            assert flag in argv
        assert _pairs(argv, "--network") == ["none"]
        assert _pairs(argv, "--cap-drop") == ["ALL"]
        assert _pairs(argv, "--security-opt") == ["no-new-privileges"]
        assert _pairs(argv, "--pull") == ["never"]
        assert _pairs(argv, "--runtime") == ["runc"]
        assert _pairs(argv, "--name") == ["stele-test"]
        assert _pairs(argv, "--user") == ["1000:1000"]
        assert "--privileged" not in argv
        assert not any(a.startswith("seccomp") for a in _pairs(argv, "--security-opt"))

    def test_resource_limits(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        backend = OciBackend(memory="1g", cpus=1.5, pids_limit=64)
        argv = _argv(backend, _config(tmp_path))
        assert _pairs(argv, "--memory") == ["1g"]
        assert _pairs(argv, "--memory-swap") == ["1g"]
        assert _pairs(argv, "--cpus") == ["1.5"]
        assert _pairs(argv, "--pids-limit") == ["64"]

    def test_layout_matches_bubblewrap(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        config = _config(tmp_path, env={"STELE_TEST": "1"})
        argv = _argv(OciBackend(), config)

        mounts = _pairs(argv, "--mount")
        assert f"type=bind,source={config.artifact_dir},target=/stele/output" in mounts
        assert (
            f"type=bind,source={config.input_path},target=/stele/input/report.pdf,readonly"
            in mounts
        )
        assert f"type=bind,source={config.script_path},target=/stele/parser,readonly" in mounts
        # The output directory is the only writable bind.
        assert [m for m in mounts if not m.endswith(",readonly")] == [
            f"type=bind,source={config.artifact_dir},target=/stele/output"
        ]
        env = _pairs(argv, "--env")
        assert "STELE_OUTPUT_DIR=/stele/output" in env
        assert "STELE_INPUT_PATH=/stele/input/report.pdf" in env
        assert "STELE_TEST=1" in env
        assert _pairs(argv, "--workdir") == ["/stele/output"]
        tmpfs = [t.split(":")[0] for t in _pairs(argv, "--tmpfs")]
        assert tmpfs == ["/tmp", "/home", "/mnt", "/root", "/run", "/var"]
        assert all(t.split(":")[1].startswith("rw,") for t in _pairs(argv, "--tmpfs"))

    def test_no_writable_scratch_makes_every_tmpfs_read_only(self, tmp_path: Path, monkeypatch) -> None:
        # Issue #38: /stele/output must then be the only writable path.
        _fake_engine(monkeypatch)
        argv = _argv(OciBackend(), _config(tmp_path, writable_scratch=False))
        tmpfs = _pairs(argv, "--tmpfs")
        assert [t.split(":")[0] for t in tmpfs] == ["/tmp", "/home", "/mnt", "/root", "/run", "/var"]
        assert all(t.split(":")[1].startswith("ro,") for t in tmpfs)
        assert "--read-only" in argv

    def test_runs_exact_command_on_inspected_image_id(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        argv = _argv(OciBackend(), _config(tmp_path))
        i = argv.index("--entrypoint")
        assert argv[i:] == ["--entrypoint", "python3", IMAGE_ID, "/stele/parser", "--flag"]

    def test_extra_ro_binds_are_readonly(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        config = _config(tmp_path, extra_ro_binds=[("/opt/weights", "/models")])
        mounts = _pairs(_argv(OciBackend(), config), "--mount")
        assert "type=bind,source=/opt/weights,target=/models,readonly" in mounts

    def test_rootless_podman_keeps_host_uid(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch, info=PODMAN_INFO, engine="/usr/bin/podman")
        argv = _argv(OciBackend(runtime="crun"), _config(tmp_path, timeout_seconds=30), user=None)
        assert _pairs(argv, "--userns") == ["keep-id"]
        assert "--user" not in argv
        assert "--read-only-tmpfs=false" in argv
        assert _pairs(argv, "--timeout") == ["30"]

    def test_mount_spec_injection_refused(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        config = _config(tmp_path, extra_ro_binds=[("/opt/w,readonly=false", "/models")])
        with pytest.raises(ValueError, match="comma"):
            _argv(OciBackend(), config)

    def test_bad_env_name_refused(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        with pytest.raises(ValueError, match="environment variable"):
            _argv(OciBackend(), _config(tmp_path, env={"A=B": "x"}))

    def test_host_root_runs_parser_as_nobody(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        seen: dict = {}

        def fake_run(argv, **kwargs):
            seen["argv"] = argv
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(oci_module, "_host_ids", lambda: (0, 0))
        monkeypatch.setattr(oci_module, "_RootHandoff", _NoopHandoff)
        _fake_subprocess(monkeypatch, fake_run)
        OciBackend().execute(_config(tmp_path))
        assert _pairs(seen["argv"], "--user") == ["65534:65534"]


class _NoopHandoff:
    def __init__(self, config) -> None:
        pass

    def give(self) -> None:
        pass

    def take_back(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Availability, runtimes, GPU, capabilities
# ---------------------------------------------------------------------------

class TestAvailability:

    def test_available_with_engine_runtime_and_image(self, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        backend = OciBackend()
        assert backend.available()
        assert backend.name == "oci-runc"
        assert backend.image_digest == DEFAULT_OCI_IMAGE.rsplit("@", 1)[1]

    def test_no_engine(self, monkeypatch) -> None:
        monkeypatch.setattr(OciBackend, "_find_engines", lambda self: [])
        backend = OciBackend()
        assert not backend.available()
        assert "no container engine found" in backend.unavailable_reason()

    def test_prefers_podman_and_falls_back_to_docker(self, monkeypatch) -> None:
        def run_json(argv):
            podman = argv[0].endswith("podman")
            if argv[1] == "info":
                return PODMAN_INFO if podman else DOCKER_INFO
            return "Error: image not known" if podman else IMAGE_INSPECT

        monkeypatch.setattr(oci_module, "_run_json", run_json)
        monkeypatch.setattr(oci_module.shutil, "which", lambda name: f"/usr/bin/{name}")
        backend = OciBackend()
        assert backend.available()
        assert backend.probe().engine == "/usr/bin/docker"

        monkeypatch.setattr(oci_module, "_run_json", lambda argv: (
            PODMAN_INFO if argv[1] == "info" else IMAGE_INSPECT
        ))
        assert OciBackend(runtime="crun").probe().engine == "/usr/bin/podman"

    def test_every_engine_failing_reports_each(self, monkeypatch) -> None:
        monkeypatch.setattr(oci_module, "_run_json", lambda argv: "daemon not running")
        monkeypatch.setattr(oci_module.shutil, "which", lambda name: f"/usr/bin/{name}")
        reason = OciBackend().unavailable_reason()
        assert "podman: `/usr/bin/podman info` failed" in reason
        assert "docker: `/usr/bin/docker info` failed" in reason

    def test_windows_container_engine_refused(self, monkeypatch) -> None:
        _fake_engine(monkeypatch, info=dict(DOCKER_INFO, OSType="windows"))
        backend = OciBackend()
        assert not backend.available()
        assert "Linux container engine" in backend.unavailable_reason()

    def test_rootless_docker_refused(self, monkeypatch) -> None:
        opts = ["name=seccomp,profile=builtin", "name=rootless"]
        _fake_engine(monkeypatch, info=dict(DOCKER_INFO, SecurityOptions=opts))
        assert "rootless Docker" in OciBackend().unavailable_reason()

    def test_missing_image_is_never_pulled(self, monkeypatch) -> None:
        calls = _fake_engine(monkeypatch, image="Error: No such image")
        backend = OciBackend()
        assert not backend.available()
        assert "never pulls" in backend.unavailable_reason()
        assert not any("pull" in argv for argv in calls)

    def test_unpinned_image_records_repo_digest(self, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        assert OciBackend(image="python:3.12-slim").image_digest == DIGEST

    def test_local_image_without_repo_digest_records_id(self, monkeypatch) -> None:
        _fake_engine(monkeypatch, image=[{"Id": IMAGE_ID, "RepoDigests": []}])
        assert OciBackend(image="local/parser:dev").image_digest == IMAGE_ID

    def test_runsc_missing_fails_clearly_never_downgrades(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        backend = OciBackend(runtime="runsc")
        assert backend.name == "oci-runsc"
        assert not backend.available()
        assert "'runsc' is not configured" in backend.unavailable_reason()
        assert "will not substitute" in backend.unavailable_reason()
        with pytest.raises(SandboxUnavailableError, match="runsc"):
            backend.execute(_config(tmp_path))
        with pytest.raises(SandboxUnavailableError, match="runsc"):
            select_backend(ParserRequirements(), [backend])

    def test_runsc_present_is_used(self, tmp_path: Path, monkeypatch) -> None:
        runtimes = dict(DOCKER_INFO["Runtimes"], runsc={"path": "/usr/bin/runsc"})
        _fake_engine(monkeypatch, info=dict(DOCKER_INFO, Runtimes=runtimes))
        backend = OciBackend(runtime="runsc")
        assert backend.available()
        assert _pairs(_argv(backend, _config(tmp_path)), "--runtime") == ["runsc"]

    def test_seccomp_claimed_only_when_active(self, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        assert Capability.SYSCALL_FILTER in OciBackend().capabilities()
        _fake_engine(monkeypatch, info=dict(DOCKER_INFO, SecurityOptions=[]))
        assert Capability.SYSCALL_FILTER not in OciBackend().capabilities()
        monkeypatch.setattr(OciBackend, "_find_engines", lambda self: [])
        assert Capability.SYSCALL_FILTER not in OciBackend().capabilities()

    def test_capabilities(self, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        caps = OciBackend().capabilities()
        assert {
            Capability.FILESYSTEM_ISOLATION,
            Capability.NETWORK_ISOLATION,
            Capability.RESOURCE_LIMITS,
            Capability.NATIVE_LIBS,
        } <= caps
        assert Capability.GPU not in caps
        assert Capability.DETERMINISTIC not in caps

    def test_resource_limits_claimed_only_when_enforced(self, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        assert Capability.RESOURCE_LIMITS in OciBackend().capabilities()
        _fake_engine(monkeypatch, info=dict(DOCKER_INFO, PidsLimit=False))
        assert Capability.RESOURCE_LIMITS not in OciBackend().capabilities()
        # Rootless Podman on cgroup v1: no delegated controllers, limits ignored.
        v1 = {"host": dict(PODMAN_INFO["host"], cgroupVersion="v1", cgroupControllers=[])}
        _fake_engine(monkeypatch, info=v1, engine="/usr/bin/podman")
        backend = OciBackend(runtime="crun")
        assert backend.available()
        assert Capability.RESOURCE_LIMITS not in backend.capabilities()


class TestGpu:

    def test_gpu_without_device_is_unavailable_not_dropped(self, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        backend = OciBackend(gpu=True)
        assert backend.name == "oci-runc-gpu"
        assert Capability.GPU in backend.capabilities()
        assert not backend.available()
        assert "no usable GPU" in backend.unavailable_reason()
        with pytest.raises(SandboxUnavailableError, match="no usable GPU"):
            select_backend(ParserRequirements(requires_gpu=True), [backend])

    def test_gpu_flags_passed_when_usable(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch, info=dict(DOCKER_INFO, Runtimes={"runc": {}, "nvidia": {}}))
        backend = OciBackend(gpu=True)
        assert backend.available()
        assert _pairs(_argv(backend, _config(tmp_path)), "--gpus") == ["all"]

    def test_no_gpu_flags_without_gpu_request(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch, info=dict(DOCKER_INFO, Runtimes={"runc": {}, "nvidia": {}}))
        argv = _argv(OciBackend(), _config(tmp_path))
        assert "--gpus" not in argv and "--device" not in argv

    def test_gvisor_and_gpu_are_never_combined(self, monkeypatch) -> None:
        runtimes = {"runc": {}, "runsc": {}, "nvidia": {}}
        _fake_engine(monkeypatch, info=dict(DOCKER_INFO, Runtimes=runtimes))
        backend = OciBackend(runtime="runsc", gpu=True)
        assert Capability.GPU not in backend.capabilities()
        assert not backend.available()
        assert "nvproxy" in backend.unavailable_reason()


# ---------------------------------------------------------------------------
# Execution, timeout, registry
# ---------------------------------------------------------------------------

class TestExitClassification:
    """Issue #38: 125-127 are the engine's only when the engine says so."""

    RUNC_NOT_FOUND = (
        'Error: runc: runc create failed: unable to start container process: error during '
        'container init: exec: "nosuch": executable file not found in $PATH: OCI runtime '
        "attempted to invoke a command that was not found"
    )
    DOCKER_NOT_FOUND = (
        "docker: Error response from daemon: failed to create task for container: failed to "
        "create shim task: OCI runtime create failed: runc create failed: unable to start "
        'container process: exec: "/no/such": stat /no/such: no such file or directory: unknown.'
    )

    @pytest.mark.parametrize("code, stderr", [
        (127, RUNC_NOT_FOUND),
        (127, DOCKER_NOT_FOUND),
        (126, 'Error: runc: ... exec: "/etc/passwd": permission denied: OCI permission denied'),
        (125, "Error: nosuchimage:1: image not known"),
        (125, 'Error: default OCI runtime "nosuch" not found: invalid argument'),
        (125, "docker: Error response from daemon: Unknown runtime specified nosuch."),
    ])
    def test_engine_failures(self, code: int, stderr: str) -> None:
        failure = OciBackend()._classify(code, "parser noise\n" + stderr + "\n", 0.1, 60, False)
        assert failure.reason is FailureReason.ENGINE_ERROR, failure
        assert failure.exit_code == code

    @pytest.mark.parametrize("code, stderr", [
        (126, "Error: parser says no"),
        (127, "Traceback ...\nSystemExit: 127"),
        (125, "the parser gave up"),
        (126, ""),
    ])
    def test_the_parsers_own_exit_status(self, code: int, stderr: str) -> None:
        failure = OciBackend()._classify(code, stderr, 0.1, 60, False)
        assert failure.reason is FailureReason.EXIT_STATUS, failure
        assert failure.exit_code == code


class TestExecution:

    def test_timeout_force_removes_container(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        monkeypatch.setattr(oci_module, "_host_ids", lambda: (1000, 1000))
        calls: list[list[str]] = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            if argv[1] == "run":
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output=b"partial")
            if argv[1:3] == ["container", "inspect"]:
                return subprocess.CompletedProcess(
                    argv, 1, "", f"Error: No such container: {argv[-1]}\n"
                )
            return subprocess.CompletedProcess(argv, 0, "", "")

        _fake_subprocess(monkeypatch, fake_run)
        outcome = OciBackend().execute(_config(tmp_path, timeout_seconds=5))
        assert outcome.timed_out and outcome.exit_code == -1
        assert outcome.stdout == "partial"
        name = _pairs(calls[0], "--name")[0]
        assert name.startswith("stele-")
        assert calls[1] == ["/usr/bin/docker", "rm", "-f", name]
        # Reported as stopped only after the engine confirms the exact name is gone.
        assert calls[2] == [
            "/usr/bin/docker", "container", "inspect", "--format", "{{.Id}}", name
        ]
        assert len(calls) == 3

    @pytest.mark.parametrize(
        "rm, inspect",
        [
            # rm fails and the container is still there.
            (lambda a: subprocess.CompletedProcess(a, 1, "", "Error: cannot kill\n"),
             lambda a: subprocess.CompletedProcess(a, 0, "abc123\n", "")),
            # rm hangs, and the engine cannot say whether the container exists.
            (lambda a: (_ for _ in ()).throw(subprocess.TimeoutExpired(a, 30)),
             lambda a: subprocess.CompletedProcess(a, 1, "", "Cannot connect to the daemon\n")),
            # rm reports success, but the container is still listed.
            (lambda a: subprocess.CompletedProcess(a, 0, "", ""),
             lambda a: subprocess.CompletedProcess(a, 0, "abc123\n", "")),
            # The engine binary disappeared.
            (lambda a: (_ for _ in ()).throw(OSError("no engine")),
             lambda a: (_ for _ in ()).throw(OSError("no engine"))),
        ],
        ids=["rm-fails", "rm-hangs-inspect-unknown", "rm-lies", "engine-gone"],
    )
    def test_timeout_without_proven_removal_is_a_containment_failure(
        self, tmp_path: Path, monkeypatch, rm, inspect
    ) -> None:
        _fake_engine(monkeypatch)
        monkeypatch.setattr(oci_module, "_host_ids", lambda: (1000, 1000))
        calls: list[list[str]] = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            if argv[1] == "run":
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            if argv[1] == "rm":
                return rm(argv)
            return inspect(argv)

        _fake_subprocess(monkeypatch, fake_run)
        config = _config(tmp_path, timeout_seconds=5)
        with pytest.raises(ContainmentCleanupError, match="could not be proven removed") as info:
            OciBackend().execute(config)
        assert info.value.artifact_dir == config.artifact_dir
        # Removal was retried before giving up.
        assert sum(1 for c in calls if c[1] == "rm") == oci_module._REMOVE_ATTEMPTS

    def test_removal_retried_until_proven(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        monkeypatch.setattr(oci_module, "_host_ids", lambda: (1000, 1000))
        inspections = iter([
            subprocess.CompletedProcess([], 0, "abc123\n", ""),
            subprocess.CompletedProcess([], 1, "", "Error: no such container stele-x\n"),
        ])

        def fake_run(argv, **kwargs):
            if argv[1] == "run":
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            if argv[1] == "rm":
                return subprocess.CompletedProcess(argv, 0, "", "")
            return next(inspections)

        _fake_subprocess(monkeypatch, fake_run)
        assert OciBackend().execute(_config(tmp_path, timeout_seconds=5)).timed_out

    def test_outcome_records_image_digest(self, tmp_path: Path, monkeypatch) -> None:
        _fake_engine(monkeypatch)
        monkeypatch.setattr(oci_module, "_host_ids", lambda: (1000, 1000))
        _fake_subprocess(
            monkeypatch, lambda argv, **kw: subprocess.CompletedProcess(argv, 3, "o", "e")
        )
        outcome = OciBackend(image="python:3.12-slim").execute(_config(tmp_path))
        assert (outcome.exit_code, outcome.stdout, outcome.stderr) == (3, "o", "e")
        assert outcome.image_digest == DIGEST


class TestRegistry:

    def test_order_and_names(self) -> None:
        names = [b.name for b in backend_module.default_backends()]
        assert names == ["bubblewrap", "oci-runc", "oci-runc-gpu", "wasmtime"]

    def test_gpu_parser_goes_to_gpu_variant_only(self, monkeypatch) -> None:
        _fake_engine(monkeypatch, info=dict(DOCKER_INFO, Runtimes={"runc": {}, "nvidia": {}}))
        chosen = select_backend(ParserRequirements(requires_gpu=True))
        assert chosen.name == "oci-runc-gpu"

    def test_plain_parser_never_gets_gpu_variant(self, monkeypatch) -> None:
        _fake_engine(monkeypatch, info=dict(DOCKER_INFO, Runtimes={"runc": {}, "nvidia": {}}))
        monkeypatch.setattr(backend_module.BubblewrapBackend, "available", lambda self: False)
        assert select_backend(ParserRequirements()).name == "oci-runc"


# ---------------------------------------------------------------------------
# Live — every available backend
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("containment_backend")
class TestLiveBackendProperties:

    PROBE = (
        "import json, os\n"
        "open(os.path.join(os.environ['STELE_OUTPUT_DIR'], 'who.json'), 'w')"
        ".write(json.dumps({'uid': os.getuid()}))\n"
    )

    def test_artifacts_belong_to_invoking_user(
        self, tmp_path: Path, containment_backend, sandbox_python: str
    ) -> None:
        out = tmp_path / "out"
        result = run_in_sandbox(SandboxConfig(
            command=[sandbox_python, "-c", self.PROBE], artifact_dir=out,
        ))
        assert result.succeeded, result.stderr
        artifact = out / "who.json"
        if hasattr(os, "getuid"):
            assert os.stat(artifact).st_uid == os.getuid()
            assert os.stat(out).st_uid == os.getuid()
        if isinstance(containment_backend, OciBackend):
            assert json.loads(artifact.read_text())["uid"] != 0, "parser ran as root"
            assert result.image_digest == containment_backend.image_digest
            assert result.image_digest and result.image_digest.startswith("sha256:")
        else:
            assert result.image_digest is None

    def test_timeout_stops_the_parser(
        self, tmp_path: Path, containment_backend, sandbox_python: str
    ) -> None:
        result = run_in_sandbox(SandboxConfig(
            command=[sandbox_python, "-c", "import time; time.sleep(120)"],
            artifact_dir=tmp_path / "out",
            timeout_seconds=3,
        ))
        assert result.timed_out and not result.succeeded
        if isinstance(containment_backend, OciBackend):
            engine = containment_backend.probe().engine
            leftover = subprocess.run(
                [engine, "ps", "-a", "-q", "--filter", "name=^stele-"],
                capture_output=True, text=True, timeout=30,
            )
            assert leftover.stdout.strip() == "", "timed-out container left behind"
