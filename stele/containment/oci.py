"""
OCI container backend (roadmap #8).

Runs parsers in a Podman or Docker container. On Linux the container shares
the host kernel (runc) or runs on gVisor's user-space kernel (runsc, opt-in).
On macOS and Windows the engine's Linux VM is an additional boundary.

The container presents exactly the bubblewrap layout: the staged input at
/stele/input/<name>, the parser at /stele/parser, and /stele/output as the only
durable writable path, with STELE_INPUT_PATH / STELE_OUTPUT_DIR set. Parsers are
therefore portable across backends as long as their command resolves inside the
image (e.g. ``python3`` rather than a host interpreter path).

Hardening applied to every run: no network, read-only root filesystem, every
capability dropped, no-new-privileges, a non-root user, memory/CPU/PID limits,
ephemeral tmpfs scratch space, and a named container that is force-removed on
timeout. Images are never pulled here: fetching is a separate trusted step, and
the backend reports itself unavailable until the image is present locally.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .backend import (
    Capability,
    ContainmentCleanupError,
    ExecutionOutcome,
    SandboxBackend,
    SandboxUnavailableError,
)
from .capture import DEFAULT_OUTPUT_LIMIT_BYTES, run_bounded
from .sandbox import (
    _EPHEMERAL_TMPFS,
    SANDBOX_INPUT_DIR,
    SANDBOX_OUTPUT,
    SANDBOX_SCRIPT,
    SandboxConfig,
)
from .telemetry import (
    FailureReason,
    RunFailure,
    exit_status_failure,
    signal_failure,
    timeout_failure,
)

# Official Python image, pinned by (multi-arch index) digest. The containment
# fixtures are Python scripts; real parsers bring their own pinned image.
DEFAULT_OCI_IMAGE = (
    "docker.io/library/python:3.12-slim"
    "@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9"
)

# Engines in preference order: rootless Podman first, then Docker.
ENGINES = ("podman", "docker")

# Runtimes whose GPU passthrough Stele can vouch for. gVisor (runsc) supports
# GPUs only through nvproxy for specific drivers, which Stele cannot verify, so
# GPU is reported unavailable under runsc rather than silently dropped.
_GPU_CAPABLE_RUNTIMES = frozenset({"runc", "crun"})

# Unprivileged uid/gid (nobody) used when Stele itself runs as host root, so the
# parser is never uid 0 inside the container.
_NOBODY = 65534

_PROBE_TIMEOUT = 30
_LIMIT_CONTROLLERS = frozenset({"memory", "pids", "cpu"})
_DIGEST_RE = re.compile(r"@(sha256:[0-9a-f]{64})$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class EngineInfo:
    """What the container engine reports about itself."""

    kind: str                   # "docker" or "podman" (by info shape, not binary name)
    os_type: str                # server OS; must be "linux"
    rootless: bool
    seccomp: bool               # default seccomp profile active
    runtimes: frozenset[str]    # OCI runtimes the engine can use by name
    resource_limits: bool       # memory, CPU and PID limits are enforced, not ignored
    gpu_flags: tuple[str, ...]  # run flags that pass NVIDIA GPUs through; () if none


def parse_engine_info(raw: dict) -> EngineInfo:
    """Normalize `<engine> info --format '{{json .}}'` for Docker or Podman."""
    if "host" in raw:  # Podman
        host = raw.get("host") or {}
        security = host.get("security") or {}
        runtime = (host.get("ociRuntime") or {}).get("name")
        return EngineInfo(
            kind="podman",
            os_type=str(host.get("os", "")),
            rootless=bool(security.get("rootless")),
            seccomp=bool(security.get("seccompEnabled")),
            runtimes=frozenset({runtime} if runtime else set()),
            # Rootless Podman on cgroup v1 (or without delegated controllers)
            # accepts --memory/--cpus/--pids-limit but silently ignores them.
            resource_limits=_LIMIT_CONTROLLERS <= set(host.get("cgroupControllers") or []),
            gpu_flags=("--device", "nvidia.com/gpu=all") if _cdi_has_nvidia_gpu() else (),
        )

    options = [str(o) for o in raw.get("SecurityOptions") or []]
    seccomp = any(
        o.startswith("name=seccomp") and "profile=unconfined" not in o for o in options
    )
    runtimes = frozenset((raw.get("Runtimes") or {}).keys())
    cdi_gpu = any(
        str(d.get("ID", "")).startswith("nvidia.com/gpu")
        for d in raw.get("DiscoveredDevices") or []
        if isinstance(d, dict)
    )
    if cdi_gpu:
        gpu_flags: tuple[str, ...] = ("--device", "nvidia.com/gpu=all")
    elif "nvidia" in runtimes:
        gpu_flags = ("--gpus", "all")
    else:
        gpu_flags = ()
    return EngineInfo(
        kind="docker",
        os_type=str(raw.get("OSType", "")),
        rootless=any(o == "name=rootless" for o in options),
        seccomp=seccomp,
        runtimes=runtimes,
        resource_limits=all(raw.get(k) for k in ("MemoryLimit", "PidsLimit", "CpuCfsQuota")),
        gpu_flags=gpu_flags,
    )


def _cdi_has_nvidia_gpu() -> bool:
    """True when a CDI spec on this host declares NVIDIA GPU devices."""
    for spec_dir in ("/etc/cdi", "/var/run/cdi"):
        try:
            entries = list(Path(spec_dir).iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.suffix not in (".json", ".yaml", ".yml"):
                continue
            try:
                if "nvidia.com/gpu" in entry.read_text(errors="replace"):
                    return True
            except OSError:
                continue
    return False


@dataclass(frozen=True)
class _Probe:
    engine: str | None = None
    info: EngineInfo | None = None
    image_id: str | None = None
    image_digest: str | None = None
    problem: str | None = None


class OciBackend(SandboxBackend):
    """Podman or Docker container with a hardened, read-only configuration.

    runtime="runc" is the default. runtime="runsc" (gVisor) is opt-in and is
    never silently replaced: if the engine does not have it, the backend is
    unavailable with that reason. gpu=True passes NVIDIA GPUs through and makes
    the backend unavailable (never GPU-less) when that cannot be done.
    """

    def __init__(
        self,
        *,
        image: str = DEFAULT_OCI_IMAGE,
        engine: str | None = None,
        runtime: str = "runc",
        gpu: bool = False,
        memory: str = "2g",
        cpus: float = 2.0,
        pids_limit: int = 512,
        tmpfs_size: str = "256m",
        output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
    ) -> None:
        if engine is not None and engine not in ENGINES:
            raise ValueError(f"unsupported container engine {engine!r}; expected one of {ENGINES}")
        self.image = image
        self.engine = engine
        self.runtime = runtime
        self.gpu = gpu
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.tmpfs_size = tmpfs_size
        # Kept of each of the parser's stdout and stderr; the rest is discarded.
        self.output_limit_bytes = output_limit_bytes
        self.name = f"oci-{runtime}" + ("-gpu" if gpu else "")
        self._probe: _Probe | None = None

    # -- probing ------------------------------------------------------------

    def probe(self, *, refresh: bool = False) -> _Probe:
        """Inspect the engine and image once; cached until refresh=True."""
        if self._probe is None or refresh:
            self._probe = self._run_probe()
        return self._probe

    def _run_probe(self) -> _Probe:
        engines = self._find_engines()
        if not engines:
            wanted = self.engine or " or ".join(ENGINES)
            return _Probe(problem=f"no container engine found ({wanted} not on PATH)")
        # Preference order; fall through to the next engine only if one cannot
        # run this backend (e.g. the image is in Docker's store, not Podman's).
        probes = [self._probe_engine(engine) for engine in engines[:1]]
        for engine in engines[1:]:
            if probes[-1].problem is None:
                break
            probes.append(self._probe_engine(engine))
        if probes[-1].problem is None or len(probes) == 1:
            return probes[-1]
        return replace(probes[0], problem="; ".join(
            f"{Path(p.engine or '?').name}: {p.problem}" for p in probes
        ))

    def _probe_engine(self, engine: str) -> _Probe:
        info_raw = _run_json([engine, "info", "--format", "{{json .}}"])
        if isinstance(info_raw, str):
            return _Probe(engine=engine, problem=f"`{engine} info` failed: {info_raw}")
        if not isinstance(info_raw, dict):
            return _Probe(engine=engine, problem=f"`{engine} info` returned unexpected output")
        info = parse_engine_info(info_raw)

        if info.os_type != "linux":
            return _Probe(engine=engine, info=info, problem=(
                f"{engine} runs {info.os_type or 'non-Linux'} containers; "
                "Stele parsers need a Linux container engine"
            ))
        if info.kind == "docker" and info.rootless:
            return _Probe(engine=engine, info=info, problem=(
                "rootless Docker maps container users to subordinate uids, so "
                "output ownership cannot be preserved; use rootless Podman instead"
            ))
        if not self._runtime_known(info):
            known = ", ".join(sorted(info.runtimes)) or "none reported"
            return _Probe(engine=engine, info=info, problem=(
                f"OCI runtime {self.runtime!r} is not configured in {engine} "
                f"(available: {known}); Stele will not substitute another runtime"
            ))
        if self.gpu:
            gpu_problem = self._gpu_problem(info)
            if gpu_problem:
                return _Probe(engine=engine, info=info, problem=gpu_problem)

        image = _run_json([engine, "image", "inspect", self.image])
        if isinstance(image, str) or not image or not isinstance(image, list):
            return _Probe(engine=engine, info=info, problem=(
                f"image {self.image} is not present locally; fetch it in the "
                f"trusted acquisition step (`{engine} pull {self.image}`) — "
                "Stele never pulls at run time"
            ))
        record = image[0] if isinstance(image[0], dict) else {}
        image_id = str(record.get("Id") or record.get("ID") or "") or None
        if image_id is None:
            return _Probe(engine=engine, info=info, problem=f"`{engine} image inspect` reported no image id")
        return _Probe(
            engine=engine,
            info=info,
            image_id=image_id,
            image_digest=_image_digest(self.image, record.get("RepoDigests") or [], image_id),
        )

    def _find_engines(self) -> list[str]:
        candidates = (self.engine,) if self.engine else ENGINES
        return [path for path in map(shutil.which, candidates) if path]

    def _runtime_known(self, info: EngineInfo) -> bool:
        if self.runtime in info.runtimes:
            return True
        # Podman reports only its default runtime but accepts any runtime
        # binary it can find by name.
        return info.kind == "podman" and shutil.which(self.runtime) is not None

    def _gpu_problem(self, info: EngineInfo) -> str | None:
        if self.runtime not in _GPU_CAPABLE_RUNTIMES:
            return (
                f"no usable GPU under {self.runtime}: gVisor GPU support (nvproxy) "
                "cannot be verified, so GPU and gVisor are not combined"
            )
        if not info.gpu_flags:
            return (
                "no usable GPU: the container engine exposes no NVIDIA GPU "
                "(no CDI nvidia.com/gpu device or nvidia runtime)"
            )
        return None

    # -- SandboxBackend -----------------------------------------------------

    def capabilities(self) -> frozenset[Capability]:
        caps = {
            Capability.FILESYSTEM_ISOLATION,
            Capability.NETWORK_ISOLATION,
            Capability.HOST_PROCESS,
            Capability.NATIVE_LIBS,
        }
        # A GPU variant hosts GPU workloads unless its runtime rules GPUs out
        # (gVisor). Whether this host actually has a usable GPU is checked by
        # available(), so a GPU parser on a GPU-less host gets
        # SandboxUnavailableError naming the missing GPU, never a GPU-less run.
        if self.gpu and self.runtime in _GPU_CAPABLE_RUNTIMES:
            caps.add(Capability.GPU)
        probe = self.probe()
        if probe.problem is None and probe.info is not None:
            # gVisor intercepts every application syscall in its own kernel and
            # runs that kernel under a host seccomp filter. Under runc/crun the
            # claim rests on the engine's default seccomp profile being active.
            if self.runtime == "runsc" or probe.info.seccomp:
                caps.add(Capability.SYSCALL_FILTER)
            # The limits are always requested; claimed only where the engine
            # reports that its cgroup setup enforces them.
            if probe.info.resource_limits:
                caps.add(Capability.RESOURCE_LIMITS)
        return frozenset(caps)

    def available(self) -> bool:
        return self.probe().problem is None

    def unavailable_reason(self) -> str:
        problem = self.probe().problem or "available"
        return f"OCI container backend ({self.name}): {problem}"

    @property
    def image_digest(self) -> str | None:
        """Content digest of the image this backend runs (parser identity, #12)."""
        return self.probe().image_digest

    def build_argv(self, config: SandboxConfig, *, container_name: str, user: tuple[int, int] | None) -> list[str]:
        """The engine argv for one run. user=None means rootless Podman keep-id."""
        probe = self.probe()
        if probe.problem is not None or probe.info is None or probe.engine is None:
            raise SandboxUnavailableError(self.unavailable_reason())
        info = probe.info
        if not config.command:
            raise ValueError("SandboxConfig.command is empty")

        argv = [
            probe.engine, "run",
            "--rm",
            "--name", container_name,
            "--pull", "never",
            "--runtime", self.runtime,
            # Isolation
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--ipc", "private",
            # Resource limits
            "--memory", self.memory,
            "--memory-swap", self.memory,  # no swap beyond the memory cap
            "--cpus", str(self.cpus),
            "--pids-limit", str(self.pids_limit),
        ]

        if info.kind == "podman":
            # Only the tmpfs mounts below; podman would otherwise add its own.
            argv += ["--read-only-tmpfs=false"]
            # Belt and braces: conmon kills the container even if Stele dies.
            argv += ["--timeout", str(max(1, int(config.timeout_seconds)))]

        if user is None:
            argv += ["--userns", "keep-id"]
        else:
            argv += ["--user", f"{user[0]}:{user[1]}"]

        if self.gpu:
            argv += list(info.gpu_flags)

        # Ephemeral scratch space, mirroring the bubblewrap layout. Without
        # writable_scratch the same (empty) mounts are read-only, so
        # /stele/output is the only writable path, as bubblewrap's Landlock
        # rules make it there.
        access = "rw" if config.writable_scratch else "ro"
        for path in ["/tmp", *_EPHEMERAL_TMPFS]:
            argv += ["--tmpfs", f"{path}:{access},nosuid,nodev,size={self.tmpfs_size},mode=1777"]

        # Writable artifact output: the ONLY path that survives the container.
        config.artifact_dir.mkdir(parents=True, exist_ok=True)
        argv += ["--mount", _bind(info, str(config.artifact_dir), SANDBOX_OUTPUT, readonly=False)]

        env: dict[str, str] = {"STELE_OUTPUT_DIR": SANDBOX_OUTPUT}
        if config.input_path is not None:
            input_dest = f"{SANDBOX_INPUT_DIR}/{config.input_path.name}"
            argv += ["--mount", _bind(info, str(config.input_path), input_dest, readonly=True)]
            env["STELE_INPUT_PATH"] = input_dest
        if config.script_path is not None:
            argv += ["--mount", _bind(info, str(config.script_path), SANDBOX_SCRIPT, readonly=True)]
        for src, dst in config.extra_ro_binds:
            # Passed through as given: a host path string, never re-rendered.
            argv += ["--mount", _bind(info, src, dst, readonly=True)]

        # The image's own environment (PATH etc.) is kept; the host's never is.
        env.update(config.env)
        for key, val in env.items():
            if not _ENV_KEY_RE.match(key):
                raise ValueError(f"invalid environment variable name: {key!r}")
            argv += ["--env", f"{key}={val}"]

        argv += ["--workdir", SANDBOX_OUTPUT]
        # Run exactly config.command, whatever ENTRYPOINT the image declares.
        # Pinned by local image id so the inspected image is the one that runs.
        argv += ["--entrypoint", config.command[0], probe.image_id or self.image]
        argv += config.command[1:]
        return argv

    def execute(self, config: SandboxConfig) -> ExecutionOutcome:
        probe = self.probe()
        if probe.problem is not None or probe.engine is None or probe.info is None:
            raise SandboxUnavailableError(self.unavailable_reason())

        rootless_podman = probe.info.kind == "podman" and probe.info.rootless
        host_uid, host_gid = _host_ids()
        if rootless_podman:
            user: tuple[int, int] | None = None
        elif host_uid == 0:
            user = (_NOBODY, _NOBODY)
        else:
            user = (host_uid, host_gid)

        container_name = f"stele-{uuid4().hex}"
        config.artifact_dir.mkdir(parents=True, exist_ok=True)
        handoff = _RootHandoff(config) if user == (_NOBODY, _NOBODY) and host_uid == 0 else None
        if handoff is not None:
            handoff.give()
        try:
            argv = self.build_argv(config, container_name=container_name, user=user)
            outcome = self._run(argv, probe.engine, container_name, config.timeout_seconds)
        except ContainmentCleanupError as exc:
            exc.artifact_dir = config.artifact_dir
            raise
        finally:
            if handoff is not None:
                handoff.take_back()
        return outcome

    def _run(self, argv: list[str], engine: str, container_name: str, timeout: int) -> ExecutionOutcome:
        # The engine runs from a private, empty working directory: Podman's
        # conmon writes an "oom" marker file into its working directory when
        # the container is OOM-killed, which would otherwise land in the
        # caller's current directory.
        with TemporaryDirectory(prefix="stele-engine-") as workdir:
            return self._run_in(argv, engine, container_name, timeout, workdir)

    def _run_in(
        self, argv: list[str], engine: str, container_name: str, timeout: int, workdir: str
    ) -> ExecutionOutcome:
        t0 = time.monotonic()
        try:
            # Bounded: the engine CLI relays the parser's output into this
            # process, outside the container's memory limit.
            proc = run_bounded(argv, timeout=timeout, limit=self.output_limit_bytes, cwd=workdir)
        except FileNotFoundError as exc:
            if exc.filename not in (None, argv[0]):
                raise
            raise SandboxUnavailableError(self.unavailable_reason()) from exc
        if proc.timed_out:
            # Killing the CLI client does not stop the container; remove it,
            # and report the run as stopped only once the engine confirms the
            # container no longer exists.
            problem = _force_remove(engine, container_name)
            if problem is not None:
                raise ContainmentCleanupError(
                    f"the parser timed out after {timeout}s, but container "
                    f"{container_name} could not be proven removed ({problem}); it may "
                    "still be running and writing to the output directory, which must "
                    "not be used"
                )
            return self._outcome(
                -1, proc.stdout, proc.stderr, time.monotonic() - t0,
                engine, timeout, timed_out=True,
            )
        return self._outcome(
            proc.returncode, proc.stdout, proc.stderr, time.monotonic() - t0, engine, timeout,
        )

    def _outcome(
        self, exit_code: int, stdout: str, stderr: str, wall: float, engine: str,
        timeout: int, *, timed_out: bool = False,
    ) -> ExecutionOutcome:
        return ExecutionOutcome(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            wall_time_seconds=wall,
            timed_out=timed_out,
            image_digest=self.image_digest,
            # The container runs with --rm, so the engine's CPU and memory
            # accounting is gone when it exits: not measurable here.
            runtime=f"{Path(engine).name}/{self.runtime}",
            limits={
                "timeout_seconds": timeout,
                "memory": self.memory,
                "cpus": self.cpus,
                "pids": self.pids_limit,
                "tmpfs": self.tmpfs_size,
                "gpu": self.gpu,
                "output_bytes": self.output_limit_bytes,
            },
            failure=self._classify(exit_code, stderr, wall, timeout, timed_out),
        )

    def _classify(
        self, exit_code: int, stderr: str, wall: float, timeout: int, timed_out: bool
    ) -> RunFailure | None:
        if timed_out:
            return timeout_failure(timeout, exit_code, "the container was force-removed")
        if exit_code == 0:
            return None
        if exit_code == _SIGKILL_EXIT and wall >= timeout - 0.5:
            # Podman's conmon enforces the same deadline with SIGKILL and can
            # get there first.
            return timeout_failure(timeout, exit_code, "the engine killed the container")
        if exit_code == _SIGKILL_EXIT:
            # Inside a memory-limited container, SIGKILL is the kernel's OOM
            # killer (or gVisor's) at the cgroup limit.
            return RunFailure(
                FailureReason.OUT_OF_MEMORY,
                f"killed by SIGKILL at the {self.memory} container memory limit",
                exit_code=exit_code, signal=9,
            )
        if exit_code in _ENGINE_ERRORS and _engine_failed(exit_code, stderr):
            last = _last_line(stderr)
            return RunFailure(
                FailureReason.ENGINE_ERROR,
                _ENGINE_ERRORS[exit_code] + (f": {last}" if last else ""),
                exit_code=exit_code,
            )
        if 128 < exit_code <= 128 + 64:
            return signal_failure(exit_code - 128, exit_code)
        return exit_status_failure(exit_code)


# docker/podman run exit statuses that mean the engine, not the parser, failed.
_ENGINE_ERRORS = {
    125: "the container engine could not run the container",
    126: "the parser command could not be executed in the image",
    127: "the parser command was not found in the image",
}
_SIGKILL_EXIT = 128 + 9

# The same statuses are ordinary exit codes for the parser itself, so they are
# an engine error only when the engine or OCI runtime says so at the end of stderr. A
# failed exec of the parser command (126/127) is always reported by the
# runtime ("OCI runtime ... executable file not found", "unable to start
# container process"); for 125 the engines' own "Error: ..." (podman) or
# "docker: ..." prefix is enough, since the engine exits 125 for many
# different messages. A parser can still imitate these lines, but then it
# only misreports its own failure.
_RUNTIME_ERROR_RE = re.compile(
    r"OCI runtime|OCI permission denied|unable to start container process"
    r"|Error response from daemon|executable file not found in \$PATH"
)
_ENGINE_PREFIX_RE = re.compile(r"^(?:Error: |docker: )")
_ENGINE_TAIL_LINES = 3


def _engine_failed(exit_code: int, stderr: str) -> bool:
    # Docker follows its error with "Run 'docker run --help' for more
    # information", so look at the last few lines, not only the last.
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    tail = lines[-_ENGINE_TAIL_LINES:]
    if any(_RUNTIME_ERROR_RE.search(line) for line in tail):
        return True
    return exit_code == 125 and any(_ENGINE_PREFIX_RE.match(line) for line in tail)


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:300] if lines else ""


class _RootHandoff:
    """When Stele runs as host root, the parser runs as nobody.

    The private staged input and the output directory are handed to nobody for
    the run and returned afterwards, so the parser can read its input and write
    artifacts while Stele's view of ownership is unchanged. Never follows
    symlinks. The parser script and extra binds must already be world-readable.
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._output = config.artifact_dir
        out_stat = os.lstat(self._output)
        self._owner = (out_stat.st_uid, out_stat.st_gid)
        self._input = config.input_path
        self._input_owners: dict[str, tuple[int, int]] = {}

    def give(self) -> None:
        os.lchown(self._output, _NOBODY, _NOBODY)
        if self._input is not None:
            for path in _walk_no_follow(self._input):
                st = os.lstat(path)
                self._input_owners[path] = (st.st_uid, st.st_gid)
                os.lchown(path, _NOBODY, _NOBODY)

    def take_back(self) -> None:
        for path in _walk_no_follow(self._output):
            os.lchown(path, *self._owner)
        for path, owner in self._input_owners.items():
            try:
                os.lchown(path, *owner)
            except FileNotFoundError:
                pass


def _walk_no_follow(root: Path) -> list[str]:
    paths = [str(root)]
    if os.path.isdir(root) and not os.path.islink(root):
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            paths += [os.path.join(dirpath, n) for n in dirnames + filenames]
    return paths


def _host_ids() -> tuple[int, int]:
    """The invoking uid:gid; nobody where the host has no POSIX ids (Windows)."""
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        return os.getuid(), os.getgid()
    return _NOBODY, _NOBODY


def _bind(info: EngineInfo, src: str, dst: str, *, readonly: bool) -> str:
    """A --mount value. Refuses characters that would change the mount spec."""
    fields = ["type=bind", f"source={src}", f"target={dst}"]
    if readonly:
        fields.append("readonly")
    for field in fields:
        if any(ch in field for ch in (",", '"', "\n", "\r", "\0")):
            raise ValueError(
                f"path cannot be passed to {info.kind} --mount safely "
                f"(contains a comma, quote or control character): {field}"
            )
    return ",".join(fields)


def _image_digest(ref: str, repo_digests: list, image_id: str) -> str:
    """The pinned digest when the reference has one, else the best local digest."""
    pinned = _DIGEST_RE.search(ref)
    if pinned:
        return pinned.group(1)
    for entry in repo_digests:
        match = _DIGEST_RE.search(str(entry))
        if match:
            return match.group(1)
    return image_id if image_id.startswith("sha256:") else f"sha256:{image_id}"


def _run_json(argv: list[str]) -> object:
    """Run a probe command; parsed JSON on success, an error string otherwise."""
    try:
        proc = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        return detail[-1] if detail else f"exit {proc.returncode}"
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return f"unparseable output: {exc}"


_REMOVE_ATTEMPTS = 3


def _force_remove(engine: str, container_name: str) -> str | None:
    """Force-remove a container; None once the engine confirms it is gone.

    Otherwise returns why its absence could not be proven: the removal
    failed and the container still exists, or the engine could not say.
    """
    problem = "not attempted"
    for _ in range(_REMOVE_ATTEMPTS):
        try:
            rm = subprocess.run(
                [engine, "rm", "-f", container_name],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT,
            )
            problem = (
                f"{engine} rm exited {rm.returncode}: {_tail(rm.stderr)}"
                if rm.returncode != 0 else "removal reported success"
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            problem = f"{engine} rm failed: {exc}"
        absent, why = _container_absent(engine, container_name)
        if absent:
            return None
        problem = f"{problem}; {why}"
    return problem


def _container_absent(engine: str, container_name: str) -> tuple[bool, str]:
    """(True, "") only when the engine reports no container by that exact name."""
    try:
        proc = subprocess.run(
            [engine, "container", "inspect", "--format", "{{.Id}}", container_name],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{engine} container inspect failed: {exc}"
    if proc.returncode == 0:
        return False, "the container still exists"
    if "no such" in (proc.stderr or "").lower():
        return True, ""
    return False, f"{engine} container inspect exited {proc.returncode}: {_tail(proc.stderr)}"


def _tail(text: str | None) -> str:
    lines = (text or "").strip().splitlines()
    return lines[-1] if lines else "no output"
