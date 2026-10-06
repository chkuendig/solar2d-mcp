"""
Process ownership and single-slot coordination for the Solar2D simulator.

Each stdio MCP server remains usable while another client owns the simulator:
simulator-dependent tools return a clear busy response rather than making the
connection fail. Only processes recorded by this server are ever terminated.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

from utils import running_projects

_RUNTIME_DIR = Path(
    os.environ.get("SOLAR2D_MCP_RUNTIME_DIR", Path(tempfile.gettempdir()) / "solar2d-mcp")
)
_LOCK_PATH = _RUNTIME_DIR / "simulator.lock"
_OWNER_PATH = _RUNTIME_DIR / "simulator-owner.json"
# Where the simulator's frame tap (SOLAR2D_VIDEO_PIPE) streams framed BGRA.
# The engine creates the FIFO itself; this path is only named here so the
# launcher and the video relay agree on it.
_VIDEO_PIPE_PATH = _RUNTIME_DIR / "video.fifo"
_slot_fd: int | None = None

# Recordings finalized by a relaunch, keyed by project directory, waiting to be
# attached to the next launch so stop_video_recording can still report them.
_unreported_recordings: dict[str, dict] = {}


def take_finished_recording(project_dir: str) -> dict | None:
    """Claim the recording a relaunch finalized for this project, if any."""
    return _unreported_recordings.pop(project_dir, None)


def _try_lock(fd: int) -> bool:
    """Take a non-blocking, process-scoped lock on every supported host."""
    if os.name == "nt":
        import msvcrt

        if os.fstat(fd).st_size == 0:
            os.write(fd, b"0")
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return

    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


def _write_owner() -> None:
    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    pending = _OWNER_PATH.with_suffix(".pending")
    pending.write_text(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
    pending.replace(_OWNER_PATH)


def acquire_simulator_slot() -> bool:
    """Claim the single simulator slot without blocking the MCP transport."""
    global _slot_fd

    if _slot_fd is not None:
        return True

    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    if not _try_lock(fd):
        os.close(fd)
        return False

    _slot_fd = fd
    _write_owner()
    return True


def simulator_busy_message() -> str | None:
    """Return None when this server owns the slot, otherwise a useful retry cue."""
    if acquire_simulator_slot():
        return None

    owner = "another MCP client"
    try:
        data = json.loads(_OWNER_PATH.read_text())
        pid = data.get("pid")
        age = max(0, int(time.time() - float(data.get("started_at", time.time()))))
        owner = f"MCP client pid {pid} (running for {age}s)"
    except (OSError, ValueError, TypeError):
        pass

    return (
        f"Solar2D runtime is busy: {owner} owns the single simulator slot. "
        "The MCP connection is healthy; retry this tool after that client disconnects."
    )


def release_simulator_slot() -> None:
    """Release our lease without disturbing any other MCP server."""
    global _slot_fd

    if _slot_fd is None:
        return
    try:
        _OWNER_PATH.unlink(missing_ok=True)
        _unlock(_slot_fd)
    finally:
        os.close(_slot_fd)
        _slot_fd = None


def _stop_process(process: Any, timeout: float = 2.0) -> None:
    """Terminate one process group created by this server, then reap it."""
    if process.poll() is not None:
        return

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except (ProcessLookupError, OSError):
        return

    try:
        process.wait(timeout=timeout)
        return
    except Exception:
        pass

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, OSError):
        return
    try:
        process.wait(timeout=timeout)
    except Exception:
        pass


def _finish_recording_process(process: Any, timeout: float = 8.0) -> None:
    """Ask ffmpeg to finalize its container before falling back to termination."""
    if process.poll() is not None:
        return

    stdin = getattr(process, "stdin", None)
    if stdin is None:
        _stop_process(process)
        return
    try:
        try:
            stdin.write(b"q\n")
        except TypeError:
            stdin.write("q\n")
        stdin.flush()
        process.wait(timeout=timeout)
    except Exception:
        _stop_process(process)
    finally:
        try:
            stdin.close()
        except OSError:
            pass


def stop_tracked_simulators() -> None:
    """Stop tracked processes and run their launch-scoped file cleanup.

    A recording still active on a tracked launch is finalized here and kept for
    the next launch of the same project to report, so ending a simulator never
    silently swallows a finished MP4.
    """
    for project in list(running_projects.values()):
        recording = project.pop("video_recording", None)
        if recording is not None:
            finalizer = recording.get("finalize")
            if callable(finalizer):
                finalizer(recording)
            else:
                _finish_recording_process(recording["process"])
            log_handle = recording.get("log_handle")
            if log_handle is not None and not log_handle.closed:
                log_handle.close()
            project_dir = project.get("project_dir")
            if project_dir is not None:
                _unreported_recordings[project_dir] = recording
        process = project.get("process")
        if process is not None:
            _stop_process(process)
        cleanup_files = project.get("cleanup_files")
        if callable(cleanup_files):
            try:
                cleanup_files(project)
            except Exception:
                # Shutdown must still clear ownership and release the shared
                # simulator lease when a project path has become inaccessible.
                pass
    running_projects.clear()


def shutdown_runtime() -> None:
    """Cleanly end this server's simulator and release its cross-client lease."""
    stop_tracked_simulators()
    for recording in _unreported_recordings.values():
        log_path = recording.get("log_path")
        if log_path:
            Path(log_path).unlink(missing_ok=True)
    _unreported_recordings.clear()
    release_simulator_slot()


atexit.register(shutdown_runtime)
