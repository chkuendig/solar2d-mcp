"""Relay-based offscreen video recording for the Solar2D simulator.

Frames arrive over the engine's SOLAR2D_VIDEO_PIPE tap (a FIFO carrying
self-describing BGRA frames) and are relayed into ffmpeg's stdin. The FIFO
itself is only ever opened by the relay: opening it connects a reader and
starts the stream, so capability and liveness are probed through the
`<fifo>.ready` marker file instead.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import select
import shutil
import struct
import subprocess
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from mcp.types import TextContent, Tool

from runtime import _VIDEO_PIPE_PATH
from runtime import _stop_process as stop_process
from utils import find_main_lua, get_current_launch, running_projects

MAX_RECORDING_SECONDS = 300
MIN_FPS = 15
MAX_FPS = 60

RELAY_JOIN_TIMEOUT = 10.0
ENCODER_EXIT_TIMEOUT = 8.0

# Wire format of one tap frame: a fixed 64-byte little-endian header followed
# by stride*height bytes of BGRA payload in GL bottom-up row order.
FRAME_HEADER = struct.Struct("<4sHHIII4sB7xQQ16x")
FRAME_MAGIC = b"S2VT"
FRAME_VERSION = 1
FOURCC_BGRA = b"BGRA"

START_VIDEO_RECORDING_TOOL = Tool(
    name="start_video_recording",
    description=(
        "Start a real-time MP4 recording of the Solar2D simulator's display via its "
        "offscreen frame tap. This captures the framebuffer directly instead of stitching "
        "periodic screenshots. Call stop_video_recording when the interaction is complete."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "project_path": {
                "type": "string",
                "description": "Path to the project directory or main.lua file",
            },
            "duration": {
                "type": "number",
                "description": "Safety limit in seconds (default: 30, max: 300)",
                "default": 30,
            },
            "fps": {
                "type": "number",
                "description": "Capture frame rate (default: 30, range: 15-60)",
                "default": 30,
            },
            "filename": {
                "type": "string",
                "description": "Output filename; .mp4 is appended when omitted",
                "default": "recording.mp4",
            },
        },
        "required": ["project_path"],
    },
)

STOP_VIDEO_RECORDING_TOOL = Tool(
    name="stop_video_recording",
    description=(
        "Stop and finalize the current real-time simulator recording. Returns the MP4 path "
        "and verified codec, pixel format, dimensions, frame rate, frame count, duration, "
        "and dropped frames."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "project_path": {
                "type": "string",
                "description": "Path to the project directory or main.lua file",
            }
        },
        "required": ["project_path"],
    },
)

TOOLS = [START_VIDEO_RECORDING_TOOL, STOP_VIDEO_RECORDING_TOOL]


def _find_binary(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for directory in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            return candidate
    return None


def _video_dir(launch: dict[str, Any]) -> Path:
    configured = os.environ.get("SOLAR2D_MCP_ARTIFACT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(launch["screenshot_dir"]) / "video"


def _tap_pid(ready_path: Path) -> int | None:
    """Return the tap owner's pid when its marker points at a live process.

    Only the marker is probed: opening the FIFO would connect a reader and
    start the frame stream, which is the relay's job alone.
    """
    try:
        text = ready_path.read_text()
    except OSError:
        return None
    match = re.fullmatch(r"pid (\d+) version (\d+)\n?", text)
    if match is None or int(match.group(2)) != FRAME_VERSION:
        return None
    pid = int(match.group(1))
    if not Path("/proc", str(pid)).exists():
        return None
    return pid


def _wait_for_frames(fd: int, stop_event: threading.Event) -> bool:
    """Block until the FIFO is readable or the recording is being stopped."""
    while not stop_event.is_set():
        readable, _, _ = select.select([fd], [], [], 0.2)
        if readable:
            return True
    return False


def _read_frame_header(
    fd: int,
    stop_event: threading.Event,
    ready_path: Path,
    tap_pid: int,
) -> tuple[bytes | None, str | None]:
    """Read one frame header; returns (header, end_reason), not both None."""
    buffer = bytearray()
    while len(buffer) < FRAME_HEADER.size:
        if stop_event.is_set():
            return None, "stopped"
        try:
            chunk = os.read(fd, FRAME_HEADER.size - len(buffer))
        except BlockingIOError:
            if not _wait_for_frames(fd, stop_event):
                return None, "stopped"
            continue
        if chunk:
            buffer += chunk
        elif not ready_path.exists() or not Path("/proc", str(tap_pid)).exists():
            return None, "tap-exited"
        else:
            # No writer is connected yet; the tap retries its side ~5x/second.
            time.sleep(0.05)
    return bytes(buffer), None


def _splice_payload(
    fd: int,
    encoder: subprocess.Popen[bytes],
    count: int,
    stop_event: threading.Event,
) -> str | None:
    """Move one frame's payload into ffmpeg's stdin; None on success."""
    moved = 0
    encoder_stdin = encoder.stdin.fileno() if encoder.stdin is not None else -1
    while moved < count:
        if stop_event.is_set():
            return "stopped"
        try:
            chunk = os.splice(fd, encoder_stdin, count - moved)
        except BlockingIOError:
            if not _wait_for_frames(fd, stop_event):
                return "stopped"
            continue
        except OSError:
            return "encoder-input-closed"
        if chunk == 0:
            return "tap-exited"
        moved += chunk
    return None


def _encoder_command(
    ffmpeg: str,
    fps: int,
    duration: int,
    width: int,
    height: int,
    out_path: Path,
) -> list[str]:
    return [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        # The tap sends BGRA; alpha is meaningless, so read it as bgr0.
        "-pixel_format",
        "bgr0",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        # Tap rows are GL bottom-up; vflip, then force even yuv420p dimensions.
        "-vf",
        "vflip,crop=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        # Shallow relay buffering keeps read time close to capture time, so
        # wall-clock timestamps reproduce the real pacing of dropped frames.
        "-use_wallclock_as_timestamps",
        "1",
        "-fps_mode",
        "cfr",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        "-t",
        str(duration),
        str(out_path),
    ]


def _spawn_encoder(recording: dict[str, Any], width: int, height: int) -> subprocess.Popen[bytes] | None:
    try:
        encoder = subprocess.Popen(
            _encoder_command(
                recording["ffmpeg"],
                recording["fps"],
                recording["duration"],
                width,
                height,
                Path(recording["out_path"]),
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=recording["log_handle"],
            start_new_session=True,
        )
    except OSError as exc:
        recording["relay_error"] = f"Could not start ffmpeg: {exc}"
        return None
    recording["segments"].append(recording["out_path"])
    return encoder


def _close_encoder(encoder: subprocess.Popen[bytes]) -> None:
    """Close ffmpeg's stdin so it finalizes the container, then reap it."""
    try:
        if encoder.stdin is not None:
            encoder.stdin.close()
    except OSError:
        pass
    try:
        encoder.wait(timeout=ENCODER_EXIT_TIMEOUT)
    except Exception:
        stop_process(encoder)


def _relay_frames(recording: dict[str, Any]) -> None:
    """Stream framed BGRA frames from the tap FIFO into ffmpeg's stdin."""
    stop_event: threading.Event = recording["stop_event"]
    ready_path = Path(recording["ready_path"])
    try:
        fd = os.open(recording["fifo_path"], os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as exc:
        recording["relay_error"] = f"Could not open the simulator frame tap: {exc}"
        recording["end_reason"] = "tap-open-failed"
        return

    encoder: subprocess.Popen[bytes] | None = None
    end_reason: str | None = None
    try:
        while True:
            header, reason = _read_frame_header(fd, stop_event, ready_path, recording["tap_pid"])
            if header is None:
                end_reason = reason
                break
            magic, version, header_len, width, height, stride, fourcc, bottom_up, seq, _ = FRAME_HEADER.unpack(header)
            if magic != FRAME_MAGIC or version != FRAME_VERSION or header_len != FRAME_HEADER.size:
                recording["relay_error"] = "Frame tap stream lost header alignment; recording ended early."
                end_reason = "misaligned"
                break
            if fourcc != FOURCC_BGRA or bottom_up != 1 or stride != width * 4:
                recording["relay_error"] = (
                    f"Unsupported frame tap payload (fourcc={fourcc!r}, bottom_up={bottom_up}, stride={stride})."
                )
                end_reason = "unsupported-frame"
                break
            if encoder is None:
                encoder = _spawn_encoder(recording, width, height)
                if encoder is None:
                    end_reason = "encoder-start-failed"
                    break
                recording["process"] = encoder
                recording["width"], recording["height"] = width, height
            elif width != recording["width"] or height != recording["height"]:
                # v1 policy: a resize finalizes the segment at the old size
                # rather than scaling or padding across the boundary.
                recording["resized_to"] = f"{width}x{height}"
                end_reason = "resized"
                break
            expected = recording["next_seq"]
            if expected is not None and seq > expected:
                recording["drops"] += seq - expected
            recording["next_seq"] = seq + 1
            reason = _splice_payload(fd, encoder, stride * height, stop_event)
            if reason is not None:
                end_reason = reason
                break
            recording["frames"] += 1
        recording["end_reason"] = end_reason
    finally:
        os.close(fd)
        if encoder is not None:
            _close_encoder(encoder)


def finalize_recording(recording: dict[str, Any]) -> None:
    """End the relay and let ffmpeg finalize its container (idempotent)."""
    if recording.get("finalized"):
        return
    recording["finalized"] = True

    stop_event = recording.get("stop_event")
    if stop_event is not None:
        stop_event.set()
    thread = recording.get("relay_thread")
    if thread is not None and thread.is_alive():
        thread.join(timeout=RELAY_JOIN_TIMEOUT)
        if thread.is_alive():
            # The relay is stuck feeding a wedged encoder; force its exit path.
            process = recording.get("process")
            if process is not None and process.poll() is None:
                stop_process(process)
            thread.join(timeout=2.0)

    log_handle = recording.get("log_handle")
    if log_handle is not None and not log_handle.closed:
        log_handle.close()


def _tail(path: Path, limit: int = 1500) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _probe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe found no video stream")
    stream = streams[0]
    rate = float(Fraction(stream.get("avg_frame_rate", "0/1")))
    return {
        "codec": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": rate,
        "frames": int(stream.get("nb_read_frames") or 0),
        "duration": float((payload.get("format") or {}).get("duration") or 0),
    }


async def handle_start_recording(arguments: dict) -> list[TextContent]:
    project_path = arguments.get("project_path")
    if not project_path:
        return [TextContent(type="text", text="Error: project_path is required")]

    launch, error = get_current_launch(project_path)
    if error:
        return [TextContent(type="text", text=f"Error: {error}")]
    assert launch is not None

    current = launch.get("video_recording")
    if current is not None:
        status = "still running" if current["relay_thread"].is_alive() else "ready to finalize"
        return [TextContent(
            type="text",
            text=f"A video recording is already {status}. Call stop_video_recording before starting another.",
        )]
    if launch.get("finished_video_recording") is not None:
        return [TextContent(
            type="text",
            text=(
                "A recording from before this launch was finalized when the simulator "
                "relaunched. Call stop_video_recording to receive its report, then start "
                "a new recording."
            ),
        )]

    fifo_path = Path(_VIDEO_PIPE_PATH)
    tap_pid = _tap_pid(Path(f"{fifo_path}.ready"))
    if tap_pid is None:
        return [TextContent(
            type="text",
            text=(
                "Real-time recording requires a runtime whose simulator streams frames over "
                "the offscreen video tap. Use screenshot tools for still-image diagnostics on "
                "runtimes without it."
            ),
        )]

    ffmpeg = _find_binary("ffmpeg")
    if not ffmpeg:
        return [TextContent(
            type="text",
            text=(
                "Real-time recording requires ffmpeg on this runtime. "
                "Use screenshot tools for still-image diagnostics without it."
            ),
        )]

    duration = min(MAX_RECORDING_SECONDS, max(1, int(arguments.get("duration", 30))))
    fps = min(MAX_FPS, max(MIN_FPS, int(arguments.get("fps", 30))))
    filename = Path(str(arguments.get("filename", "recording.mp4"))).name
    if not filename.lower().endswith(".mp4"):
        filename += ".mp4"

    out_dir = _video_dir(launch)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    log_path = out_dir / f".{filename}.ffmpeg.log"
    out_path.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)
    log_handle = log_path.open("wb")

    recording: dict[str, Any] = {
        "fifo_path": str(fifo_path),
        "ready_path": f"{fifo_path}.ready",
        "tap_pid": tap_pid,
        "stop_event": threading.Event(),
        "relay_thread": None,
        "process": None,
        "log_handle": log_handle,
        "log_path": str(log_path),
        "out_path": str(out_path),
        "ffmpeg": ffmpeg,
        "fps": fps,
        "duration": duration,
        "width": None,
        "height": None,
        "frames": 0,
        "drops": 0,
        "next_seq": None,
        "end_reason": None,
        "resized_to": None,
        "relay_error": None,
        "segments": [],
        "finalized": False,
    }
    recording["finalize"] = finalize_recording

    relay_thread = threading.Thread(
        target=_relay_frames,
        args=(recording,),
        name="solar2d-video-relay",
        daemon=True,
    )
    recording["relay_thread"] = relay_thread
    relay_thread.start()
    launch["video_recording"] = recording

    await asyncio.sleep(0.25)
    encoder = recording["process"]
    failure = recording["relay_error"] or (
        f"ffmpeg exited during startup (exit {encoder.returncode})"
        if encoder is not None and encoder.poll() is not None
        else None
    )
    if failure is not None:
        launch.pop("video_recording", None)
        await asyncio.to_thread(finalize_recording, recording)
        stderr = _tail(log_path)
        return [TextContent(type="text", text=f"Recording failed during startup: {failure}\n{stderr}")]

    size_note = ""
    if recording["width"] is not None:
        size_note = f"\nFrame size: {recording['width']}x{recording['height']} (from first frame)"
    return [TextContent(type="text", text=(
        "Real-time simulator recording started.\n\n"
        f"Path: {out_path}\n"
        f"Frame tap: {fifo_path} (pid {tap_pid})\n"
        f"Capture: {fps} fps, up to {duration}s, H.264/yuv420p{size_note}\n\n"
        "Drive the simulator now, then call stop_video_recording to finalize and verify the MP4."
    ))]


def _recording_summary(recording: dict[str, Any], probe: dict[str, Any] | None) -> str:
    """Compose the playback-contract lines shared by every stop outcome."""
    lines: list[str] = []
    segments = recording.get("segments") or []
    if len(segments) > 1 or recording.get("end_reason") == "resized":
        lines.append(f"Segments: {', '.join(segments)}")
    if probe is not None:
        lines.append(f"Codec: {probe['codec']} / {probe['pix_fmt']}")
        lines.append(f"Dimensions: {probe['width']}x{probe['height']}")
        lines.append(f"Timeline: {probe['frames']} frames over {probe['duration']:.2f}s @ {probe['fps']:.2f} fps")
    lines.append(f"Tap frames: {recording.get('frames', 0)} ({recording.get('drops', 0)} dropped)")
    return "\n".join(lines)


def _recording_notes(recording: dict[str, Any], code: int) -> list[str]:
    """Human-readable explanations for how this recording ended."""
    notes = []
    end_reason = recording.get("end_reason")
    if recording.get("relay_error"):
        notes.append(recording["relay_error"])
    if end_reason == "resized":
        notes.append(
            "The recording ended early: the window was resized to "
            f"{recording.get('resized_to')} mid-recording, and frames at the new size "
            "were not encoded."
        )
    elif end_reason == "tap-exited":
        notes.append("The frame stream ended when the simulator exited.")
    if code != 0:
        notes.append(f"ffmpeg exited with status {code}.")
    return notes


async def handle_stop_recording(arguments: dict) -> list[TextContent]:
    project_path = arguments.get("project_path")
    if not project_path:
        return [TextContent(type="text", text="Error: project_path is required")]

    launch, error = get_current_launch(project_path)
    if error:
        # ffmpeg can still finalize after the simulator exits; other tools stay
        # on the live-only get_current_launch path.
        project_dir = str(Path(find_main_lua(project_path)).parent)
        stopped_launch = running_projects.get(project_dir)
        process = stopped_launch.get("process") if stopped_launch is not None else None
        if (
            stopped_launch is None
            or (
                stopped_launch.get("video_recording") is None
                and stopped_launch.get("finished_video_recording") is None
            )
            or not stopped_launch.get("launch_id")
            or process is None
            or process.poll() is None
        ):
            return [TextContent(type="text", text=f"Error: {error}")]
        launch = stopped_launch
    assert launch is not None

    recording = launch.pop("video_recording", None)
    finalized_by_relaunch = False
    if recording is None:
        recording = launch.pop("finished_video_recording", None)
        finalized_by_relaunch = recording is not None
    if recording is None:
        return [TextContent(type="text", text="No real-time video recording is active for this launch.")]

    if not recording.get("finalized"):
        await asyncio.to_thread(finalize_recording, recording)

    out_path = Path(recording["out_path"])
    drops = recording.get("drops", 0)
    if recording.get("process") is None:
        Path(recording["log_path"]).unlink(missing_ok=True)
        detail = recording.get("relay_error") or "no frames arrived from the simulator"
        return [TextContent(type="text", text=(
            f"Recording produced no MP4: {detail}.\n"
            f"Dropped frames: {drops}"
        ))]

    process: subprocess.Popen[bytes] = recording["process"]
    code = process.returncode or 0
    stderr = _tail(Path(recording["log_path"]))
    Path(recording["log_path"]).unlink(missing_ok=True)
    notes = _recording_notes(recording, code)
    if finalized_by_relaunch:
        notes.insert(0, "This recording was finalized automatically when the simulator relaunched.")

    if not out_path.exists() or out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        detail = "; ".join(notes) if notes else "no explanation recorded"
        return [TextContent(type="text", text=(
            f"Recording failed to finalize the MP4 ({detail}):\n{stderr}\n"
            f"Dropped frames: {drops}"
        ))]

    summary = _recording_summary(recording, None)

    ffprobe = _find_binary("ffprobe")
    if not ffprobe:
        return [TextContent(
            type="text",
            text=(
                f"Recording finalized at {out_path}, but ffprobe is unavailable so its "
                f"playback contract is unverified.\n\n{summary}"
            ),
        )]
    try:
        probe = await asyncio.to_thread(_probe_video, out_path, ffprobe)
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        return [TextContent(type="text", text=(
            f"Recording finalized at {out_path}, but ffprobe failed: {exc}\n\n{summary}"
        ))]

    problems = []
    if probe["codec"] != "h264":
        problems.append(f"codec is {probe['codec']}, expected h264")
    if probe["pix_fmt"] != "yuv420p":
        problems.append(f"pixel format is {probe['pix_fmt']}, expected yuv420p")
    if probe["fps"] < MIN_FPS:
        problems.append(f"frame rate is {probe['fps']:.2f}, expected at least {MIN_FPS}")
    if probe["width"] <= 0 or probe["height"] <= 0 or probe["width"] % 2 or probe["height"] % 2:
        problems.append(f"dimensions are not positive and even: {probe['width']}x{probe['height']}")
    if probe["frames"] <= 0 or probe["duration"] <= 0:
        problems.append(f"empty timeline: {probe['frames']} frames over {probe['duration']:.2f}s")

    summary = "\n".join(
        [f"Path: {out_path}", _recording_summary(recording, probe)]
    )
    if notes:
        summary = "\n".join(notes) + "\n\n" + summary

    if problems:
        return [TextContent(type="text", text="Recording failed verification:\n- " + "\n- ".join(problems) + "\n\n" + summary)]
    return [TextContent(type="text", text="Real-time simulator recording finalized and verified.\n\n" + summary)]
