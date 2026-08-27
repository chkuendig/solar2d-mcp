"""Live X11 video recording for the Solar2D simulator."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from mcp.types import TextContent, Tool

from runtime import _finish_recording_process
from utils import find_main_lua, get_current_launch, running_projects

MAX_RECORDING_SECONDS = 300
MIN_FPS = 15
MAX_FPS = 60

START_VIDEO_RECORDING_TOOL = Tool(
    name="start_video_recording",
    description=(
        "Start a real-time MP4 recording of the Solar2D simulator's X11 display. "
        "This captures the framebuffer directly instead of stitching periodic screenshots. "
        "Call stop_video_recording when the interaction is complete."
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
        "and verified codec, pixel format, dimensions, frame rate, frame count, and duration."
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


def _display_dimensions(display: str, xdpyinfo: str) -> tuple[int, int]:
    result = subprocess.run(
        [xdpyinfo, "-display", display],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    match = re.search(r"dimensions:\s+(\d+)x(\d+)\s+pixels", result.stdout)
    if not match:
        raise RuntimeError("xdpyinfo did not report the X11 display dimensions")
    return int(match.group(1)), int(match.group(2))


def _display_input(display: str) -> str:
    screen = display if re.search(r"\.\d+$", display) else f"{display}.0"
    return f"{screen}+0,0"


def _tail(path: Path, limit: int = 1500) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _finish_process(recording: dict[str, Any]) -> tuple[int, str]:
    process: subprocess.Popen[bytes] = recording["process"]
    _finish_recording_process(process)

    log_handle = recording.get("log_handle")
    if log_handle is not None and not log_handle.closed:
        log_handle.close()
    return process.returncode or 0, _tail(Path(recording["log_path"]))


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
        status = "still running" if current["process"].poll() is None else "ready to finalize"
        return [TextContent(
            type="text",
            text=f"A video recording is already {status}. Call stop_video_recording before starting another.",
        )]

    display = os.environ.get("DISPLAY")
    ffmpeg = _find_binary("ffmpeg")
    xdpyinfo = _find_binary("xdpyinfo")
    if not display or not ffmpeg or not xdpyinfo:
        return [TextContent(
            type="text",
            text=(
                "Real-time recording requires an X11 runtime with DISPLAY, ffmpeg, and xdpyinfo. "
                "Use screenshot tools for still-image diagnostics on other platforms."
            ),
        )]

    duration = min(MAX_RECORDING_SECONDS, max(1, int(arguments.get("duration", 30))))
    fps = min(MAX_FPS, max(MIN_FPS, int(arguments.get("fps", 30))))
    filename = Path(str(arguments.get("filename", "recording.mp4"))).name
    if not filename.lower().endswith(".mp4"):
        filename += ".mp4"

    try:
        width, height = await asyncio.to_thread(_display_dimensions, display, xdpyinfo)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return [TextContent(type="text", text=f"Could not inspect X11 display {display}: {exc}")]

    out_dir = _video_dir(launch)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    log_path = out_dir / f".{filename}.ffmpeg.log"
    out_path.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)
    log_handle = log_path.open("wb")

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "x11grab",
        "-draw_mouse",
        "0",
        "-framerate",
        str(fps),
        "-video_size",
        f"{width}x{height}",
        "-i",
        _display_input(display),
        "-an",
        "-vf",
        "crop=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
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
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log_handle,
            start_new_session=True,
        )
    except OSError as exc:
        log_handle.close()
        return [TextContent(type="text", text=f"Could not start ffmpeg: {exc}")]

    recording = {
        "process": process,
        "log_handle": log_handle,
        "log_path": str(log_path),
        "out_path": str(out_path),
        "fps": fps,
        "width": width,
        "height": height,
    }
    launch["video_recording"] = recording
    await asyncio.sleep(0.25)
    if process.poll() is not None:
        launch.pop("video_recording", None)
        code, stderr = await asyncio.to_thread(_finish_process, recording)
        return [TextContent(type="text", text=f"ffmpeg exited during startup (exit {code}):\n{stderr}")]

    return [TextContent(type="text", text=(
        "Real-time simulator recording started.\n\n"
        f"Path: {out_path}\n"
        f"Display: {display} ({width}x{height})\n"
        f"Capture: {fps} fps, up to {duration}s, H.264/yuv420p\n\n"
        "Drive the simulator now, then call stop_video_recording to finalize and verify the MP4."
    ))]


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
            or stopped_launch.get("video_recording") is None
            or not stopped_launch.get("launch_id")
            or process is None
            or process.poll() is None
        ):
            return [TextContent(type="text", text=f"Error: {error}")]
        launch = stopped_launch
    assert launch is not None

    recording = launch.pop("video_recording", None)
    if recording is None:
        return [TextContent(type="text", text="No real-time video recording is active for this launch.")]

    code, stderr = await asyncio.to_thread(_finish_process, recording)
    out_path = Path(recording["out_path"])
    Path(recording["log_path"]).unlink(missing_ok=True)
    if not out_path.exists() or out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        return [TextContent(type="text", text=f"ffmpeg failed to finalize the recording (exit {code}):\n{stderr}")]

    ffprobe = _find_binary("ffprobe")
    if not ffprobe:
        return [TextContent(
            type="text",
            text=f"Recording finalized at {out_path}, but ffprobe is unavailable so its playback contract is unverified.",
        )]
    try:
        probe = await asyncio.to_thread(_probe_video, out_path, ffprobe)
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        return [TextContent(type="text", text=f"Recording finalized at {out_path}, but ffprobe failed: {exc}")]

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

    summary = (
        f"Path: {out_path}\n"
        f"Codec: {probe['codec']} / {probe['pix_fmt']}\n"
        f"Dimensions: {probe['width']}x{probe['height']}\n"
        f"Timeline: {probe['frames']} frames over {probe['duration']:.2f}s @ {probe['fps']:.2f} fps"
    )
    if code != 0:
        summary += f"\nffmpeg stop status: {code} (artifact verified independently)"
    if problems:
        return [TextContent(type="text", text="Recording failed verification:\n- " + "\n- ".join(problems) + "\n\n" + summary)]
    return [TextContent(type="text", text="Real-time simulator recording finalized and verified.\n\n" + summary)]
