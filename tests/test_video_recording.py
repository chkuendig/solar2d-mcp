"""Regression tests for the relay-based offscreen simulator recording."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import signal
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import runtime
from tools import video
from utils import running_projects

# Wire format of one SOLAR2D_VIDEO_PIPE frame: 64-byte header + w*h*4 BGRA.
FRAME_HEADER = struct.Struct("<4sHHIII4sB7xQQ16x")


def build_frame(width: int, height: int, seq: int) -> bytes:
    payload = bytes([(0x30 + seq) % 0x100]) * (width * height * 4)
    header = FRAME_HEADER.pack(
        b"S2VT", 1, FRAME_HEADER.size, width, height, width * 4, b"BGRA", 1, seq, time.time_ns()
    )
    return header + payload


def write_fake_ffmpeg(root: Path) -> str:
    """A stand-in encoder that drains stdin and leaves a non-empty output."""
    script = root / "fake-ffmpeg"
    script.write_text(
        "#!/bin/sh\n"
        "cat > /dev/null\n"
        "for last; do :; done\n"
        "printf 'ftypisom' > \"$last\"\n"
    )
    script.chmod(0o755)
    return str(script)


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def valid_probe(frames: int = 90, duration: float = 3.0, width: int = 64, height: int = 48) -> dict:
    return {
        "codec": "h264",
        "pix_fmt": "yuv420p",
        "width": width,
        "height": height,
        "fps": 30.0,
        "frames": frames,
        "duration": duration,
    }


class FakeSimulator:
    def poll(self) -> None:
        return None


class ExitedSimulator:
    def poll(self) -> int:
        return 0


class DeadThread:
    def is_alive(self) -> bool:
        return False


class FakeStdin:
    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeRecorder:
    def __init__(self, returncode: int | None = None, args: list[str] | None = None) -> None:
        self.pid = 43210
        self.returncode = returncode
        self.args = args or []
        self.stdin = FakeStdin()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


class FakeTap:
    """Engine-side stand-in: owns the FIFO and marker, streams framed BGRA."""

    def __init__(self, root: Path) -> None:
        self.fifo_path = root / "video.fifo"
        self.ready_path = root / "video.fifo.ready"
        os.mkfifo(self.fifo_path, 0o600)
        self.commands: queue.Queue[tuple | None] = queue.Queue()
        self.error: OSError | None = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "FakeTap":
        self.ready_path.write_text(f"pid {os.getpid()} version 1\n")
        self.thread.start()
        return self

    def send_frame(self, width: int, height: int, seq: int) -> None:
        self.commands.put((width, height, seq))

    def finish(self, timeout: float = 5.0) -> None:
        self.commands.put(None)
        if self.done.wait(0.5):
            return
        # The relay never connected; provide a reader so the writer's open returns.
        try:
            os.close(os.open(self.fifo_path, os.O_RDONLY | os.O_NONBLOCK))
        except OSError:
            pass
        if not self.done.wait(timeout):
            raise AssertionError("fake tap writer did not finish")

    def _run(self) -> None:
        fd: int | None = None
        try:
            fd = os.open(self.fifo_path, os.O_WRONLY | os.O_CLOEXEC)
            while True:
                command = self.commands.get()
                if command is None:
                    break
                width, height, seq = command
                data = build_frame(width, height, seq)
                while data:
                    data = data[os.write(fd, data):]
        except OSError as exc:
            self.error = exc
        finally:
            if fd is not None:
                os.close(fd)
            self.ready_path.unlink(missing_ok=True)
            self.done.set()


class VideoRecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        running_projects.clear()
        runtime._unreported_recordings.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "main.lua").write_text("-- test\n")
        self.screenshots = self.root / "screenshots"
        self.screenshots.mkdir()
        self.launch = {
            "launch_id": "launch",
            "project_dir": str(self.project),
            "process": FakeSimulator(),
            "screenshot_dir": str(self.screenshots),
        }
        running_projects[str(self.project)] = self.launch
        self.previous_artifacts = os.environ.get("SOLAR2D_MCP_ARTIFACT_DIR")
        os.environ["SOLAR2D_MCP_ARTIFACT_DIR"] = str(self.root / "artifacts")
        self.tap_patch = mock.patch.object(video, "_VIDEO_PIPE_PATH", self.root / "video.fifo")
        self.tap_patch.start()
        self.taps: list[FakeTap] = []

    def tearDown(self) -> None:
        for launch in [self.launch, *running_projects.values()]:
            for key in ("video_recording", "finished_video_recording"):
                recording = launch.get(key)
                if recording is not None and not recording.get("finalized"):
                    video.finalize_recording(recording)
                log_handle = (recording or {}).get("log_handle")
                if log_handle is not None and not log_handle.closed:
                    log_handle.close()
        for tap in self.taps:
            if not tap.done.is_set():
                tap.finish()
        self.tap_patch.stop()
        running_projects.clear()
        runtime._unreported_recordings.clear()
        self.temp_dir.cleanup()
        if self.previous_artifacts is None:
            os.environ.pop("SOLAR2D_MCP_ARTIFACT_DIR", None)
        else:
            os.environ["SOLAR2D_MCP_ARTIFACT_DIR"] = self.previous_artifacts

    def _wait_for(self, predicate, description: str, timeout: float = 8.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail(f"timed out waiting for {description}")

    def _start_recording(self, ffmpeg: str, **arguments: Any) -> dict:
        self.taps.append(FakeTap(self.root).start())
        with mock.patch.object(video, "_find_binary", return_value=ffmpeg):
            result = asyncio.run(video.handle_start_recording({
                "project_path": str(self.project),
                **arguments,
            }))
        self.assertIn("recording started", result[0].text)
        return self.launch["video_recording"]

    def _finished_recording(self, **overrides: Any) -> dict:
        out_path = self.root / "artifacts" / "recording.mp4"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_bytes(b"video")
        log_path = self.root / ".recording.mp4.ffmpeg.log"
        log_path.write_text("")
        recording: dict[str, Any] = {
            "stop_event": threading.Event(),
            "relay_thread": DeadThread(),
            "process": FakeRecorder(0),
            "log_handle": log_path.open("ab"),
            "log_path": str(log_path),
            "out_path": str(out_path),
            "frames": 90,
            "drops": 0,
            "end_reason": None,
            "resized_to": None,
            "relay_error": None,
            "segments": [str(out_path)],
            "finalized": True,
        }
        recording.update(overrides)
        return recording

    def test_missing_ready_marker_reports_unsupported_runtime_fast(self) -> None:
        started = time.monotonic()
        with mock.patch.object(video, "_find_binary", return_value="/usr/bin/ffmpeg"):
            result = asyncio.run(video.handle_start_recording({"project_path": str(self.project)}))

        self.assertIn("requires a runtime", result[0].text)
        self.assertIn("offscreen video tap", result[0].text)
        self.assertNotIn("X11", result[0].text)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertNotIn("video_recording", self.launch)

    def test_dead_tap_pid_reports_unsupported_runtime(self) -> None:
        self.root.joinpath("video.fifo.ready").write_text("pid 999999999 version 1\n")
        with mock.patch.object(video, "_find_binary", return_value="/usr/bin/ffmpeg"):
            result = asyncio.run(video.handle_start_recording({"project_path": str(self.project)}))

        self.assertIn("requires a runtime", result[0].text)
        self.assertNotIn("video_recording", self.launch)

    def test_start_reports_missing_ffmpeg(self) -> None:
        self.taps.append(FakeTap(self.root).start())
        with mock.patch.object(video, "_find_binary", return_value=None):
            result = asyncio.run(video.handle_start_recording({"project_path": str(self.project)}))

        self.assertIn("requires ffmpeg", result[0].text)
        self.assertNotIn("video_recording", self.launch)

    def test_start_spawns_encoder_from_first_frame_and_clamps_fps(self) -> None:
        recording = self._start_recording(write_fake_ffmpeg(self.root), fps=5)
        self.taps[0].send_frame(64, 48, 0)
        self._wait_for(lambda: recording["process"] is not None, "encoder spawn")

        command = recording["process"].args
        self.assertEqual(command[command.index("-f") + 1], "rawvideo")
        self.assertEqual(command[command.index("-pixel_format") + 1], "bgr0")
        self.assertEqual(command[command.index("-video_size") + 1], "64x48")
        self.assertEqual(command[command.index("-framerate") + 1], "15")
        self.assertEqual(command[command.index("-r") + 1], "15")
        self.assertIn("pipe:0", command)
        self.assertIn("vflip,crop=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", command)
        self.assertEqual(
            command[command.index("-movflags") + 1], "+frag_keyframe+empty_moov+default_base_moof"
        )
        self.assertEqual(command[command.index("-frag_duration") + 1], "1000000")
        self.assertEqual(command[command.index("-flush_packets") + 1], "1")
        self.taps[0].finish()

    def test_drops_are_counted_from_sequence_gaps(self) -> None:
        recording = self._start_recording(write_fake_ffmpeg(self.root))
        for seq in (0, 1, 2, 5, 6):
            self.taps[0].send_frame(64, 48, seq)
        self._wait_for(lambda: recording["frames"] == 5, "all frames relayed")
        self.taps[0].finish()

        with (
            mock.patch.object(video, "_find_binary", return_value="/usr/bin/ffprobe"),
            mock.patch.object(video, "_probe_video", return_value=valid_probe()),
        ):
            result = asyncio.run(video.handle_stop_recording({"project_path": str(self.project)}))

        self.assertIn("finalized and verified", result[0].text)
        self.assertIn("Tap frames: 5 (2 dropped)", result[0].text)

    def test_resize_finalizes_the_segment_and_reports_it(self) -> None:
        recording = self._start_recording(write_fake_ffmpeg(self.root))
        for seq in range(4):
            self.taps[0].send_frame(64, 48, seq)
        self._wait_for(lambda: recording["frames"] == 4, "pre-resize frames")
        self.taps[0].send_frame(32, 32, 4)
        self._wait_for(lambda: not recording["relay_thread"].is_alive(), "relay to end on resize")

        with (
            mock.patch.object(video, "_find_binary", return_value="/usr/bin/ffprobe"),
            mock.patch.object(video, "_probe_video", return_value=valid_probe()),
        ):
            result = asyncio.run(video.handle_stop_recording({"project_path": str(self.project)}))

        self.assertIn("finalized and verified", result[0].text)
        self.assertIn("window was resized to 32x32", result[0].text)
        self.assertIn("Segments:", result[0].text)
        self.assertIn(recording["out_path"], result[0].text)
        self.assertNotIn("video_recording", self.launch)
        self.taps[0].finish()

    def test_stop_reports_recording_finalized_on_relaunch(self) -> None:
        recording = self._start_recording(write_fake_ffmpeg(self.root))
        for seq in range(3):
            self.taps[0].send_frame(64, 48, seq)
        self._wait_for(lambda: recording["frames"] == 3, "frames before relaunch")
        self.taps[0].finish()

        with mock.patch.object(runtime, "_stop_process"):
            runtime.stop_tracked_simulators()
        self.assertNotIn("video_recording", self.launch)

        new_launch = {
            "launch_id": "relaunch",
            "project_dir": str(self.project),
            "process": FakeSimulator(),
            "screenshot_dir": str(self.screenshots),
        }
        running_projects[str(self.project)] = new_launch
        carried = runtime.take_finished_recording(str(self.project))
        self.assertIsNotNone(carried)
        self.assertIsNone(runtime.take_finished_recording(str(self.project)))
        new_launch["finished_video_recording"] = carried

        with (
            mock.patch.object(video, "_find_binary", return_value="/usr/bin/ffprobe"),
            mock.patch.object(video, "_probe_video", return_value=valid_probe()),
        ):
            result = asyncio.run(video.handle_stop_recording({"project_path": str(self.project)}))

        self.assertIn("finalized and verified", result[0].text)
        self.assertIn("finalized automatically when the simulator relaunched", result[0].text)
        self.assertNotIn("finished_video_recording", new_launch)

    def test_start_mentions_finalized_recording_from_previous_launch(self) -> None:
        self.launch["finished_video_recording"] = self._finished_recording()
        self.taps.append(FakeTap(self.root).start())
        with mock.patch.object(video, "_find_binary", return_value=write_fake_ffmpeg(self.root)):
            result = asyncio.run(video.handle_start_recording({"project_path": str(self.project)}))

        self.assertIn("finalized when the simulator relaunched", result[0].text)
        self.assertNotIn("video_recording", self.launch)

    def test_stop_reports_when_no_frames_arrived(self) -> None:
        recording = self._finished_recording(process=None, finalized=False, frames=0)
        self.launch["video_recording"] = recording

        result = asyncio.run(video.handle_stop_recording({"project_path": str(self.project)}))

        self.assertIn("no frames arrived", result[0].text)
        self.assertIn("Dropped frames: 0", result[0].text)
        self.assertFalse(Path(recording["log_path"]).exists())

    def test_stop_finalizes_after_simulator_exit(self) -> None:
        self.launch["process"] = ExitedSimulator()
        self.launch["video_recording"] = self._finished_recording()
        with (
            mock.patch.object(video, "_find_binary", return_value="/usr/bin/ffprobe"),
            mock.patch.object(video, "_probe_video", return_value=valid_probe()),
        ):
            result = asyncio.run(video.handle_stop_recording({"project_path": str(self.project)}))

        self.assertIn("finalized and verified", result[0].text)
        self.assertNotIn("video_recording", self.launch)

    def test_stop_does_not_bypass_live_readiness_error(self) -> None:
        self.launch.pop("launch_id")
        self.launch["video_recording"] = self._finished_recording()
        with mock.patch.object(video, "finalize_recording") as finalize:
            result = asyncio.run(video.handle_stop_recording({"project_path": str(self.project)}))

        self.assertIn("predates launch readiness", result[0].text)
        finalize.assert_not_called()

    def test_stop_rejects_a_slideshow(self) -> None:
        self.launch["video_recording"] = self._finished_recording()
        probe = valid_probe(frames=5, duration=5.0)
        probe["fps"] = 1.0
        with (
            mock.patch.object(video, "_find_binary", return_value="/usr/bin/ffprobe"),
            mock.patch.object(video, "_probe_video", return_value=probe),
        ):
            result = asyncio.run(video.handle_stop_recording({"project_path": str(self.project)}))

        self.assertIn("failed verification", result[0].text)
        self.assertIn("frame rate is 1.00", result[0].text)

    def test_stop_reports_verified_playback_contract(self) -> None:
        self.launch["video_recording"] = self._finished_recording()
        with (
            mock.patch.object(video, "_find_binary", return_value="/usr/bin/ffprobe"),
            mock.patch.object(video, "_probe_video", return_value=valid_probe()),
        ):
            result = asyncio.run(video.handle_stop_recording({"project_path": str(self.project)}))

        self.assertIn("finalized and verified", result[0].text)
        self.assertIn("90 frames over 3.00s @ 30.00 fps", result[0].text)
        self.assertIn("Tap frames: 90 (0 dropped)", result[0].text)
        self.assertNotIn("video_recording", self.launch)

    @unittest.skipIf(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        "ffmpeg and ffprobe are required",
    )
    def test_relay_produces_a_valid_mp4(self) -> None:
        self.taps.append(FakeTap(self.root).start())
        result = asyncio.run(video.handle_start_recording({
            "project_path": str(self.project),
            "duration": 10,
            "filename": "clip.mp4",
        }))
        self.assertIn("recording started", result[0].text)
        recording = self.launch["video_recording"]

        for seq in range(12):
            self.taps[0].send_frame(64, 48, seq)
            time.sleep(0.04)
        self._wait_for(lambda: recording["frames"] == 12, "all frames relayed")
        self.taps[0].finish()

        result = asyncio.run(video.handle_stop_recording({"project_path": str(self.project)}))

        self.assertIn("finalized and verified", result[0].text)
        self.assertIn("Dimensions: 64x48", result[0].text)
        self.assertIn("(0 dropped)", result[0].text)
        self.assertGreater(Path(recording["out_path"]).stat().st_size, 0)

    @unittest.skipIf(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        "ffmpeg and ffprobe are required",
    )
    def test_sigkilled_encoder_still_leaves_a_playable_mp4(self) -> None:
        self.taps.append(FakeTap(self.root).start())
        result = asyncio.run(video.handle_start_recording({
            "project_path": str(self.project),
            "duration": 30,
            "filename": "killed.mp4",
        }))
        self.assertIn("recording started", result[0].text)
        recording = self.launch["video_recording"]
        self.taps[0].send_frame(64, 48, 0)
        seq = 1
        self._wait_for(lambda: recording["process"] is not None, "encoder spawn")
        out_path = Path(recording["out_path"])

        for _ in range(14):
            self.taps[0].send_frame(64, 48, seq)
            seq += 1
            time.sleep(0.033)
        # -frag_duration closes a moof every second of content, independent of
        # the fps cap and the encoder GOP, so a fragment must appear quickly.
        deadline = time.monotonic() + 3.0
        while b"moof" not in _read_bytes(out_path) and time.monotonic() < deadline:
            self.taps[0].send_frame(64, 48, seq)
            seq += 1
            time.sleep(0.033)
        self.assertIn(b"moof", _read_bytes(out_path), "no fragment closed before the kill")

        os.kill(recording["process"].pid, signal.SIGKILL)
        result = asyncio.run(video.handle_stop_recording({"project_path": str(self.project)}))

        self.assertIn("finalized and verified", result[0].text)
        probe = video._probe_video(out_path, shutil.which("ffprobe"))
        self.assertGreater(probe["frames"], 0)
        self.assertGreater(probe["duration"], 0.0)

    def test_probe_parses_ffprobe_json(self) -> None:
        payload = {
            "streams": [{
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 640,
                "height": 1390,
                "avg_frame_rate": "30/1",
                "nb_read_frames": "91",
            }],
            "format": {"duration": "3.034"},
        }
        completed = mock.Mock(stdout=json.dumps(payload))
        with mock.patch.object(video.subprocess, "run", return_value=completed):
            result = video._probe_video(Path("clip.mp4"), "ffprobe")

        self.assertEqual(result["codec"], "h264")
        self.assertEqual(result["fps"], 30.0)
        self.assertEqual(result["frames"], 91)

    def test_runtime_finalizes_relay_recording_and_keeps_it_reportable(self) -> None:
        simulator = self.launch["process"]
        finalize = mock.Mock()
        log_path = self.root / ".recording.mp4.ffmpeg.log"
        log_path.write_text("")
        recording = {
            "finalize": finalize,
            "log_handle": log_path.open("ab"),
            "log_path": str(log_path),
        }
        self.launch["video_recording"] = recording
        cleanup_files = mock.Mock()
        self.launch["cleanup_files"] = cleanup_files

        with (
            mock.patch.object(runtime, "_finish_recording_process") as finish_recording,
            mock.patch.object(runtime, "_stop_process") as stop_process,
        ):
            runtime.stop_tracked_simulators()

        finalize.assert_called_once_with(recording)
        finish_recording.assert_not_called()
        stop_process.assert_called_once_with(simulator)
        cleanup_files.assert_called_once_with(self.launch)
        self.assertTrue(recording["log_handle"].closed)
        self.assertTrue(log_path.exists())
        self.assertIs(runtime.take_finished_recording(str(self.project)), recording)

    def test_runtime_falls_back_to_direct_finalization_for_plain_recordings(self) -> None:
        recorder = FakeRecorder()
        log_path = self.root / ".recording.mp4.ffmpeg.log"
        log_path.write_text("")
        recording = {
            "process": recorder,
            "log_handle": log_path.open("ab"),
            "log_path": str(log_path),
        }
        self.launch["video_recording"] = recording

        with (
            mock.patch.object(runtime, "_finish_recording_process") as finish_recording,
            mock.patch.object(runtime, "_stop_process"),
        ):
            runtime.stop_tracked_simulators()

        finish_recording.assert_called_once_with(recorder)
        self.assertIs(runtime.take_finished_recording(str(self.project)), recording)

    def test_runtime_shutdown_continues_when_file_cleanup_fails(self) -> None:
        self.launch["cleanup_files"] = mock.Mock(side_effect=OSError("project disappeared"))

        with mock.patch.object(runtime, "_stop_process") as stop_process:
            runtime.stop_tracked_simulators()

        stop_process.assert_called_once_with(self.launch["process"])
        self.launch["cleanup_files"].assert_called_once_with(self.launch)
        self.assertFalse(running_projects)

    def test_shutdown_unlinks_logs_of_unreported_recordings(self) -> None:
        self._start_recording(write_fake_ffmpeg(self.root))
        log_path = self.launch["video_recording"]["log_path"]

        with mock.patch.object(runtime, "_stop_process"):
            runtime.stop_tracked_simulators()
        self.assertTrue(Path(log_path).exists())
        with mock.patch.object(runtime, "_stop_process"):
            runtime.shutdown_runtime()
        self.assertFalse(Path(log_path).exists())
        self.assertFalse(runtime._unreported_recordings)

    def test_recording_process_is_finalized_with_ffmpeg_quit_command(self) -> None:
        recorder = FakeRecorder()

        runtime._finish_recording_process(recorder)

        self.assertEqual(recorder.stdin.data, b"q\n")
        self.assertTrue(recorder.stdin.closed)
        self.assertEqual(recorder.returncode, 0)


if __name__ == "__main__":
    unittest.main()
