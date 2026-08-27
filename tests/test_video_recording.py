"""Regression tests for real-time X11 simulator recording."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import runtime
from tools import video
from utils import running_projects


class FakeSimulator:
    def poll(self) -> None:
        return None


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
    def __init__(self, returncode: int | None = None) -> None:
        self.pid = 43210
        self.returncode = returncode
        self.stdin = FakeStdin()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


class VideoRecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        running_projects.clear()
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
        self.previous_display = os.environ.get("DISPLAY")
        self.previous_artifacts = os.environ.get("SOLAR2D_MCP_ARTIFACT_DIR")
        os.environ["DISPLAY"] = ":99"
        os.environ["SOLAR2D_MCP_ARTIFACT_DIR"] = str(self.root / "artifacts")

    def tearDown(self) -> None:
        recording = self.launch.get("video_recording")
        if recording is not None and not recording["log_handle"].closed:
            recording["log_handle"].close()
        running_projects.clear()
        self.temp_dir.cleanup()
        if self.previous_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = self.previous_display
        if self.previous_artifacts is None:
            os.environ.pop("SOLAR2D_MCP_ARTIFACT_DIR", None)
        else:
            os.environ["SOLAR2D_MCP_ARTIFACT_DIR"] = self.previous_artifacts

    def test_start_uses_x11grab_at_30fps_and_even_crop(self) -> None:
        recorder = FakeRecorder()
        with (
            mock.patch.object(video, "_find_binary", side_effect=lambda name: f"/usr/bin/{name}"),
            mock.patch.object(video, "_display_dimensions", return_value=(639, 1409)),
            mock.patch.object(video.subprocess, "Popen", return_value=recorder) as popen,
            mock.patch.object(video.asyncio, "sleep", return_value=None),
        ):
            result = asyncio.run(video.handle_start_recording({"project_path": str(self.project)}))

        command = popen.call_args.args[0]
        self.assertIn("x11grab", command)
        self.assertNotIn("-nostdin", command)
        self.assertEqual(command[command.index("-framerate") + 1], "30")
        self.assertEqual(command[command.index("-video_size") + 1], "639x1409")
        self.assertIn("crop=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", command)
        self.assertIn("H.264/yuv420p", result[0].text)

    def test_start_clamps_requested_frame_rate(self) -> None:
        recorder = FakeRecorder()
        with (
            mock.patch.object(video, "_find_binary", side_effect=lambda name: f"/usr/bin/{name}"),
            mock.patch.object(video, "_display_dimensions", return_value=(640, 1390)),
            mock.patch.object(video.subprocess, "Popen", return_value=recorder) as popen,
            mock.patch.object(video.asyncio, "sleep", return_value=None),
        ):
            asyncio.run(video.handle_start_recording({"project_path": str(self.project), "fps": 5}))

        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("-framerate") + 1], "15")

    def _track_finished_recording(self) -> None:
        out_path = self.root / "recording.mp4"
        out_path.write_bytes(b"video")
        log_path = self.root / "recording.log"
        log_path.write_text("")
        self.launch["video_recording"] = {
            "process": FakeRecorder(0),
            "log_handle": log_path.open("ab"),
            "log_path": str(log_path),
            "out_path": str(out_path),
        }

    def test_stop_reports_verified_playback_contract(self) -> None:
        self._track_finished_recording()
        probe = {
            "codec": "h264",
            "pix_fmt": "yuv420p",
            "width": 640,
            "height": 1390,
            "fps": 30.0,
            "frames": 90,
            "duration": 3.0,
        }
        with (
            mock.patch.object(video, "_find_binary", return_value="/usr/bin/ffprobe"),
            mock.patch.object(video, "_probe_video", return_value=probe),
        ):
            result = asyncio.run(video.handle_stop_recording({"project_path": str(self.project)}))

        self.assertIn("finalized and verified", result[0].text)
        self.assertIn("90 frames over 3.00s @ 30.00 fps", result[0].text)
        self.assertNotIn("video_recording", self.launch)

    def test_stop_rejects_a_slideshow(self) -> None:
        self._track_finished_recording()
        probe = {
            "codec": "h264",
            "pix_fmt": "yuv420p",
            "width": 640,
            "height": 1390,
            "fps": 1.0,
            "frames": 5,
            "duration": 5.0,
        }
        with (
            mock.patch.object(video, "_find_binary", return_value="/usr/bin/ffprobe"),
            mock.patch.object(video, "_probe_video", return_value=probe),
        ):
            result = asyncio.run(video.handle_stop_recording({"project_path": str(self.project)}))

        self.assertIn("failed verification", result[0].text)
        self.assertIn("frame rate is 1.00", result[0].text)

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

    def test_runtime_shutdown_stops_recorder_before_simulator(self) -> None:
        recorder = FakeRecorder()
        simulator = self.launch["process"]
        log_path = self.root / "recording.log"
        log_path.write_text("")
        log_handle = log_path.open("ab")
        self.launch["video_recording"] = {
            "process": recorder,
            "log_handle": log_handle,
            "log_path": str(log_path),
        }

        with (
            mock.patch.object(runtime, "_finish_recording_process") as finish_recording,
            mock.patch.object(runtime, "_stop_process") as stop_process,
        ):
            runtime.stop_tracked_simulators()

        finish_recording.assert_called_once_with(recorder)
        stop_process.assert_called_once_with(simulator)
        self.assertTrue(log_handle.closed)
        self.assertFalse(log_path.exists())
        self.assertFalse(running_projects)

    def test_recording_process_is_finalized_with_ffmpeg_quit_command(self) -> None:
        recorder = FakeRecorder()

        runtime._finish_recording_process(recorder)

        self.assertEqual(recorder.stdin.data, b"q\n")
        self.assertTrue(recorder.stdin.closed)
        self.assertEqual(recorder.returncode, 0)


if __name__ == "__main__":
    unittest.main()
