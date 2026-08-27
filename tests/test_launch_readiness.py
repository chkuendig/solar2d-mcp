"""Deterministic regression tests for per-launch simulator readiness."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tools import run_project, touch
from utils import running_projects


class FakeProcess:
    def __init__(self, pid: int = 43210) -> None:
        self.pid = pid
        self.running = True

    def poll(self) -> int | None:
        return None if self.running else 0


def make_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir(parents=True)
    (project / "main.lua").write_text("print('test')\n")
    return project


def track_launch(
    project: Path,
    display_info_file: Path,
    *,
    launch_id: str = "current-launch",
    started_at_ns: int = 0,
    process: FakeProcess | None = None,
) -> dict:
    launch = {
        "launch_id": launch_id,
        "project_dir": str(project),
        "main_lua": str(project / "main.lua"),
        "display_info_file": str(display_info_file),
        "screenshot_control_file": str(display_info_file.with_suffix(".screenshot-control")),
        "screenshot_dir": str(display_info_file.with_suffix(".screenshots")),
        "touch_control_file": str(display_info_file.with_suffix(".touch-control")),
        "started_at_ns": started_at_ns,
        "process": process or FakeProcess(),
    }
    running_projects[str(project)] = launch
    return launch


def display_payload(launch_id: str | None) -> dict:
    payload = {
        "contentWidth": 640,
        "contentHeight": 1390,
        "actualContentWidth": 640,
        "actualContentHeight": 1390,
        "screenOriginX": 0,
        "screenOriginY": 0,
    }
    if launch_id is not None:
        payload["launchId"] = launch_id
    return payload


class LaunchReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        running_projects.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        running_projects.clear()
        self.temp_dir.cleanup()

    def test_stale_display_json_is_rejected(self) -> None:
        project = make_project(self.tmp_path)
        info_file = self.tmp_path / "display.json"
        info_file.write_text(json.dumps(display_payload(None)))
        track_launch(project, info_file)

        result = asyncio.run(touch.handle_get_display_info({"project_path": str(project)}))

        self.assertIn("Rejected stale display info", result[0].text)

    def test_matching_fresh_launch_is_accepted(self) -> None:
        project = make_project(self.tmp_path)
        info_file = self.tmp_path / "display.json"
        info_file.write_text(json.dumps(display_payload("current-launch")))
        track_launch(project, info_file, started_at_ns=info_file.stat().st_mtime_ns)

        result = asyncio.run(touch.handle_get_display_info({"project_path": str(project)}))

        self.assertIn("Solar2D Display Info", result[0].text)
        self.assertIn("640 x 1390", result[0].text)

    def test_control_and_display_paths_are_namespaced_per_launch(self) -> None:
        first = run_project._launch_paths("project", "first-launch")
        second = run_project._launch_paths("project", "second-launch")

        for key in (
            "display_info_file",
            "screenshot_control_file",
            "screenshot_dir",
            "touch_control_file",
        ):
            self.assertNotEqual(first[key], second[key])

    def test_launch_timeout_returns_promptly_and_stops_only_that_launch(self) -> None:
        project = make_project(self.tmp_path)
        simulator = self.tmp_path / "simulator"
        simulator.write_text("")
        process = FakeProcess()
        stopped: list[FakeProcess] = []

        def prepare(**kwargs: object) -> dict:
            launch_id = str(kwargs["launch_id"])
            paths = run_project._launch_paths(project.name, launch_id)
            return {
                **paths,
                "launch_id": launch_id,
                "project_dir": str(project),
                "main_lua": str(project / "main.lua"),
                "log_file": str(self.tmp_path / "corona.log"),
                "started_at_ns": time.time_ns(),
                "pid": process.pid,
                "process": process,
                "logger_injected": False,
                "screenshot_injected": False,
                "touch_injected": False,
            }

        def stop(fake: FakeProcess) -> None:
            stopped.append(fake)
            fake.running = False

        with (
            mock.patch.object(
                run_project.config,
                "get_simulator_or_detect",
                return_value=(str(simulator), [], False),
            ),
            mock.patch.object(run_project, "_prepare_and_spawn", side_effect=prepare),
            mock.patch.object(run_project, "stop_process", side_effect=stop),
            mock.patch.object(run_project, "LAUNCH_TIMEOUT_SECONDS", 0.05),
            mock.patch.object(run_project, "READINESS_POLL_SECONDS", 0.005),
        ):
            started = time.monotonic()
            result = asyncio.run(run_project.handle({"project_path": str(project)}))
            elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5)
        self.assertIn("did not publish launch-specific readiness within 0.05s", result[0].text)
        self.assertIn("MCP connection is healthy", result[0].text)
        self.assertEqual(stopped, [process])
        self.assertNotIn(str(project), running_projects)

    def test_cleanup_is_process_and_launch_scoped(self) -> None:
        first_project = make_project(self.tmp_path / "first")
        second_project = make_project(self.tmp_path / "second")
        first_info = self.tmp_path / "first-display.json"
        second_info = self.tmp_path / "second-display.json"
        first_process = FakeProcess(1)
        second_process = FakeProcess(2)
        first = track_launch(first_project, first_info, process=first_process)
        second = track_launch(second_project, second_info, process=second_process)

        for launch in (first, second):
            Path(launch["display_info_file"]).write_text(json.dumps(display_payload(launch["launch_id"])))
            Path(launch["screenshot_control_file"]).write_text(launch["launch_id"])
            Path(launch["touch_control_file"]).write_text(launch["launch_id"])

        stopped: list[FakeProcess] = []

        def stop(process: FakeProcess) -> None:
            stopped.append(process)
            process.running = False

        with mock.patch.object(run_project, "stop_process", side_effect=stop):
            asyncio.run(run_project._cleanup_launch(first))

        self.assertEqual(stopped, [first_process])
        self.assertNotIn(str(first_project), running_projects)
        self.assertIs(running_projects[str(second_project)], second)
        self.assertFalse(Path(first["display_info_file"]).exists())
        self.assertFalse(Path(first["screenshot_control_file"]).exists())
        self.assertFalse(Path(first["touch_control_file"]).exists())
        self.assertTrue(Path(second["display_info_file"]).exists())
        self.assertTrue(Path(second["screenshot_control_file"]).exists())
        self.assertTrue(Path(second["touch_control_file"]).exists())
        self.assertIsNone(second_process.poll())


if __name__ == "__main__":
    unittest.main()
