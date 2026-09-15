"""Tests for reloading an already-running simulator in place."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import run_project
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


def display_payload(launch_id: str) -> dict:
    return {
        "contentWidth": 640,
        "contentHeight": 1390,
        "actualContentWidth": 640,
        "actualContentHeight": 1390,
        "screenOriginX": 0,
        "screenOriginY": 0,
        "launchId": launch_id,
    }


def track_launch(
    project: Path,
    *,
    launch_id: str,
    process: FakeProcess,
) -> dict:
    paths = run_project._launch_paths(project.name, launch_id)
    launch = {
        "launch_id": launch_id,
        "project_dir": str(project),
        "main_lua": str(project / "main.lua"),
        "log_file": str(project.parent / "corona.log"),
        "started_at_ns": 0,
        "pid": process.pid,
        "process": process,
        **paths,
    }
    running_projects[str(project)] = launch
    return launch


def fake_prepare(project: Path, process: FakeProcess):
    def prepare(**kwargs: object) -> dict:
        launch_id = str(kwargs["launch_id"])
        paths = run_project._launch_paths(project.name, launch_id)
        Path(paths["display_info_file"]).write_text(json.dumps(display_payload(launch_id)))
        return {
            **paths,
            "launch_id": launch_id,
            "project_dir": str(project),
            "main_lua": str(project / "main.lua"),
            "log_file": str(project.parent / "corona.log"),
            "started_at_ns": 0,
            "pid": process.pid,
            "process": process,
            "logger_injected": False,
            "screenshot_injected": False,
            "touch_injected": False,
        }

    return prepare


class ReloadPathSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        running_projects.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        running_projects.clear()
        self.temp_dir.cleanup()

    def test_reload_succeeds_against_a_tracked_running_launch(self) -> None:
        project = make_project(self.tmp_path)
        simulator = self.tmp_path / "simulator"
        simulator.write_text("")
        process = FakeProcess()
        launch = track_launch(project, launch_id="existing-launch", process=process)
        info_file = Path(launch["display_info_file"])

        def fake_touch(main_lua_path: str) -> None:
            info_file.write_text(json.dumps(display_payload("existing-launch")))

        with (
            mock.patch.object(
                run_project.config, "get_simulator_or_detect", return_value=(str(simulator), [], False)
            ),
            mock.patch.object(run_project, "_touch_main_lua", side_effect=fake_touch),
            mock.patch.object(run_project, "READINESS_POLL_SECONDS", 0.005),
            # Readiness requires the info file's mtime >= the reload's start
            # time; pin the reload start to the epoch so this doesn't race
            # against filesystem mtime resolution under host load, same as
            # the pre-existing fresh-spawn tests do for readiness checks.
            mock.patch.object(run_project.time, "time_ns", return_value=0),
        ):
            result = asyncio.run(run_project.handle({"project_path": str(project), "reload": True}))

        text = result[0].text
        self.assertIn("Launch path: reload", text)
        self.assertIn("Reload latency", text)
        self.assertIn(f"PID: {process.pid}", text)
        self.assertIs(running_projects[str(project)], launch)
        self.assertIsNone(process.poll())

    def test_reload_falls_back_to_fresh_spawn_when_no_launch_is_tracked(self) -> None:
        project = make_project(self.tmp_path)
        simulator = self.tmp_path / "simulator"
        simulator.write_text("")
        new_process = FakeProcess(22222)

        with (
            mock.patch.object(
                run_project.config, "get_simulator_or_detect", return_value=(str(simulator), [], False)
            ),
            mock.patch.object(run_project, "_prepare_and_spawn", side_effect=fake_prepare(project, new_process)),
        ):
            result = asyncio.run(run_project.handle({"project_path": str(project), "reload": True}))

        text = result[0].text
        self.assertIn(
            "Reload requested but fell back to a fresh spawn: "
            "No current Solar2D launch is tracked for this project.",
            text,
        )
        self.assertIn("Launch path: fresh spawn", text)
        self.assertIs(running_projects[str(project)]["process"], new_process)

    def test_reload_falls_back_to_fresh_spawn_when_tracked_process_exited(self) -> None:
        project = make_project(self.tmp_path)
        simulator = self.tmp_path / "simulator"
        simulator.write_text("")
        dead_process = FakeProcess(11111)
        dead_process.running = False
        track_launch(project, launch_id="dead-launch", process=dead_process)
        new_process = FakeProcess(22222)

        with (
            mock.patch.object(
                run_project.config, "get_simulator_or_detect", return_value=(str(simulator), [], False)
            ),
            mock.patch.object(run_project, "_prepare_and_spawn", side_effect=fake_prepare(project, new_process)),
        ):
            result = asyncio.run(run_project.handle({"project_path": str(project), "reload": True}))

        text = result[0].text
        self.assertIn("Reload requested but fell back to a fresh spawn: The current Solar2D launch has stopped", text)
        self.assertIn("Launch path: fresh spawn", text)
        self.assertIs(running_projects[str(project)]["process"], new_process)

    def test_reload_timeout_leaves_tracked_simulator_running(self) -> None:
        project = make_project(self.tmp_path)
        simulator = self.tmp_path / "simulator"
        simulator.write_text("")
        old_process = FakeProcess(11111)
        track_launch(project, launch_id="stale-launch", process=old_process)
        with (
            mock.patch.object(
                run_project.config, "get_simulator_or_detect", return_value=(str(simulator), [], False)
            ),
            mock.patch.object(run_project, "LAUNCH_TIMEOUT_SECONDS", 0.03),
            mock.patch.object(run_project, "READINESS_POLL_SECONDS", 0.005),
            mock.patch.object(run_project, "_prepare_and_spawn") as prepare,
        ):
            result = asyncio.run(run_project.handle({"project_path": str(project), "reload": True}))

        text = result[0].text
        self.assertIn("reload did not publish fresh instrumentation within 0.03s", text)
        self.assertIn("Launch path: reload", text)
        self.assertIn(f"PID: {old_process.pid} (left running)", text)
        prepare.assert_not_called()
        self.assertIsNone(old_process.poll())
        self.assertIs(running_projects[str(project)]["process"], old_process)

    def test_reload_falls_back_when_process_exits_during_reload(self) -> None:
        project = make_project(self.tmp_path)
        simulator = self.tmp_path / "simulator"
        simulator.write_text("")
        old_process = FakeProcess(11111)
        launch = track_launch(project, launch_id="old-launch", process=old_process)
        new_process = FakeProcess(22222)

        def fake_touch(main_lua_path: str) -> None:
            old_process.running = False

        with (
            mock.patch.object(
                run_project.config, "get_simulator_or_detect", return_value=(str(simulator), [], False)
            ),
            mock.patch.object(run_project, "_touch_main_lua", side_effect=fake_touch),
            mock.patch.object(run_project, "_prepare_and_spawn", side_effect=fake_prepare(project, new_process)),
        ):
            result = asyncio.run(run_project.handle({"project_path": str(project), "reload": True}))

        text = result[0].text
        self.assertIn("the tracked simulator exited with code 0", text)
        self.assertIn("Launch path: fresh spawn", text)
        self.assertIsNot(running_projects[str(project)], launch)
        self.assertIs(running_projects[str(project)]["process"], new_process)

    def test_reload_trigger_error_leaves_tracked_simulator_running(self) -> None:
        project = make_project(self.tmp_path)
        simulator = self.tmp_path / "simulator"
        simulator.write_text("")
        process = FakeProcess()
        track_launch(project, launch_id="existing-launch", process=process)

        with (
            mock.patch.object(
                run_project.config, "get_simulator_or_detect", return_value=(str(simulator), [], False)
            ),
            mock.patch.object(run_project, "_touch_main_lua", side_effect=OSError("read-only filesystem")),
            mock.patch.object(run_project, "_prepare_and_spawn") as prepare,
        ):
            result = asyncio.run(run_project.handle({"project_path": str(project), "reload": True}))

        self.assertIn("tracked simulator was left running: read-only filesystem", result[0].text)
        prepare.assert_not_called()
        self.assertIsNone(process.poll())
        self.assertIs(running_projects[str(project)]["process"], process)

    def test_reload_not_attempted_when_flag_is_omitted(self) -> None:
        project = make_project(self.tmp_path)
        simulator = self.tmp_path / "simulator"
        simulator.write_text("")
        tracked_process = FakeProcess(11111)
        track_launch(project, launch_id="existing-launch", process=tracked_process)
        new_process = FakeProcess(22222)

        with (
            mock.patch.object(
                run_project.config, "get_simulator_or_detect", return_value=(str(simulator), [], False)
            ),
            mock.patch.object(run_project, "_prepare_and_spawn", side_effect=fake_prepare(project, new_process)),
            mock.patch.object(run_project, "_touch_main_lua") as touch,
        ):
            result = asyncio.run(run_project.handle({"project_path": str(project)}))

        touch.assert_not_called()
        text = result[0].text
        self.assertNotIn("Reload requested", text)
        self.assertIn("Launch path: fresh spawn", text)
        self.assertIs(running_projects[str(project)]["process"], new_process)


class SimulatorConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_ensure_relaunch_on_file_change_writes_key_value_pair(self) -> None:
        fake_home = self.tmp_path / "home"

        with (
            mock.patch.object(run_project.platform, "system", return_value="Linux"),
            mock.patch.object(run_project.Path, "home", return_value=fake_home),
        ):
            run_project._ensure_relaunch_on_file_change()

        conf_path = fake_home / ".Solar2D" / "Sandbox" / "homescreen" / "app.conf"
        self.assertEqual(conf_path.read_text(), "relaunchOnFileChange=Always\n")

    def test_ensure_relaunch_on_file_change_is_idempotent_and_preserves_other_keys(self) -> None:
        fake_home = self.tmp_path / "home"

        with (
            mock.patch.object(run_project.platform, "system", return_value="Linux"),
            mock.patch.object(run_project.Path, "home", return_value=fake_home),
        ):
            run_project._ensure_relaunch_on_file_change()
            conf_path = fake_home / ".Solar2D" / "Sandbox" / "homescreen" / "app.conf"
            conf_path.write_text(conf_path.read_text() + "showWelcome=false\n")

            run_project._ensure_relaunch_on_file_change()

        content = conf_path.read_text()
        self.assertEqual(content.count("relaunchOnFileChange=Always"), 1)
        self.assertIn("showWelcome=false", content)

    def test_ensure_relaunch_on_file_change_is_a_noop_off_linux(self) -> None:
        fake_home = self.tmp_path / "home"

        with (
            mock.patch.object(run_project.platform, "system", return_value="Darwin"),
            mock.patch.object(run_project.Path, "home", return_value=fake_home),
        ):
            run_project._ensure_relaunch_on_file_change()

        self.assertFalse((fake_home / ".Solar2D").exists())

    def test_touch_main_lua_rewrites_content_unchanged(self) -> None:
        main_lua = self.tmp_path / "main.lua"
        original = b"-- header\r\nprint('test')\r\n"
        main_lua.write_bytes(original)

        run_project._touch_main_lua(str(main_lua))

        self.assertEqual(main_lua.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
