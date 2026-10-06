"""Regression tests for launch-scoped social preview screenshots."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from tools.social import preview
from utils import running_projects


class FakeProcess:
    def poll(self) -> None:
        return None


class SocialPreviewMediaTests(unittest.TestCase):
    def setUp(self) -> None:
        running_projects.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.project = self.tmp_path / "project"
        self.project.mkdir()
        (self.project / "main.lua").write_text("print('test')\n")

        self.current_dir = self.tmp_path / "screenshots-current"
        self.current_dir.mkdir()
        self.old_dir = self.tmp_path / "screenshots-old-launch"
        self.old_dir.mkdir()
        running_projects[str(self.project)] = {
            "launch_id": "current-launch",
            "project_dir": str(self.project),
            "process": FakeProcess(),
            "screenshot_dir": str(self.current_dir),
        }

    def tearDown(self) -> None:
        running_projects.clear()
        self.temp_dir.cleanup()

    def test_last_uses_current_launch_directory_only(self) -> None:
        (self.current_dir / "screenshot_001.jpg").write_bytes(b"current-1")
        expected = self.current_dir / "screenshot_002.jpg"
        expected.write_bytes(b"current-2")
        (self.old_dir / "screenshot_999.jpg").write_bytes(b"old-launch")

        media_path, error = preview._resolve_media_path("last", str(self.project))

        self.assertIsNone(error)
        self.assertEqual(Path(media_path), expected)
        self.assertEqual(Path(media_path).read_bytes(), b"current-2")

    def test_numbered_screenshot_cannot_leak_from_an_old_launch(self) -> None:
        expected = self.current_dir / "screenshot_007.jpg"
        expected.write_bytes(b"current-7")
        (self.old_dir / "screenshot_007.jpg").write_bytes(b"old-launch-7")

        media_path, error = preview._resolve_media_path("7", str(self.project))

        self.assertIsNone(error)
        self.assertEqual(Path(media_path), expected)
        self.assertEqual(Path(media_path).read_bytes(), b"current-7")

        expected.unlink()
        media_path, error = preview._resolve_media_path("7", str(self.project))
        self.assertIsNone(media_path)
        self.assertIn("current Solar2D launch", error)

    def test_no_active_launch_returns_a_clear_tool_error(self) -> None:
        (self.old_dir / "screenshot_001.jpg").write_bytes(b"old-launch")
        running_projects.clear()

        result = asyncio.run(
            preview.handle(
                {
                    "content": "Update",
                    "platforms": ["twitter"],
                    "media": "last",
                    "project_path": str(self.project),
                }
            )
        )

        self.assertIn("No current Solar2D launch is tracked", result[0].text)
        self.assertIn("Run the project first", result[0].text)


if __name__ == "__main__":
    unittest.main()
