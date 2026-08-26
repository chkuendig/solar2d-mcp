"""Regression tests for shared-runtime simulator ownership."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_second_client_reports_busy_without_losing_connection(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    env["SOLAR2D_MCP_RUNTIME_DIR"] = str(tmp_path)

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from runtime import acquire_simulator_slot; "
                "import time; "
                "print(acquire_simulator_slot(), flush=True); "
                "time.sleep(10)"
            ),
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "True"

        contender = subprocess.run(
            [
                sys.executable,
                "-c",
                "from runtime import simulator_busy_message; print(simulator_busy_message())",
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )

        assert "Solar2D runtime is busy" in contender.stdout
        assert "The MCP connection is healthy" in contender.stdout
    finally:
        holder.terminate()
        holder.wait(timeout=5)


def test_video_directory_can_be_exported(tmp_path: Path) -> None:
    previous = os.environ.get("SOLAR2D_MCP_ARTIFACT_DIR")
    os.environ["SOLAR2D_MCP_ARTIFACT_DIR"] = str(tmp_path)
    try:
        from tools.screenshot import _get_video_dir

        assert Path(_get_video_dir("unused")) == tmp_path
    finally:
        if previous is None:
            os.environ.pop("SOLAR2D_MCP_ARTIFACT_DIR", None)
        else:
            os.environ["SOLAR2D_MCP_ARTIFACT_DIR"] = previous
