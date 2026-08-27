"""Shared utilities for the Solar2D MCP server."""

import os
import uuid
from pathlib import Path
from typing import Any

# Track running Corona processes and their log files
running_projects = {}  # {project_path: {"pid": int, "log_file": str, "process": subprocess.Popen}}


def find_main_lua(project_path: str) -> str:
    """Find main.lua in the given project path."""
    path = Path(project_path)

    # If the path is already main.lua
    if path.name == "main.lua" and path.exists():
        return str(path.absolute())

    # If the path is a directory, look for main.lua inside
    if path.is_dir():
        main_lua = path / "main.lua"
        if main_lua.exists():
            return str(main_lua.absolute())

    # If neither works, return the original path (will error later)
    return str(Path(project_path).absolute())


def get_current_launch(project_path: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return the live, readiness-aware launch for a project."""
    main_lua_path = find_main_lua(project_path)
    project_dir = str(Path(main_lua_path).parent)
    launch = running_projects.get(project_dir)
    if launch is None:
        return None, "No current Solar2D launch is tracked for this project. Run the project first."

    process = launch.get("process")
    if process is None or process.poll() is not None:
        return None, "The current Solar2D launch has stopped. Run the project again."

    if not launch.get("launch_id"):
        return None, "The tracked Solar2D launch predates launch readiness. Run the project again."

    return launch, None


def write_launch_control(path: str, launch_id: str, command: str) -> None:
    """Atomically publish one command scoped to a specific launch."""
    pending = f"{path}.{uuid.uuid4().hex}.pending"
    try:
        with open(pending, "w") as file:
            file.write(f"{launch_id}\n{command}")
        os.replace(pending, path)
    finally:
        try:
            os.remove(pending)
        except FileNotFoundError:
            pass
