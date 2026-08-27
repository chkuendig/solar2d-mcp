"""
Touch simulation tools - Simulate taps on the Solar2D simulator.
"""

import json
from pathlib import Path

from mcp.types import TextContent, Tool

from utils import get_current_launch, write_launch_control

# Duration in seconds for auto-recording on each interaction
AUTO_RECORD_SECONDS = 3


def _start_auto_recording(launch: dict) -> None:
    """Start a short screenshot recording to capture interaction visuals."""
    write_launch_control(
        launch["screenshot_control_file"],
        launch["launch_id"],
        str(AUTO_RECORD_SECONDS),
    )


# Tool definitions
SIMULATE_TAP_TOOL = Tool(
    name="simulate_tap",
    description="Simulate a tap/click in the Solar2D simulator. Specify a bounding box using percentages and the tool taps the center. Example: a button spanning 30-50% horizontally and 60-70% vertically would use left=30, right=50, top=60, bottom=70.",
    inputSchema={
        "type": "object",
        "properties": {
            "project_path": {
                "type": "string",
                "description": "Path to the project directory or main.lua file"
            },
            "left": {
                "type": "number",
                "description": "Left edge of target as percentage (0=left edge of screen)"
            },
            "right": {
                "type": "number",
                "description": "Right edge of target as percentage (100=right edge of screen)"
            },
            "top": {
                "type": "number",
                "description": "Top edge of target as percentage (0=top of screen)"
            },
            "bottom": {
                "type": "number",
                "description": "Bottom edge of target as percentage (100=bottom of screen)"
            }
        },
        "required": ["project_path", "left", "right", "top", "bottom"]
    }
)

GET_DISPLAY_INFO_TOOL = Tool(
    name="get_display_info",
    description="Get the Solar2D display coordinate system. Call this before tapping to understand how screenshot pixels map to tap coordinates. Screenshots are captured at contentWidth x contentHeight. Tap coordinates use the same content space.",
    inputSchema={
        "type": "object",
        "properties": {
            "project_path": {
                "type": "string",
                "description": "Path to the project directory or main.lua file"
            }
        },
        "required": ["project_path"]
    }
)

FIND_OBJECT_TOOL = Tool(
    name="find_object",
    description="Draw a persistent green rectangle around an area of interest in the Solar2D simulator. Useful for highlighting UI elements, objects, or regions. The rectangle stays on screen until the next tap, drag, or find interaction.",
    inputSchema={
        "type": "object",
        "properties": {
            "project_path": {
                "type": "string",
                "description": "Path to the project directory or main.lua file"
            },
            "left": {
                "type": "number",
                "description": "Left edge of target as percentage (0=left edge of screen)"
            },
            "right": {
                "type": "number",
                "description": "Right edge of target as percentage (100=right edge of screen)"
            },
            "top": {
                "type": "number",
                "description": "Top edge of target as percentage (0=top of screen)"
            },
            "bottom": {
                "type": "number",
                "description": "Bottom edge of target as percentage (100=bottom of screen)"
            },
            "label": {
                "type": "string",
                "description": "Optional label to display above the rectangle",
                "default": ""
            }
        },
        "required": ["project_path", "left", "right", "top", "bottom"]
    }
)

SIMULATE_DRAG_TOOL = Tool(
    name="simulate_drag",
    description="Simulate a drag/swipe gesture in the Solar2D simulator. Specify start and end bounding boxes using percentages. The gesture moves from the center of the start box to the center of the end box over the given duration.",
    inputSchema={
        "type": "object",
        "properties": {
            "project_path": {
                "type": "string",
                "description": "Path to the project directory or main.lua file"
            },
            "start_left": {
                "type": "number",
                "description": "Left edge of start area as percentage (0=left edge of screen)"
            },
            "start_right": {
                "type": "number",
                "description": "Right edge of start area as percentage (100=right edge of screen)"
            },
            "start_top": {
                "type": "number",
                "description": "Top edge of start area as percentage (0=top of screen)"
            },
            "start_bottom": {
                "type": "number",
                "description": "Bottom edge of start area as percentage (100=bottom of screen)"
            },
            "end_left": {
                "type": "number",
                "description": "Left edge of end area as percentage (0=left edge of screen)"
            },
            "end_right": {
                "type": "number",
                "description": "Right edge of end area as percentage (100=right edge of screen)"
            },
            "end_top": {
                "type": "number",
                "description": "Top edge of end area as percentage (0=top of screen)"
            },
            "end_bottom": {
                "type": "number",
                "description": "Bottom edge of end area as percentage (100=bottom of screen)"
            },
            "duration": {
                "type": "number",
                "description": "Duration of the drag in milliseconds (default: 300)",
                "default": 300
            }
        },
        "required": ["project_path", "start_left", "start_right", "start_top", "start_bottom",
                      "end_left", "end_right", "end_top", "end_bottom"]
    }
)

# Export all tools
TOOLS = [SIMULATE_TAP_TOOL, SIMULATE_DRAG_TOOL, FIND_OBJECT_TOOL, GET_DISPLAY_INFO_TOOL]


def _read_current_display_info(
    project_path: str,
) -> tuple[dict | None, dict | None, str | None]:
    launch, error = get_current_launch(project_path)
    if error:
        return None, None, error
    assert launch is not None

    info_path = Path(launch["display_info_file"])
    try:
        stat = info_path.stat()
        with info_path.open() as file:
            info = json.load(file)
    except FileNotFoundError:
        return launch, None, "Current-launch display readiness was not found. Run the project again."
    except (OSError, json.JSONDecodeError) as exc:
        return launch, None, f"Error reading current-launch display info: {exc}"

    if (
        info.get("launchId") != launch["launch_id"]
        or stat.st_mtime_ns < launch["started_at_ns"]
    ):
        return launch, None, (
            "Rejected stale display info that does not belong to the current live launch. "
            "Run the project again."
        )

    return launch, info, None


async def handle_simulate_tap(arguments: dict) -> list[TextContent]:
    """Handle simulate_tap tool call."""
    project_path = arguments.get("project_path")
    left = arguments.get("left")
    right = arguments.get("right")
    top = arguments.get("top")
    bottom = arguments.get("bottom")

    if not project_path:
        return [TextContent(type="text", text="Error: project_path is required")]

    if None in (left, right, top, bottom):
        return [TextContent(type="text", text="Error: left, right, top, bottom are all required")]

    launch, info, error = _read_current_display_info(project_path)
    if error:
        return [TextContent(type="text", text=f"Error: {error}")]
    assert launch is not None
    assert info is not None

    content_width = info.get("contentWidth")
    content_height = info.get("contentHeight")
    if not content_width or not content_height:
        return [TextContent(type="text", text="Error: Invalid display info")]

    x_percent = (left + right) / 2
    y_percent = (top + bottom) / 2
    x = int(content_width * x_percent / 100)
    y = int(content_height * y_percent / 100)

    _start_auto_recording(launch)
    command = f"tap,{x},{y}"
    write_launch_control(
        launch["touch_control_file"],
        launch["launch_id"],
        command,
    )

    return [TextContent(
        type="text",
        text=f"Tap sent to center of box ({left}-{right}%, {top}-{bottom}%)"
    )]


async def handle_find_object(arguments: dict) -> list[TextContent]:
    """Handle find_object tool call."""
    project_path = arguments.get("project_path")
    left = arguments.get("left")
    right = arguments.get("right")
    top = arguments.get("top")
    bottom = arguments.get("bottom")
    label = arguments.get("label", "")

    if not project_path:
        return [TextContent(type="text", text="Error: project_path is required")]

    if None in (left, right, top, bottom):
        return [TextContent(type="text", text="Error: left, right, top, bottom are all required")]

    launch, info, error = _read_current_display_info(project_path)
    if error:
        return [TextContent(type="text", text=f"Error: {error}")]
    assert launch is not None
    assert info is not None

    content_width = info.get("contentWidth")
    content_height = info.get("contentHeight")
    if not content_width or not content_height:
        return [TextContent(type="text", text="Error: Invalid display info")]

    x1 = int(content_width * left / 100)
    y1 = int(content_height * top / 100)
    x2 = int(content_width * right / 100)
    y2 = int(content_height * bottom / 100)

    _start_auto_recording(launch)
    command = f"find,{x1},{y1},{x2},{y2},{label}"
    write_launch_control(
        launch["touch_control_file"],
        launch["launch_id"],
        command,
    )

    label_msg = f" label=\"{label}\"" if label else ""
    return [TextContent(
        type="text",
        text=f"Find rectangle drawn at ({left}-{right}%, {top}-{bottom}%){label_msg}"
    )]


async def handle_simulate_drag(arguments: dict) -> list[TextContent]:
    """Handle simulate_drag tool call."""
    project_path = arguments.get("project_path")
    start_left = arguments.get("start_left")
    start_right = arguments.get("start_right")
    start_top = arguments.get("start_top")
    start_bottom = arguments.get("start_bottom")
    end_left = arguments.get("end_left")
    end_right = arguments.get("end_right")
    end_top = arguments.get("end_top")
    end_bottom = arguments.get("end_bottom")
    duration = arguments.get("duration", 300)

    if not project_path:
        return [TextContent(type="text", text="Error: project_path is required")]

    if None in (start_left, start_right, start_top, start_bottom,
                end_left, end_right, end_top, end_bottom):
        return [TextContent(type="text", text="Error: all start and end bounding box parameters are required")]

    launch, info, error = _read_current_display_info(project_path)
    if error:
        return [TextContent(type="text", text=f"Error: {error}")]
    assert launch is not None
    assert info is not None

    content_width = info.get("contentWidth")
    content_height = info.get("contentHeight")
    if not content_width or not content_height:
        return [TextContent(type="text", text="Error: Invalid display info")]

    sx_percent = (start_left + start_right) / 2
    sy_percent = (start_top + start_bottom) / 2
    x1 = int(content_width * sx_percent / 100)
    y1 = int(content_height * sy_percent / 100)

    ex_percent = (end_left + end_right) / 2
    ey_percent = (end_top + end_bottom) / 2
    x2 = int(content_width * ex_percent / 100)
    y2 = int(content_height * ey_percent / 100)

    _start_auto_recording(launch)
    command = f"drag,{x1},{y1},{x2},{y2},{int(duration)}"
    write_launch_control(
        launch["touch_control_file"],
        launch["launch_id"],
        command,
    )

    return [TextContent(
        type="text",
        text=f"Drag sent from ({start_left}-{start_right}%, {start_top}-{start_bottom}%) to ({end_left}-{end_right}%, {end_top}-{end_bottom}%) over {int(duration)}ms"
    )]


async def handle_get_display_info(arguments: dict) -> list[TextContent]:
    """Handle get_display_info tool call."""
    project_path = arguments.get("project_path")

    if not project_path:
        return [TextContent(type="text", text="Error: project_path is required")]

    _, info, error = _read_current_display_info(project_path)
    if error:
        return [TextContent(type="text", text=f"Error: {error}")]
    assert info is not None

    lines = [
        "Solar2D Display Info:",
        "",
        f"Content Size: {info.get('contentWidth', '?')} x {info.get('contentHeight', '?')}",
        f"Actual Content Size: {info.get('actualContentWidth', '?')} x {info.get('actualContentHeight', '?')}",
        f"Screen Origin: ({info.get('screenOriginX', '?')}, {info.get('screenOriginY', '?')})",
        "",
        "Note: Screenshots are captured at content size.",
        "Tap coordinates should be in content space (0,0 is top-left of content area)."
    ]
    return [TextContent(type="text", text="\n".join(lines))]
