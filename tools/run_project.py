"""
run_solar2d_project tool - Run a Solar2D project in the simulator.
"""

import asyncio
import json
import os
import platform
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from mcp.types import TextContent, Tool

import config
from runtime import _stop_process as stop_process
from runtime import stop_tracked_simulators
from utils import find_main_lua, running_projects

LAUNCH_TIMEOUT_SECONDS = 20.0
READINESS_POLL_SECONDS = 0.05
_launch_lock: asyncio.Lock | None = None
_launch_lock_loop: asyncio.AbstractEventLoop | None = None


TOOL = Tool(
    name="run_solar2d_project",
    description="Run a Solar2D project in the simulator. Provide either a path to main.lua or a project directory.",
    inputSchema={
        "type": "object",
        "properties": {
            "project_path": {
                "type": "string",
                "description": "Path to the project directory or main.lua file"
            },
            "debug": {
                "type": "boolean",
                "description": "Enable debug mode (default: true)",
                "default": True
            },
            "no_console": {
                "type": "boolean",
                "description": "Disable console output (default: false to capture logs)",
                "default": False
            }
        },
        "required": ["project_path"]
    }
)


def create_logging_wrapper(project_dir: str, log_file: str) -> str:
    """Create a Lua file that redirects print() to a log file."""
    lua_logger = f'''
-- MCP Logger: Redirects print() to file for MCP server access
local mcp_log_file = "{log_file}"
local original_print = print

-- Truncate log file on simulator start (clear old logs)
do
    local file = io.open(mcp_log_file, "w")
    if file then
        file:write("=== Solar2D Simulator Started ===\\n")
        file:close()
    end
end

_G.print = function(...)
    local args = {{...}}
    local message = ""
    for i, v in ipairs(args) do
        if i > 1 then message = message .. "\\t" end
        message = message .. tostring(v)
    end

    -- Call original print
    original_print(...)

    -- Also write to MCP log file (append mode)
    local file = io.open(mcp_log_file, "a")
    if file then
        file:write(message .. "\\n")
        file:flush()
        file:close()
    end
end

print("[MCP] Logging initialized - output will be captured for Claude")
'''

    logger_path = os.path.join(project_dir, "_mcp_logger.lua")
    with open(logger_path, 'w') as f:
        f.write(lua_logger)

    return logger_path


def create_screenshot_module(project_dir: str, project_name: str, launch_id: str) -> str:
    """Create a Lua file that captures screenshots on demand."""
    launch_name = f"{project_name}_{launch_id}"
    screenshot_dir = os.path.join(tempfile.gettempdir(), f"solar2d_screenshots_{launch_name}")
    control_file = os.path.join(tempfile.gettempdir(), f"solar2d_screenshots_{launch_name}.control")

    lua_screenshot = f'''
-- MCP Screenshot: Captures screenshots periodically when recording is enabled
local lfs = require("lfs")
local launchId = "{launch_id}"
local screenshotDir = "{screenshot_dir}"
local controlFile = "{control_file}"
local captureInterval = 100  -- 100ms between captures
local screenshotCount = 0
local recordingEndTime = 0
-- Hard safety ceiling: recording can never run longer than this, even if a
-- 'stop' is never sent or the control file is written directly. At 10fps this
-- bounds the frames on disk (~5 min -> ~3000 frames).
local MAX_RECORDING_SECONDS = 300

-- Helper to check if file exists
local function fileExists(path)
    local file = io.open(path, "r")
    if file then
        file:close()
        return true
    end
    return false
end

-- Helper to read and consume control file
local function readControlFile()
    local file = io.open(controlFile, "r")
    if file then
        local content = file:read("*all")
        file:close()
        os.remove(controlFile)  -- Consume the command
        local token, command = content:match("^([^\\n]+)\\n(.*)$")
        if token == launchId then
            return command
        end
        print("[MCP Screenshot] Ignored control command for another launch")
    end
    return nil
end

-- Clear screenshot directory on start
local function clearScreenshotDir()
    -- Create directory if it doesn't exist
    lfs.mkdir(screenshotDir)
    -- Remove existing screenshots
    for file in lfs.dir(screenshotDir) do
        if file ~= "." and file ~= ".." then
            os.remove(screenshotDir .. "/" .. file)
        end
    end
end

-- Check if currently recording
local function isRecording()
    return system.getTimer() < recordingEndTime
end

-- Publish across a filesystem boundary, where rename fails EXDEV: copy beside
-- the destination and rename within that directory. Returns nil on success.
local function copyThenRename(src, finalPath)
    local infile = io.open(src, "rb")
    if not infile then return "staged capture disappeared" end
    local content = infile:read("*all")
    infile:close()

    local nearby = finalPath .. ".part"
    local outfile = io.open(nearby, "wb")
    if not outfile then return "cannot write to " .. tostring(finalPath) end
    outfile:write(content)
    outfile:close()

    local ok, err = os.rename(nearby, finalPath)
    if not ok then
        os.remove(nearby)
        return tostring(err)
    end
    return nil
end

-- Publish a capture under its final name, atomically. display.save() writes on
-- a later render frame, so this waits for the staged file to exist and be
-- non-empty; the rename is what the reader sees, so never a partial image.
local function publishWhenReady(stagedName, finalPath, label)
    local tries = 0
    local function attempt()
        tries = tries + 1
        local staged = system.pathForFile(stagedName, system.TemporaryDirectory)
        local ready = false
        if staged then
            local f = io.open(staged, "rb")
            if f then
                -- A file that exists but is still empty is a capture mid-flight.
                ready = (f:seek("end") or 0) > 0
                f:close()
            end
        end
        if ready then
            if not os.rename(staged, finalPath) then
                local err = copyThenRename(staged, finalPath)
                if err then
                    print("[MCP Screenshot] Warning: could not publish " .. tostring(label) ..
                          ": " .. tostring(err))
                end
            end
            os.remove(staged)
            return true
        end
        if tries >= 30 then   -- ~half a second at 60fps; something is wrong
            print("[MCP Screenshot] Warning: capture never appeared for " .. tostring(label))
            -- An empty staged file may still be lying there.
            if staged then os.remove(staged) end
            return true
        end
        return false
    end

    local function onFrame()
        if attempt() then
            Runtime:removeEventListener("enterFrame", onFrame)
        end
    end
    Runtime:addEventListener("enterFrame", onFrame)
end

-- Capture screenshot
local function captureScreen()
    if not isRecording() then return end

    screenshotCount = screenshotCount + 1
    local filename = string.format("screenshot_%03d.jpg", screenshotCount)
    local fullPath = screenshotDir .. "/" .. filename
    -- Still .jpg: Solar2D picks the encoder from the extension.
    local staged = string.format("_staging_%03d.jpg", screenshotCount)

    display.save(display.currentStage, {{
        filename = staged,
        baseDir = system.TemporaryDirectory,
        captureOffscreenArea = false,
        isFullResolution = false
    }})

    publishWhenReady(staged, fullPath, filename)
end

-- Capture on demand, published as screenshot_<k>.jpg: a name that has never
-- existed, so the reader cannot be handed a leftover from an earlier request.
local function captureOnDemand(k)
    local filename = "screenshot_" .. tostring(k) .. ".jpg"
    local fullPath = screenshotDir .. "/" .. filename
    local staged = "_staging_" .. tostring(k) .. ".jpg"

    display.save(display.currentStage, {{
        filename = staged,
        baseDir = system.TemporaryDirectory,
        captureOffscreenArea = false,
        isFullResolution = false
    }})

    publishWhenReady(staged, fullPath, filename)
end

-- Check control file for recording commands
local function checkControl()
    local content = readControlFile()
    if not content then return end

    -- Bare "now" stays understood, for an older client.
    local k = content:match("^now:(%w+)$")
    if k then
        captureOnDemand(k)
        return
    end
    if content == "now" then
        captureOnDemand("latest")
        return
    end

    local duration = tonumber(content)
    if duration == nil then
        -- Not a number, ignore
        return
    elseif duration > 0 then
        if duration > MAX_RECORDING_SECONDS then
            print("[MCP Screenshot] Requested " .. duration .. "s exceeds max; capping at " .. MAX_RECORDING_SECONDS .. "s")
            duration = MAX_RECORDING_SECONDS
        end
        recordingEndTime = system.getTimer() + (duration * 1000)
        print("[MCP Screenshot] Recording for " .. duration .. " seconds (screenshots continue from #" .. (screenshotCount + 1) .. ")")
    elseif duration == 0 then
        -- Explicit stop command
        recordingEndTime = 0
        print("[MCP Screenshot] Recording stopped at screenshot #" .. screenshotCount)
    end
end

-- Initialize
clearScreenshotDir()
print("[MCP Screenshot] Module initialized - screenshots will be saved to: " .. screenshotDir)

-- Start timers
timer.performWithDelay(captureInterval, captureScreen, 0)
timer.performWithDelay(500, checkControl, 0)
'''

    screenshot_path = os.path.join(project_dir, "_mcp_screenshot.lua")
    with open(screenshot_path, 'w') as f:
        f.write(lua_screenshot)

    return screenshot_path


def create_touch_module(project_dir: str, project_name: str, launch_id: str) -> str:
    """Create a Lua file that handles touch simulation via control file."""
    launch_name = f"{project_name}_{launch_id}"
    control_file = os.path.join(tempfile.gettempdir(), f"solar2d_touch_{launch_name}.control")
    info_file = os.path.join(tempfile.gettempdir(), f"solar2d_display_{launch_name}.json")

    lua_touch = f'''
-- MCP Touch: Simulates touch events from control file commands
local launchId = "{launch_id}"
local controlFile = "{control_file}"
local infoFile = "{info_file}"
local checkInterval = 100  -- Check for commands every 100ms
local json = require("json")

-- Stored target from "began" phase for consistent event dispatch
local touchTarget = nil
local touchStartX, touchStartY = 0, 0

-- Overlay group — fades out after each interaction
local overlayGroup = nil
local fadeHandle = nil
local fadeDelay = 25000
local fadeTime = 5000

local function clearOverlay()
    if fadeHandle then
        transition.cancel(fadeHandle)
        fadeHandle = nil
    end
    if overlayGroup then
        display.remove(overlayGroup)
    end
    overlayGroup = display.newGroup()
    overlayGroup.alpha = 1
    -- Keep overlay on top every frame
    Runtime:addEventListener("enterFrame", function()
        if overlayGroup and overlayGroup.removeSelf then
            display.getCurrentStage():insert(overlayGroup)
        end
    end)
end

local function scheduleFade()
    local group = overlayGroup
    if not group then return end
    if fadeHandle then transition.cancel(fadeHandle) end
    fadeHandle = transition.to(group, {{
        delay = fadeDelay,
        time = fadeTime,
        alpha = 0,
        onComplete = function()
            if group and group.removeSelf then
                display.remove(group)
            end
            if overlayGroup == group then overlayGroup = nil end
            fadeHandle = nil
        end
    }})
end

-- Helper to read and consume control file
local function readControlFile()
    local file = io.open(controlFile, "r")
    if file then
        local content = file:read("*all")
        file:close()
        os.remove(controlFile)  -- Consume the command
        local token, command = content:match("^([^\\n]+)\\n(.*)$")
        if token == launchId then
            return command
        end
        print("[MCP Touch] Ignored control command for another launch")
    end
    return nil
end

-- Parse command string
local function parseCommand(content)
    local parts = {{}}
    for part in string.gmatch(content, "[^,]+") do
        table.insert(parts, part)
    end
    return parts
end

-- Check if object has a touch listener
local function hasTouchListener(obj)
    -- Check for touch_handler (used by some frameworks)
    if obj.touch_handler then return true end
    -- Check for _tableListeners.touch (standard Solar2D listener table)
    if obj._tableListeners and obj._tableListeners.touch then return true end
    -- Check for _functionListeners.touch
    if obj._functionListeners and obj._functionListeners.touch then return true end
    return false
end

-- Find the topmost touchable object at coordinates via hit testing
local function findHitObject(group, x, y)
    if not group or not group.numChildren then return nil end

    -- Traverse in reverse order (higher index = on top)
    for i = group.numChildren, 1, -1 do
        local child = group[i]
        if child and child.isVisible ~= false then
            -- Recurse into groups first
            if child.numChildren then
                local hit = findHitObject(child, x, y)
                if hit then return hit end
            end

            -- Check if this object is within bounds
            if child.contentBounds then
                local bounds = child.contentBounds
                if x >= bounds.xMin and x <= bounds.xMax and
                   y >= bounds.yMin and y <= bounds.yMax then
                    -- Check if it has a touch listener
                    if hasTouchListener(child) then
                        return child
                    end
                end
            end
        end
    end
    return nil
end

-- Dispatch a touch event to appropriate target
local function dispatchTouch(phase, x, y)
    local target = nil

    if phase == "began" then
        -- Find the topmost touchable object at this point
        target = findHitObject(display.getCurrentStage(), x, y)
        if target then
            touchTarget = target
            touchStartX, touchStartY = x, y
        end
    else
        target = touchTarget
        if phase == "ended" then
            touchTarget = nil
        end
    end

    local event = {{
        name = "touch",
        phase = phase,
        x = x,
        y = y,
        xStart = touchStartX or x,
        yStart = touchStartY or y,
        time = system.getTimer(),
        target = target
    }}

    if target then
        target:dispatchEvent(event)
    else
        -- Fallback to Runtime if no target found
        Runtime:dispatchEvent(event)
    end
end

-- Write display info to file
local function writeDisplayInfo()
    local info = {{
        contentWidth = display.contentWidth,
        contentHeight = display.contentHeight,
        actualContentWidth = display.actualContentWidth,
        actualContentHeight = display.actualContentHeight,
        screenOriginX = display.screenOriginX,
        screenOriginY = display.screenOriginY,
        launchId = launchId
    }}

    local pending = infoFile .. ".pending"
    local file = io.open(pending, "w")
    if file then
        file:write(json.encode(info))
        file:close()
        local ok, err = os.rename(pending, infoFile)
        if not ok then
            os.remove(pending)
            print("[MCP Touch] Could not publish display readiness: " .. tostring(err))
        end
    end
end

-- Show a persistent visual indicator at the tap/drag point
local function showTouchIndicator(x, y, isTap)
    local radius = isTap and 20 or 12
    local circle = display.newCircle(overlayGroup, x, y, radius)
    circle:setFillColor(1, 0, 0, 0.5)
    circle:setStrokeColor(1, 1, 0)
    circle.strokeWidth = 3
    local hLine = display.newLine(overlayGroup, x - radius, y, x + radius, y)
    hLine:setStrokeColor(1, 1, 0, 0.8)
    hLine.strokeWidth = 1
    local vLine = display.newLine(overlayGroup, x, y - radius, x, y + radius)
    vLine:setStrokeColor(1, 1, 0, 0.8)
    vLine.strokeWidth = 1
end

-- Show a persistent trail dot during drag movement
local function showDragDot(x, y)
    local dot = display.newCircle(overlayGroup, x, y, 5)
    dot:setFillColor(1, 0.5, 0, 0.6)
end

-- Show a persistent rectangle around a found object
local function showFindRect(x1, y1, x2, y2, label)
    local w = x2 - x1
    local h = y2 - y1
    local cx = x1 + w / 2
    local cy = y1 + h / 2
    local rect = display.newRect(overlayGroup, cx, cy, w, h)
    rect:setFillColor(0, 0, 0, 0)
    rect:setStrokeColor(1, 0, 0)
    rect.strokeWidth = 3
    if label and label ~= "" then
        local labelBg = display.newRoundedRect(overlayGroup, cx, y1 - 16, 10, 24, 4)
        local labelObj = display.newText(overlayGroup, label, cx, y1 - 16, native.systemFontBold, 18)
        labelObj:setFillColor(1, 0, 0)
        labelBg.width = labelObj.width + 12
        labelBg:setFillColor(0, 0, 0, 0.7)
    end
end

-- Execute a tap at coordinates
local function executeTap(x, y)
    print("[MCP Touch] Tap at (" .. x .. ", " .. y .. ")")
    clearOverlay()
    showTouchIndicator(x, y, true)
    scheduleFade()
    dispatchTouch("began", x, y)
    timer.performWithDelay(50, function()
        dispatchTouch("ended", x, y)
    end)
end

-- Execute a drag from (x1,y1) to (x2,y2) over duration ms
local function executeDrag(x1, y1, x2, y2, duration)
    print("[MCP Touch] Drag from (" .. x1 .. ", " .. y1 .. ") to (" .. x2 .. ", " .. y2 .. ") over " .. duration .. "ms")
    clearOverlay()
    showTouchIndicator(x1, y1, false)
    local steps = math.max(1, math.floor(duration / 16))
    local stepDelay = duration / steps
    dispatchTouch("began", x1, y1)
    for i = 1, steps do
        timer.performWithDelay(math.floor(stepDelay * i), function()
            local t = i / steps
            local x = x1 + (x2 - x1) * t
            local y = y1 + (y2 - y1) * t
            if i % 3 == 0 then showDragDot(x, y) end
            dispatchTouch("moved", x, y)
            if i == steps then
                showTouchIndicator(x2, y2, false)
                scheduleFade()
                timer.performWithDelay(16, function()
                    dispatchTouch("ended", x2, y2)
                end)
            end
        end)
    end
end

-- Execute a find — draw rectangle around an area
local function executeFind(x1, y1, x2, y2, label)
    print("[MCP Touch] Find at (" .. x1 .. ", " .. y1 .. ") to (" .. x2 .. ", " .. y2 .. ")" .. (label and (" label=" .. label) or ""))
    clearOverlay()
    showFindRect(x1, y1, x2, y2, label)
end

-- Check control file for commands
local function checkControl()
    local content = readControlFile()
    if content then
        local parts = parseCommand(content)
        local cmd = parts[1]

        if cmd == "tap" then
            local x = tonumber(parts[2])
            local y = tonumber(parts[3])
            if x and y then
                executeTap(x, y)
            else
                print("[MCP Touch] Invalid tap coordinates")
            end
        elseif cmd == "drag" then
            local x1 = tonumber(parts[2])
            local y1 = tonumber(parts[3])
            local x2 = tonumber(parts[4])
            local y2 = tonumber(parts[5])
            local dur = tonumber(parts[6])
            if x1 and y1 and x2 and y2 and dur then
                executeDrag(x1, y1, x2, y2, dur)
            else
                print("[MCP Touch] Invalid drag parameters")
            end
        elseif cmd == "find" then
            local x1 = tonumber(parts[2])
            local y1 = tonumber(parts[3])
            local x2 = tonumber(parts[4])
            local y2 = tonumber(parts[5])
            local label = parts[6] or ""
            if x1 and y1 and x2 and y2 then
                executeFind(x1, y1, x2, y2, label)
            else
                print("[MCP Touch] Invalid find parameters")
            end
        else
            print("[MCP Touch] Unknown command: " .. tostring(cmd))
        end
    end
end

-- Initialize
-- A timer runs only after every top-level require in main.lua has completed.
-- Publishing here therefore means screenshot and touch instrumentation loaded.
timer.performWithDelay(1, writeDisplayInfo)
print("[MCP Touch] Module initialized - listening for touch commands")

-- Start polling for commands
timer.performWithDelay(checkInterval, checkControl, 0)
'''

    touch_path = os.path.join(project_dir, "_mcp_touch.lua")
    with open(touch_path, 'w') as f:
        f.write(lua_touch)

    return touch_path


def create_touch_overlay_module(project_dir: str) -> str:
    """Create Lua module for visual touch indicators (crosshairs, drag trails, find rectangles)."""
    lua_overlay = '''
-- MCP Touch Overlay: Persistent visual indicators for tap/drag/find
-- Auto-generated by MCP server
-- Indicators persist until the next interaction clears them

local overlayGroup = nil
local fadeHandle = nil
local fadeDelay = 25000
local fadeTime = 5000

local function clearOverlay()
    if fadeHandle then
        transition.cancel(fadeHandle)
        fadeHandle = nil
    end
    if overlayGroup then
        display.remove(overlayGroup)
    end
    overlayGroup = display.newGroup()
    overlayGroup.alpha = 1
    Runtime:addEventListener("enterFrame", function()
        if overlayGroup and overlayGroup.removeSelf then
            display.getCurrentStage():insert(overlayGroup)
        end
    end)
end

local function scheduleFade()
    local group = overlayGroup
    if not group then return end
    if fadeHandle then transition.cancel(fadeHandle) end
    fadeHandle = transition.to(group, {
        delay = fadeDelay,
        time = fadeTime,
        alpha = 0,
        onComplete = function()
            if group and group.removeSelf then
                display.remove(group)
            end
            if overlayGroup == group then overlayGroup = nil end
            fadeHandle = nil
        end
    })
end

local function showIndicator(x, y, isTap)
    local radius = isTap and 30 or 18
    local g = display.newGroup()
    overlayGroup:insert(g)
    local circle = display.newCircle(g, x, y, radius)
    circle:setFillColor(1, 0, 0, 0.5)
    circle:setStrokeColor(1, 1, 0)
    circle.strokeWidth = 4
    local h = display.newLine(g, x - radius - 8, y, x + radius + 8, y)
    h:setStrokeColor(1, 1, 0, 0.9)
    h.strokeWidth = 3
    local v = display.newLine(g, x, y - radius - 8, x, y + radius + 8)
    v:setStrokeColor(1, 1, 0, 0.9)
    v.strokeWidth = 3
    local labelText = x .. ", " .. y
    local cw = display.contentWidth or 1080
    local ch = display.contentHeight or 1920
    local labelW, labelH = 160, 40
    local margin = ch * 0.15
    local lx = x + radius + 90
    if lx + labelW / 2 > cw then
        lx = x - radius - 90
    end
    local ly = y - radius - 30
    if y < margin then
        ly = y + radius + 30
    end
    local labelBg = display.newRoundedRect(g, lx, ly, labelW, labelH, 8)
    labelBg:setFillColor(0, 0, 0, 0.7)
    local label = display.newText(g, labelText, lx, ly, native.systemFontBold, 32)
    label:setFillColor(1, 1, 0)
end

local function showDragDot(x, y)
    local dot = display.newCircle(overlayGroup, x, y, 12)
    dot:setFillColor(1, 0.5, 0, 0.9)
    dot:setStrokeColor(1, 1, 0, 0.5)
    dot.strokeWidth = 2
end

local function showFindRect(x1, y1, x2, y2, label)
    local w = x2 - x1
    local h = y2 - y1
    local cx = x1 + w / 2
    local cy = y1 + h / 2
    local rect = display.newRect(overlayGroup, cx, cy, w, h)
    rect:setFillColor(0, 0, 0, 0)
    rect:setStrokeColor(1, 0, 0)
    rect.strokeWidth = 4
    if label and label ~= "" then
        local labelBg = display.newRoundedRect(overlayGroup, cx, y1 - 20, 10, 32, 6)
        local labelObj = display.newText(overlayGroup, label, cx, y1 - 20, native.systemFontBold, 24)
        labelObj:setFillColor(1, 0, 0)
        labelBg.width = labelObj.width + 16
        labelBg:setFillColor(0, 0, 0, 0.7)
    end
end

local function hookMcpTouch()
    local current_print = _G.print
    _G.print = function(...)
        local args = {...}
        local msg = tostring(args[1] or "")
        local tx, ty = msg:match("%[MCP Touch%] Tap at %((%d+), (%d+)%)")
        if tx and ty then
            clearOverlay()
            showIndicator(tonumber(tx), tonumber(ty), true)
            scheduleFade()
        end
        local dx1, dy1, dx2, dy2 = msg:match(
            "%[MCP Touch%] Drag from %((%d+), (%d+)%) to %((%d+), (%d+)%)")
        if dx1 then
            clearOverlay()
            local x1, y1 = tonumber(dx1), tonumber(dy1)
            local x2, y2 = tonumber(dx2), tonumber(dy2)
            showIndicator(x1, y1, false)
            local numDots = 40
            for d = 1, numDots do
                local t = d / numDots
                local dotX = x1 + (x2 - x1) * t
                local dotY = y1 + (y2 - y1) * t
                timer.performWithDelay(d * 25, function()
                    showDragDot(dotX, dotY)
                end)
            end
            timer.performWithDelay(numDots * 25 + 100, function()
                showIndicator(x2, y2, false)
                scheduleFade()
            end)
        end
        local fx1, fy1, fx2, fy2 = msg:match(
            "%[MCP Touch%] Find at %((%d+), (%d+)%) to %((%d+), (%d+)%)")
        if fx1 then
            clearOverlay()
            local flabel = msg:match("label=(.+)$") or ""
            showFindRect(tonumber(fx1), tonumber(fy1), tonumber(fx2), tonumber(fy2), flabel)
        end
        current_print(...)
    end
end

timer.performWithDelay(1000, hookMcpTouch)
print("[MCP Touch Overlay] Initialized - persistent visual indicators enabled")
'''

    overlay_path = os.path.join(project_dir, "_mcp_touch_overlay.lua")
    with open(overlay_path, 'w') as f:
        f.write(lua_overlay)

    return overlay_path


def inject_module_into_main_lua(main_lua_path: str, module_name: str) -> bool:
    """Inject a require statement into main.lua if not already present."""
    try:
        # Work in bytes so cancellation can remove our line without changing
        # the project's original encoding or newline convention (notably CRLF
        # projects on Linux).
        path = Path(main_lua_path)
        content = path.read_bytes()

        require_str = f'require("{module_name}")'
        require_str_single = f"require('{module_name}')"

        # Check if already injected
        if require_str.encode() in content or require_str_single.encode() in content:
            return False  # Already present

        lines = content.splitlines(keepends=True)
        line_text = [line.rstrip(b'\r\n').decode('utf-8', 'replace') for line in lines]
        if b'\r\n' in content:
            newline = b'\r\n'
        elif b'\n' in content:
            newline = b'\n'
        elif b'\r' in content:
            newline = b'\r'
        else:
            newline = b'\n'

        # Find the best insertion point
        # Look for mobdebug line, or first require, or beginning
        insert_index = 0

        for i, line in enumerate(line_text):
            # Insert after mobdebug if present
            if 'mobdebug' in line.lower() and 'require' in line:
                insert_index = i + 1
                break
            # Otherwise, insert before first require that's not a comment
            elif 'require' in line and not line.strip().startswith('--'):
                insert_index = i
                break

        # If no requires found, insert after initial comments/blank lines
        if insert_index == 0:
            for i, line in enumerate(line_text):
                stripped = line.strip()
                if stripped and not stripped.startswith('--'):
                    insert_index = i
                    break

        # Insert the require line
        lines.insert(insert_index, f'{require_str}  -- Auto-injected by MCP server'.encode() + newline)

        # Write back to file
        path.write_bytes(b''.join(lines))

        return True  # Successfully injected

    except Exception:
        # If we can't modify the file, just return False
        return False


def inject_logger_into_main_lua(main_lua_path: str) -> bool:
    """Inject require("_mcp_logger") into main.lua if not already present."""
    try:
        with open(main_lua_path, 'r') as f:
            content = f.read()

        # Check if already injected
        if 'require("_mcp_logger")' in content or "require('_mcp_logger')" in content:
            return False  # Already present

        lines = content.split('\n')

        # Find the best insertion point
        # Look for mobdebug line, or first require, or beginning
        insert_index = 0

        for i, line in enumerate(lines):
            # Insert after mobdebug if present
            if 'mobdebug' in line.lower() and 'require' in line:
                insert_index = i + 1
                break
            # Otherwise, insert before first require that's not a comment
            elif 'require' in line and not line.strip().startswith('--'):
                insert_index = i
                break

        # If no requires found, insert after initial comments/blank lines
        if insert_index == 0:
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith('--'):
                    insert_index = i
                    break

        # Insert the require line
        lines.insert(insert_index, 'require("_mcp_logger")  -- Auto-injected by MCP server for log capture')

        # Write back to file
        with open(main_lua_path, 'w') as f:
            f.write('\n'.join(lines))

        return True  # Successfully injected

    except Exception:
        # If we can't modify the file, just return False
        return False


class _LaunchCancelled(Exception):
    """Preparation was abandoned before its process could be handed off."""


def _get_launch_lock() -> asyncio.Lock:
    """Return the launch mutex for this MCP server event loop."""
    global _launch_lock, _launch_lock_loop
    loop = asyncio.get_running_loop()
    if _launch_lock is None or _launch_lock_loop is not loop:
        _launch_lock = asyncio.Lock()
        _launch_lock_loop = loop
    return _launch_lock


def _launch_paths(project_name: str, launch_id: str) -> dict[str, str]:
    launch_name = f"{project_name}_{launch_id}"
    temp_dir = tempfile.gettempdir()
    return {
        "display_info_file": os.path.join(temp_dir, f"solar2d_display_{launch_name}.json"),
        "screenshot_control_file": os.path.join(temp_dir, f"solar2d_screenshots_{launch_name}.control"),
        "screenshot_dir": os.path.join(temp_dir, f"solar2d_screenshots_{launch_name}"),
        "touch_control_file": os.path.join(temp_dir, f"solar2d_touch_{launch_name}.control"),
    }


def _remove_launch_ipc(launch: dict) -> None:
    """Remove only files whose unguessable names belong to this launch."""
    for key in ("display_info_file", "screenshot_control_file", "touch_control_file"):
        value = launch.get(key)
        if not value:
            continue
        path = Path(value)
        path.unlink(missing_ok=True)
        path.with_name(f"{path.name}.pending").unlink(missing_ok=True)


def _helper_paths(project_dir: str) -> dict[str, str]:
    """Return helper files owned by this launch."""
    return {
        "logger": os.path.join(project_dir, "_mcp_logger.lua"),
        "screenshot": os.path.join(project_dir, "_mcp_screenshot.lua"),
        "touch": os.path.join(project_dir, "_mcp_touch.lua"),
        "touch_overlay": os.path.join(project_dir, "_mcp_touch_overlay.lua"),
    }


def _snapshot_owned_file(path: str) -> bytes | None:
    """Capture a file's pre-launch content so cancellation can restore it."""
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.read_bytes()


def _restore_owned_file(path: str, original: bytes | None, generated: bytes | None) -> None:
    """Restore an untouched helper, leaving post-launch user edits intact."""
    file_path = Path(path)
    if original is None:
        if generated is not None and file_path.exists() and file_path.read_bytes() == generated:
            file_path.unlink()
        return
    if generated is not None and file_path.exists() and file_path.read_bytes() == generated:
        file_path.write_bytes(original)


def _remove_injected_module(main_lua_path: str, module_name: str) -> None:
    """Remove one auto-injected require line without touching user code."""
    path = Path(main_lua_path)
    try:
        content = path.read_bytes()
    except OSError:
        return

    injected_lines = {
        f'require("{module_name}")  -- Auto-injected by MCP server',
        f"require('{module_name}')  -- Auto-injected by MCP server",
        f'require("{module_name}")  -- Auto-injected by MCP server for log capture',
        f"require('{module_name}')  -- Auto-injected by MCP server for log capture",
    }
    lines = content.splitlines(keepends=True)
    removed = False
    filtered = []
    for line in lines:
        line_text = line.rstrip(b"\r\n").decode("utf-8", "replace")
        if not removed and line_text in injected_lines:
            removed = True
            continue
        filtered.append(line)
    if not removed:
        return
    path.write_bytes(b"".join(filtered))


def _restore_launch_artifacts(launch: dict) -> None:
    """Undo only the helper files and require lines created for this launch."""
    main_lua_path = launch.get("main_lua")
    if main_lua_path:
        for flag, module_name in (
            ("logger_injected", "_mcp_logger"),
            ("screenshot_injected", "_mcp_screenshot"),
            ("touch_injected", "_mcp_touch"),
            ("touch_overlay_injected", "_mcp_touch_overlay"),
        ):
            if launch.get(flag):
                _remove_injected_module(main_lua_path, module_name)

    helper_backups = launch.get("helper_backups") or {}
    generated_helpers = launch.get("generated_helpers") or {}
    for path, original in helper_backups.items():
        _restore_owned_file(path, original, generated_helpers.get(path))


def _stop_launch(launch: dict) -> None:
    try:
        process = launch.get("process")
        if process is not None:
            stop_process(process)
    finally:
        try:
            _restore_launch_artifacts(launch)
        finally:
            _remove_launch_ipc(launch)


def _prepare_and_spawn(
    *,
    cmd: list[str],
    project_dir: str,
    project_name: str,
    main_lua_path: str,
    log_file: str,
    launch_id: str,
    cancelled: threading.Event,
) -> dict:
    """Perform blocking file/process work away from the MCP event loop."""
    if cancelled.is_set():
        raise _LaunchCancelled

    previous_launches = list(running_projects.values())
    stop_tracked_simulators()
    for previous in previous_launches:
        _remove_launch_ipc(previous)

    if cancelled.is_set():
        raise _LaunchCancelled

    launch = {
        "launch_id": launch_id,
        "project_dir": project_dir,
        "main_lua": main_lua_path,
        "log_file": log_file,
        "helper_backups": {},
        "generated_helpers": {},
        **_launch_paths(project_name, launch_id),
    }
    _remove_launch_ipc(launch)

    for path in _helper_paths(project_dir).values():
        launch["helper_backups"][path] = _snapshot_owned_file(path)

    try:
        create_logging_wrapper(project_dir, log_file)
        logger_path = _helper_paths(project_dir)["logger"]
        launch["generated_helpers"][logger_path] = Path(logger_path).read_bytes()
        create_screenshot_module(project_dir, project_name, launch_id)
        screenshot_path = _helper_paths(project_dir)["screenshot"]
        launch["generated_helpers"][screenshot_path] = Path(screenshot_path).read_bytes()
        create_touch_module(project_dir, project_name, launch_id)
        touch_path = _helper_paths(project_dir)["touch"]
        launch["generated_helpers"][touch_path] = Path(touch_path).read_bytes()
        create_touch_overlay_module(project_dir)
        overlay_path = _helper_paths(project_dir)["touch_overlay"]
        launch["generated_helpers"][overlay_path] = Path(overlay_path).read_bytes()

        launch["logger_injected"] = inject_module_into_main_lua(main_lua_path, "_mcp_logger")
        launch["screenshot_injected"] = inject_module_into_main_lua(main_lua_path, "_mcp_screenshot")
        launch["touch_injected"] = inject_module_into_main_lua(main_lua_path, "_mcp_touch")
        launch["touch_overlay_injected"] = inject_module_into_main_lua(main_lua_path, "_mcp_touch_overlay")

        if cancelled.is_set():
            raise _LaunchCancelled

        launch["started_at_ns"] = time.time_ns()
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        launch["pid"] = process.pid
        launch["process"] = process

        if cancelled.is_set():
            _stop_launch(launch)
            raise _LaunchCancelled

        return launch
    except Exception:
        # A helper generator may fail after writing its file. Capture any such
        # partial output before rollback so newly-created files are removed and
        # pre-existing files are restored safely.
        for path in _helper_paths(project_dir).values():
            if path not in launch["generated_helpers"] and Path(path).exists():
                launch["generated_helpers"][path] = Path(path).read_bytes()
        _stop_launch(launch)
        raise


def _read_launch_readiness(launch: dict) -> bool:
    info_path = Path(launch["display_info_file"])
    try:
        stat = info_path.stat()
        with info_path.open() as file:
            info = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    return (
        info.get("launchId") == launch["launch_id"]
        and stat.st_mtime_ns >= launch["started_at_ns"]
    )


async def _wait_for_launch(launch: dict, deadline: float) -> tuple[str, int | None]:
    while True:
        return_code = launch["process"].poll()
        if return_code is not None:
            return "exited", return_code
        if _read_launch_readiness(launch):
            return "ready", None

        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return "timeout", None
        await asyncio.sleep(min(READINESS_POLL_SECONDS, remaining))


async def _cleanup_launch(launch: dict) -> None:
    current = running_projects.get(launch["project_dir"])
    if current is launch or (
        current is not None and current.get("launch_id") == launch["launch_id"]
    ):
        running_projects.pop(launch["project_dir"], None)
    await asyncio.to_thread(_stop_launch, launch)


def _abandon_preparation(
    task: asyncio.Task,
    cancelled: threading.Event,
    finished: Callable[[], None],
) -> None:
    """Ensure a worker that outlives its caller cannot leak a simulator."""
    cancelled.set()

    def cleanup_if_spawned(done: asyncio.Task) -> None:
        async def finish_abandoned_launch() -> None:
            try:
                launch = done.result()
            except (Exception, asyncio.CancelledError):
                pass
            else:
                await _cleanup_launch(launch)
            finally:
                finished()

        asyncio.create_task(finish_abandoned_launch())

    task.add_done_callback(cleanup_if_spawned)


async def _handle_owned_launch(
    arguments: dict,
    abandon: Callable[[asyncio.Task, threading.Event], None],
    release_after: Callable[[asyncio.Task], None],
) -> list[TextContent]:
    """Handle run_solar2d_project tool call."""
    project_path = arguments.get("project_path")
    debug = arguments.get("debug", True)
    no_console = arguments.get("no_console", False)

    if not project_path:
        return [TextContent(type="text", text="Error: project_path is required")]

    # Check if Solar2D is configured
    simulator_path, detected_paths, needs_confirmation = config.get_simulator_or_detect()

    if needs_confirmation:
        # Prompt user to configure first
        lines = ["Solar2D simulator needs to be configured before running projects.", ""]

        if detected_paths:
            lines.append("Detected simulators:")
            for path in detected_paths:
                lines.append(f"  - {path}")
            lines.append("")
            lines.append("Please use the configure_solar2d tool to confirm or select a simulator:")
            lines.append("  - Call configure_solar2d with confirm=true to use the detected path")
            lines.append("  - Or provide a specific path with simulator_path=\"...\"")
        else:
            lines.append("No Solar2D simulators were detected.")
            lines.append("Please use the configure_solar2d tool to set the simulator path:")
            lines.append("  configure_solar2d(simulator_path=\"/path/to/Corona Simulator\")")

        return [TextContent(type="text", text="\n".join(lines))]

    if not simulator_path or not os.path.exists(simulator_path):
        return [TextContent(
            type="text",
            text="Error: Solar2D Simulator not found. Please run configure_solar2d to set the path."
        )]

    # Find main.lua
    main_lua_path = find_main_lua(project_path)
    project_dir = str(Path(main_lua_path).parent)

    if not os.path.exists(main_lua_path):
        return [TextContent(
            type="text",
            text=f"Error: main.lua not found at {main_lua_path}"
        )]

    project_name = Path(project_dir).name
    log_file = os.path.join(tempfile.gettempdir(), f"corona_log_{project_name}.txt")
    launch_id = uuid.uuid4().hex

    cmd = [simulator_path]
    if platform.system() == "Darwin":
        if no_console:
            cmd.extend(["-no-console", "YES"])
        if debug:
            cmd.extend(["-debug", "1"])
        cmd.extend(["-project", main_lua_path])
    else:
        cmd.append(main_lua_path)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + LAUNCH_TIMEOUT_SECONDS
    cancelled = threading.Event()
    preparation = asyncio.create_task(asyncio.to_thread(
        _prepare_and_spawn,
        cmd=cmd,
        project_dir=project_dir,
        project_name=project_name,
        main_lua_path=main_lua_path,
        log_file=log_file,
        launch_id=launch_id,
        cancelled=cancelled,
    ))

    try:
        launch = await asyncio.wait_for(
            asyncio.shield(preparation),
            timeout=max(0, deadline - loop.time()),
        )
    except asyncio.TimeoutError:
        abandon(preparation, cancelled)
        return [TextContent(
            type="text",
            text=(
                f"Error: Solar2D launch timed out after {LAUNCH_TIMEOUT_SECONDS:g}s "
                "during project setup/start. The MCP connection is healthy; "
                "only this launch is being stopped."
            ),
        )]
    except asyncio.CancelledError:
        abandon(preparation, cancelled)
        raise
    except _LaunchCancelled:
        return [TextContent(type="text", text="Error: Solar2D launch was cancelled during setup.")]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error launching Solar2D Simulator during project setup/start: {str(e)}"
        )]

    running_projects[project_dir] = launch
    try:
        readiness, return_code = await _wait_for_launch(launch, deadline)
    except asyncio.CancelledError:
        release_after(asyncio.create_task(_cleanup_launch(launch)))
        raise

    if readiness != "ready":
        cleanup = asyncio.create_task(_cleanup_launch(launch))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            release_after(cleanup)
            raise
        if readiness == "exited":
            detail = f"exited with code {return_code}"
        else:
            detail = f"did not publish launch-specific readiness within {LAUNCH_TIMEOUT_SECONDS:g}s"
        return [TextContent(
            type="text",
            text=(
                f"Error: Solar2D Simulator {detail} while waiting for instrumentation "
                f"(launch {launch_id}). The MCP connection is healthy; only this "
                "launch was stopped."
            ),
        )]

    status_lines = [
        "Logger injected into main.lua" if launch["logger_injected"] else "Logger already present in main.lua",
        (
            "Screenshot module injected into main.lua"
            if launch["screenshot_injected"]
            else "Screenshot module already present in main.lua"
        ),
        "Touch module injected into main.lua" if launch["touch_injected"] else "Touch module already present in main.lua",
    ]

    return [TextContent(
        type="text",
        text=(
            "Solar2D Simulator launched and instrumentation is ready!\n\n"
            f"Project: {main_lua_path}\n"
            f"PID: {launch['pid']}\n"
            f"Launch ID: {launch_id}\n"
            f"Log file: {log_file}\n"
            f"Screenshot dir: {launch['screenshot_dir']}\n"
            f"Debug: {debug}\n"
            f"No Console: {no_console}\n\n"
            f"{chr(10).join(status_lines)}\n\n"
            "All print() output will be captured automatically.\n"
            "Use read_solar2d_logs to view the console output.\n"
            "Use start_screenshot_recording to capture screenshots."
        ),
    )]


async def handle(arguments: dict) -> list[TextContent]:
    """Serialize launch ownership without blocking the MCP event loop."""
    launch_lock = _get_launch_lock()
    if launch_lock.locked():
        return [TextContent(
            type="text",
            text=(
                "Error: Another Solar2D launch is already in progress in this MCP server. "
                "The MCP connection is healthy; retry after that launch finishes."
            ),
        )]

    await launch_lock.acquire()
    handed_off = False

    def release_after(task: asyncio.Task) -> None:
        nonlocal handed_off
        handed_off = True
        task.add_done_callback(lambda _: launch_lock.release())

    def abandon(preparation: asyncio.Task, cancelled: threading.Event) -> None:
        nonlocal handed_off
        handed_off = True
        _abandon_preparation(preparation, cancelled, launch_lock.release)

    try:
        return await _handle_owned_launch(
            arguments,
            abandon=abandon,
            release_after=release_after,
        )
    finally:
        if not handed_off:
            launch_lock.release()
