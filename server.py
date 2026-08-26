#!/usr/bin/env python3
"""
Solar2D MCP Server
A Model Context Protocol server for working with Solar2D (Corona SDK) projects.
"""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool

from resources import RESOURCES, read_resource
from runtime import shutdown_runtime
from tools import TOOLS, call_tool

# Initialize the MCP server
app = Server("solar2d-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for Solar2D projects."""
    return TOOLS


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    """Handle tool calls."""
    return await call_tool(name, arguments)


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    return RESOURCES


@app.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read a resource by URI."""
    return read_resource(uri)


async def main():
    """Run the MCP server and clean up its owned simulator on disconnect."""
    # Tool dispatch claims the slot lazily, so initialization is always
    # healthy even while another MCP session has the simulator.
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    finally:
        shutdown_runtime()


if __name__ == "__main__":
    asyncio.run(main())
