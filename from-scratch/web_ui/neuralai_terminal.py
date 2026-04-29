# neuralai_terminal.py
#
# NeuralAI Terminal Engine
# - Executes shell commands
# - Streams output line-by-line
# - Supports long-running tasks
# - Integrates with Uplink Ops Agent

import asyncio
import asyncio.subprocess as asp
from typing import AsyncGenerator

async def run_shell_command(cmd: str) -> AsyncGenerator[str, None]:
    """
    Execute a shell command and stream output line-by-line.
    """
    proc = await asp.create_subprocess_shell(
        cmd,
        stdout=asp.PIPE,
        stderr=asp.PIPE,
    )

    # Stream STDOUT
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        yield line.decode()

    # Stream STDERR
    while True:
        line = await proc.stderr.readline()
        if not line:
            break
        yield line.decode()

    await proc.wait()


async def terminal_execute(msg: str) -> AsyncGenerator[str, None]:
    """
    NeuralAI Terminal entrypoint.
    Strips command prefixes and executes.
    """
    lower = msg.lower()
    cmd = msg
    for prefix in ["run ", "execute ", "shell ", "command "]:
        if lower.startswith(prefix):
            cmd = msg[len(prefix):].strip()
            break

    async for line in run_shell_command(cmd):
        yield line
