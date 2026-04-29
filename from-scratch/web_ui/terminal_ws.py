# terminal_ws.py
#
# Terminal WebSocket Endpoint
# - Accepts commands from frontend
# - Streams output back

from fastapi import APIRouter, WebSocket
from neuralai_engine import run_shell_command

router = APIRouter()


@router.websocket("/ws/terminal")
async def terminal_ws(ws: WebSocket):
    """
    WebSocket endpoint for terminal.
    Frontend connects, sends command, receives streamed output.
    """
    await ws.accept()
    
    try:
        cmd = await ws.receive_text()
        
        async for line in run_shell_command(cmd):
            await ws.send_text(line)
        
        await ws.send_text("[DONE]")
    except Exception as e:
        await ws.send_text(f"[Error] {str(e)}")
    finally:
        await ws.close()
