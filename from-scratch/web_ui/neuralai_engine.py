# neuralai_engine.py
#
# Unified NeuralAI Engine
# - Router (local/uplink/tool)
# - Local model wrapper with LoRA support
# - Uplink client (4 parallel agents)
# - Fusion logic
# - Tool calling (terminal)
# - Streaming helpers

import asyncio
from typing import AsyncGenerator, Dict, Any, List, Tuple
import aiohttp
import asyncio.subprocess as asp
import os
from pathlib import Path

# Ports
UPLINK_BASE = "http://localhost"
DIALOG_PORT = 7101
DATA_PORT = 7102
OPS_PORT = 7103
WORLD_PORT = 7104

# Model globals
model = None
tokenizer = None
model_error = None

def load_local_model():
    """Load the local model with LoRA adapter."""
    global model, tokenizer, model_error
    
    if model is not None or model_error:
        return
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        base_model = "HuggingFaceTB/SmolLM2-360M-Instruct"
        adapter_path = Path(__file__).parent.parent.parent / "checkpoints" / "final_model"
        
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token
        
        if adapter_path.exists():
            model = AutoModelForCausalLM.from_pretrained(
                str(adapter_path),
                device_map=None,
                torch_dtype="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map=None,
                torch_dtype="auto"
            )
        
        model.eval()
        model_error = None
    except Exception as e:
        model = None
        tokenizer = None
        model_error = str(e)


class LocalModel:
    """Local model wrapper."""
    
    def generate(self, prompt: str, max_new_tokens: int = 256, stream: bool = True) -> AsyncGenerator[str, None]:
        """Generate text from local model."""
        load_local_model()
        
        if model is None or tokenizer is None:
            # Fallback streaming
            text = f"[Local Model] I'm ready but the model isn't loaded. Your question: {prompt[:100]}..."
            for ch in text:
                yield ch
            return
        
        try:
            # Build prompt
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = tokenizer(full_prompt, return_tensors="pt")
            
            import torch
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Decode only the new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Stream character by character
            for ch in text:
                yield ch
                
        except Exception as e:
            text = f"[Model Error] {str(e)}"
            for ch in text:
                yield ch


# Global model instance
local_model = LocalModel()


def neuralai_route(msg: str) -> Tuple[str, str | None]:
    """
    Route message to appropriate handler.
    Returns (route_type, tool_name)
    - route_type: "local" | "uplink" | "tool"
    - tool_name: "terminal" | None
    """
    uplink_triggers = [
        "research", "analyze", "debug", "explain deeply",
        "worldbuild", "simulate", "generate dataset",
        "compare", "break down", "step by step",
        "plan", "design", "architecture", "system",
        "help me build", "help me create"
    ]
    
    tool_triggers = {
        "terminal": ["run ", "execute ", "shell ", "command "],
    }
    
    lower = msg.lower()
    
    # Check for tool triggers first
    for tool, keys in tool_triggers.items():
        if any(k in lower for k in keys):
            return ("tool", tool)
    
    # Check for uplink triggers
    if any(k in lower for k in uplink_triggers) or len(msg) > 200:
        return ("uplink", None)
    
    # Default to local
    return ("local", None)


async def neuralai_local(prompt: str) -> AsyncGenerator[str, None]:
    """Generate response using local model."""
    for token in local_model.generate(prompt, stream=True):
        yield token


async def _post_json(session, url: str, payload: dict) -> dict:
    """Post JSON and return response."""
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"HTTP {resp.status}", "agent": "unknown"}
    except Exception as e:
        return {"error": str(e), "agent": "unknown"}


async def neuralai_uplink(prompt: str) -> str:
    """
    Send prompt to all 4 agents in parallel and fuse responses.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [
            _post_json(session, f"{UPLINK_BASE}:{DIALOG_PORT}/task", {"goal": prompt, "task_id": "uplink-dialog"}),
            _post_json(session, f"{UPLINK_BASE}:{DATA_PORT}/task", {"goal": prompt, "task_id": "uplink-data"}),
            _post_json(session, f"{UPLINK_BASE}:{OPS_PORT}/task", {"goal": prompt, "task_id": "uplink-ops"}),
            _post_json(session, f"{UPLINK_BASE}:{WORLD_PORT}/task", {"goal": prompt, "task_id": "uplink-world"}),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    outputs = [r for r in results if not isinstance(r, Exception) and isinstance(r, dict)]
    return neuralai_fuse(outputs)


def neuralai_fuse(outputs: List[Dict[str, Any]]) -> str:
    """
    Fuse multiple agent outputs into coherent response.
    """
    if not outputs:
        return "[NeuralAI Uplink] No agent responses."
    
    parts = []
    for agent in outputs:
        resp = agent.get("result", agent.get("response", ""))
        if isinstance(resp, dict):
            resp = resp.get("result", str(resp))
        name = agent.get("agent", "agent")
        
        if resp and not resp.startswith("[") and len(resp) > 10:
            parts.append(f"**{name.upper()}**: {resp}")
    
    if not parts:
        return "[NeuralAI Uplink] Agents are processing. Please try again."
    
    return "\n\n".join(parts[:2])  # Top 2 responses


async def run_shell_command(cmd: str) -> AsyncGenerator[str, None]:
    """
    Execute a shell command and stream output.
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
    Execute terminal command from message.
    """
    lower = msg.lower()
    cmd = msg
    for prefix in ["run ", "execute ", "shell ", "command "]:
        if lower.startswith(prefix):
            cmd = msg[len(prefix):].strip()
            break
    
    async for line in run_shell_command(cmd):
        yield line


async def neuralai_tool_call(tool: str, msg: str) -> AsyncGenerator[str, None]:
    """
    Handle tool calls.
    """
    if tool == "terminal":
        yield "```bash\n"
        async for line in terminal_execute(msg):
            yield line
        yield "```\n"
        return
    
    yield f"[TOOL] Unknown tool '{tool}'"


async def stream_text(text: str) -> AsyncGenerator[str, None]:
    """Stream text character by character."""
    for ch in text:
        yield ch


async def neuralai_chat(msg: str) -> AsyncGenerator[str, None]:
    """
    Main chat entrypoint.
    Routes to local model, uplink, or tools.
    """
    route, tool = neuralai_route(msg)
    
    if route == "local":
        async for token in neuralai_local(msg):
            yield token
    
    elif route == "uplink":
        yield "[Neural Uplink] Routing to agent network...\n\n"
        response = await neuralai_uplink(msg)
        async for token in stream_text(response):
            yield token
    
    elif route == "tool":
        async for token in neuralai_tool_call(tool, msg):
            yield token
    
    else:
        async for token in stream_text(f"[NeuralAI] Unknown route for: {msg}"):
            yield token
