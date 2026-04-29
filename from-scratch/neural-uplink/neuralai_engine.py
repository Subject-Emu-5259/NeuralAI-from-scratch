# neuralai_engine.py
#
# Unified NeuralAI Engine
# - Router
# - Local model wrapper
# - Uplink client (4 agents, parallel)
# - Fusion logic
# - Tool calling stub
# - Streaming helpers
# - Top-level neuralai_chat entrypoint

import asyncio
from typing import AsyncGenerator, Dict, Any, List, Tuple, Optional
import aiohttp
import json

# =========================
# CONFIG
# =========================

UPLINK_BASE = "http://localhost"
DIALOG_PORT = 7101
DATA_PORT = 7102
OPS_PORT = 7103
WORLD_PORT = 7104

# Local model will be set by the Flask app
_model = None
_tokenizer = None
_torch = None


def set_local_model(model, tokenizer, torch_module):
    """Set the local model from Flask app."""
    global _model, _tokenizer, _torch
    _model = model
    _tokenizer = tokenizer
    _torch = torch_module


# =========================
# ROUTER
# =========================

def neuralai_route(msg: str) -> Tuple[str, Optional[str]]:
    """
    Decide where to send the message:
    - ("local", None)
    - ("uplink", None)
    - ("tool", tool_name)
    """
    uplink_triggers = [
        "research", "analyze", "debug", "explain deeply",
        "worldbuild", "simulate", "generate dataset",
        "compare", "break down", "step by step",
        "plan", "design a system", "architect", "investigate",
        "strategy", "workflow", "pipeline", "help me build",
        "help me create", "create a plan", "deep dive", "outline"
    ]

    tool_triggers: Dict[str, List[str]] = {
        "terminal": ["run", "execute", "shell", "command", "bash"],
        "files": ["open file", "read file", "write file", "upload"],
        "web": ["search", "lookup", "find online", "google"],
    }

    lower = msg.lower()

    # Tool routing
    for tool, keys in tool_triggers.items():
        if any(k in lower for k in keys):
            return ("tool", tool)

    # Uplink routing
    if any(k in lower for k in uplink_triggers):
        return ("uplink", None)

    # Long messages → Uplink
    if len(msg) > 200:
        return ("uplink", None)

    # Default → Local model
    return ("local", None)


# =========================
# LOCAL MODEL PATH
# =========================

def neuralai_local_sync(prompt: str, max_new_tokens: int = 256) -> str:
    """Generate response from local model synchronously."""
    global _model, _tokenizer, _torch
    
    if _model is None or _tokenizer is None:
        return "[NeuralAI] Local model not loaded. Using fallback response."
    
    try:
        # Build prompt
        inputs = _tokenizer(prompt, return_tensors="pt")
        device = "cuda" if _torch and _torch.cuda.is_available() else "cpu"
        
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with _torch.no_grad():
            output = _model.generate(
                **inputs,
                max_new_tokens=max(32, min(max_new_tokens, 512)),
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.05,
                pad_token_id=_tokenizer.eos_token_id,
            )
        
        input_len = inputs["input_ids"].shape[-1]
        decoded = _tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
        
        if not decoded:
            decoded = _tokenizer.decode(output[0], skip_special_tokens=True).strip()
        
        return decoded or "[NeuralAI] Could not generate a response."
    
    except Exception as e:
        return f"[NeuralAI Model Error] {str(e)}"


async def neuralai_local(prompt: str, max_new_tokens: int = 256) -> AsyncGenerator[str, None]:
    """Stream tokens from the local model."""
    response = neuralai_local_sync(prompt, max_new_tokens)
    for token in response:
        yield token


# =========================
# UPLINK CLIENT
# =========================

async def _post_json(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"HTTP {resp.status}", "agent": "unknown"}
    except asyncio.TimeoutError:
        return {"error": "timeout", "agent": "unknown"}
    except Exception as e:
        return {"error": str(e), "agent": "unknown"}


async def neuralai_uplink(prompt: str) -> str:
    """Call all 4 Uplink agents in parallel and fuse their responses."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            _post_json(session, f"{UPLINK_BASE}:{DIALOG_PORT}/task", {"goal": prompt, "task_id": "uplink-task"}),
            _post_json(session, f"{UPLINK_BASE}:{DATA_PORT}/task", {"goal": prompt, "task_id": "uplink-task"}),
            _post_json(session, f"{UPLINK_BASE}:{OPS_PORT}/task", {"goal": prompt, "task_id": "uplink-task"}),
            _post_json(session, f"{UPLINK_BASE}:{WORLD_PORT}/task", {"goal": prompt, "task_id": "uplink-task"}),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

    outputs: List[Dict[str, Any]] = []
    for r in results:
        if isinstance(r, Exception):
            continue
        outputs.append(r)

    return neuralai_fuse(outputs)


# =========================
# FUSION LOGIC
# =========================

def neuralai_fuse(outputs: List[Dict[str, Any]]) -> str:
    """Combine agent outputs into a single coherent response."""
    if not outputs:
        return "[NeuralAI Uplink] No agent responses received."

    final_parts: List[str] = []

    for agent_out in outputs:
        if "error" in agent_out and agent_out.get("error"):
            continue
        
        agent_name = agent_out.get("agent", "unknown")
        resp = agent_out.get("result", agent_out.get("response", ""))
        
        if isinstance(resp, dict):
            resp = resp.get("response", str(resp))
        
        if not resp or not str(resp).strip():
            continue

        final_parts.append(f"({agent_name.upper()}) {resp}")

    if not final_parts:
        return "[NeuralAI Uplink] Agents responded but no valid outputs."

    return "\n\n".join(final_parts)


# =========================
# TOOL CALLING
# =========================

async def neuralai_tool_call(tool: str, msg: str) -> str:
    """Handle tool-based requests."""
    if tool == "terminal":
        return f"[TOOL: Terminal] Ready to execute commands. Use the Terminal tab for shell access."
    elif tool == "files":
        return f"[TOOL: Files] Use the Files tab to upload and manage documents. Attach files to messages for RAG."
    elif tool == "web":
        return f"[TOOL: Web] Web search capability coming soon."
    else:
        return f"[TOOL] Unknown tool '{tool}'."


# =========================
# STREAMING HELPERS
# =========================

async def stream_tokens(text: str) -> AsyncGenerator[str, None]:
    """Stream text character by character."""
    for ch in text:
        yield ch


# =========================
# TOP-LEVEL CHAT LOOP
# =========================

async def neuralai_chat(msg: str, use_uplink: bool = False, max_tokens: int = 256) -> AsyncGenerator[str, None]:
    """
    Main NeuralAI entrypoint.
    Yields tokens as they should be streamed to the UI.
    """
    # If use_uplink is explicitly set, respect it
    if use_uplink:
        response = await neuralai_uplink(msg)
        async for token in stream_tokens(response):
            yield token
        return
    
    # Otherwise, route automatically
    route, tool = neuralai_route(msg)

    if route == "local":
        async for token in neuralai_local(msg, max_tokens):
            yield token

    elif route == "uplink":
        response = await neuralai_uplink(msg)
        async for token in stream_tokens(response):
            yield token

    elif route == "tool":
        result = await neuralai_tool_call(tool, msg)
        async for token in stream_tokens(result):
            yield token

    else:
        fallback = f"[NeuralAI] Unknown route '{route}' for message: {msg}"
        async for token in stream_tokens(fallback):
            yield token


# =========================
# SYNC WRAPPER
# =========================

def neuralai_chat_sync(msg: str, use_uplink: bool = False, max_tokens: int = 256) -> str:
    """Convenience wrapper for blocking calls."""
    async def _run():
        out = []
        async for t in neuralai_chat(msg, use_uplink, max_tokens):
            out.append(t)
        return "".join(out)

    return asyncio.run(_run())


if __name__ == "__main__":
    # Quick manual test
    print("Testing NeuralAI Engine...")
    print(neuralai_chat_sync("research how neural uplink works and break it down step by step"))
