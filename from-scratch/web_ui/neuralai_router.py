# neuralai_router.py
#
# Clean routing logic for NeuralAI
# - Local model is DEFAULT
# - Uplink ONLY for heavy tasks
# - Terminal ONLY for shell commands

from typing import Tuple, Optional


def neuralai_route(msg: str) -> Tuple[str, Optional[str]]:
    """
    Route messages to the correct handler.
    
    Returns:
        (route_type, tool_name)
        
    Routes:
        ("local", None) → Main SmolLM2 model
        ("uplink", None) → Neural Uplink agent network
        ("tool", "terminal") → Real shell execution
    """
    lower = msg.lower().strip()
    
    # 1. Terminal commands ONLY go to terminal
    terminal_prefixes = ["run ", "execute ", "shell ", "command "]
    if any(lower.startswith(p) for p in terminal_prefixes):
        return ("tool", "terminal")
    
    # 2. Uplink ONLY for heavy tasks
    uplink_keywords = [
        "research", "analyze", "debug", "explain deeply",
        "worldbuild", "simulate", "generate dataset",
        "compare", "break down", "step by step",
        "investigate", "architecture", "design a system",
        "multi-step", "pipeline", "workflow"
    ]
    
    if any(k in lower for k in uplink_keywords):
        return ("uplink", None)
    
    # 3. Long messages → Uplink
    if len(msg) > 200:
        return ("uplink", None)
    
    # 4. EVERYTHING ELSE → MAIN MODEL (default)
    return ("local", None)


def should_use_uplink(msg: str) -> bool:
    """Legacy compatibility - returns True only for uplink-worthy messages."""
    route, _ = neuralai_route(msg)
    return route == "uplink"


def strip_terminal_prefix(msg: str) -> str:
    """Strip terminal prefixes from message to get the actual command."""
    lower = msg.lower()
    prefixes = ["run ", "execute ", "shell ", "command "]
    for p in prefixes:
        if lower.startswith(p):
            return msg[len(p):].strip()
    return msg.strip()
