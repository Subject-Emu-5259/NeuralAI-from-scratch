# neuralai_router.py
#
# Clean routing logic for NeuralAI
# - Local model is DEFAULT
# - Uplink ONLY for heavy tasks
# - Tools: terminal, code_exec, file_manager, web_fetcher, database, git

from typing import Tuple, Optional, Dict, List
import re


# Tool detection patterns
TOOL_PATTERNS: Dict[str, List[str]] = {
    "terminal": ["shell ", "command ", "bash ", "terminal "],
    
    "code_exec": [
        "run this code", "execute this code", "run the code",
        "run this python", "execute this python", "run python code",
        "run python", "execute python",
        "run this javascript", "execute this javascript", "run js code",
        "run js", "execute js", "execute javascript",
        "execute the script", "run the script", "run code", "execute code"
    ],
    
    # NEW: Code generation requests - AI should write AND run code
    "code_gen": [
        "write code", "write a program", "write a script", "write a function",
        "write python", "write javascript", "write js",
        "create a program", "create a script", "create code",
        "generate code", "generate a program", "generate a script",
        "make a program", "make a script", "make code",
        "build a program", "build a script", "build code",
        "code for", "program for", "script for",
        "you write", "write your own", "you do it",
        "u do it", "u write", "you generate",
        # More flexible patterns
        "write a", "write some", "write the",
        "can you write", "could you write", "please write",
        "i want you to write", "help me write",
        "create a", "create some", "make a", "make some",
        "generate a", "generate some", "build a",
        "write me", "write us", "write something"
    ],
    
    "file_manager": [
        "read file", "write file", "create file", "delete file",
        "list files", "show files", "search files", "find files",
        "file operations", "open file", "edit file",
        "list directory", "show directory", "what files",
        "search for", "find in files", "files"
    ],
    
    "web_fetcher": [
        "fetch url", "get url", "fetch this url", "get content from",
        "scrape this page", "scrape url", "fetch page",
        "get webpage", "fetch website", "extract from url",
        "read website", "download page", "get page content", "fetch"
    ],
    
    "database": [
        "query database", "query the database", "database query",
        "connect to database", "sqlite query", "run sql",
        "execute sql", "database operations", "db query",
        "show tables", "get schema", "database schema"
    ],
    
    "git": [
        "git status", "git commit", "git push", "git pull", "git branch",
        "git log", "git diff", "git add", "git stash",
        "create a commit", "make a commit", "push to git",
        "create a pr", "create pull request", "git pr",
        "create issue", "git issue"
    ]
}


def neuralai_route(msg: str) -> Tuple[str, Optional[str]]:
    """
    Route messages to the correct handler.
    
    Returns:
        (route_type, tool_name)
        
    Routes:
        ("local", None) → Main SmolLM2 model
        ("uplink", None) → Neural Uplink agent network
        ("tool", "terminal") → Shell execution
        ("tool", "code_exec") → Code sandbox (run provided code)
        ("tool", "code_gen") → Code generation + execution (AI writes code)
        ("tool", "file_manager") → File operations
        ("tool", "web_fetcher") → Web fetching
        ("tool", "database") → Database operations
        ("tool", "git") → Git operations
    """
    lower = msg.lower().strip()
    
    # 1. Check for tool triggers FIRST (priority order)
    # Terminal commands have highest priority
    for prefix in TOOL_PATTERNS["terminal"]:
        if lower.startswith(prefix):
            return ("tool", "terminal")
    
    # Check code generation requests BEFORE code execution
    # This catches "write code", "you write", etc.
    for pattern in TOOL_PATTERNS["code_gen"]:
        if pattern in lower:
            return ("tool", "code_gen")
    
    # Check other tools (order matters for priority)
    tool_order = ["git", "database", "web_fetcher", "file_manager", "code_exec"]
    for tool in tool_order:
        for pattern in TOOL_PATTERNS[tool]:
            if pattern in lower:
                return ("tool", tool)
    
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


def detect_tool(msg: str) -> Optional[str]:
    """
    Detect if a message should trigger a tool.
    Returns tool name or None.
    """
    route, tool = neuralai_route(msg)
    if route == "tool":
        return tool
    return None


def extract_tool_params(msg: str, tool: str) -> Dict[str, str]:
    """
    Extract parameters for a tool from the message.
    Returns dict of extracted params.
    """
    lower = msg.lower()
    
    if tool == "terminal":
        # Extract the command
        for prefix in TOOL_PATTERNS["terminal"]:
            if lower.startswith(prefix):
                return {"command": msg[len(prefix):].strip()}
        return {"command": msg}
    
    if tool == "code_exec":
        # Check for language
        language = "python"
        if "javascript" in lower or "js" in lower:
            language = "javascript"
        
        # Extract code from message - look for code patterns
        # Try to find code in backticks first
        code_match = re.search(r'```(?:python|javascript|js)?\s*([\s\S]*?)```', msg)
        if code_match:
            return {"language": language, "code": code_match.group(1).strip()}
        
        # Try to find code after "code:" or similar
        code_patterns = [
            r'code[:\s]+(.+)$',
            r'run this (?:python|javascript|js)? code[:\s]+(.+)$',
            r'execute (?:this )?(?:python|javascript|js)? code[:\s]+(.+)$',
        ]
        for pattern in code_patterns:
            match = re.search(pattern, msg, re.IGNORECASE | re.DOTALL)
            if match:
                return {"language": language, "code": match.group(1).strip()}
        
        # If the message contains "print(" or other code indicators, extract that part
        code_indicators = ['print(', 'console.log(', 'def ', 'function ', 'import ', 'var ', 'let ', 'const ']
        for indicator in code_indicators:
            if indicator in msg:
                idx = msg.find(indicator)
                return {"language": language, "code": msg[idx:].strip()}
        
        # Fallback: return the message minus the trigger words
        trigger_words = ["run this python code:", "run this javascript code:", "run this js code:",
                        "run this code:", "execute this code:", "run code:", "execute code:",
                        "run python", "execute python", "run javascript", "execute javascript"]
        code = msg
        for trigger in trigger_words:
            if trigger in lower:
                idx = lower.find(trigger)
                code = msg[idx + len(trigger):].strip()
                break
        
        return {"language": language, "code": code}
    
    if tool == "web_fetcher":
        # Extract URL
        url_pattern = r'https?://[^\s]+'
        match = re.search(url_pattern, msg)
        if match:
            return {"url": match.group(0)}
        return {}
    
    if tool == "git":
        # Extract git command
        for pattern in TOOL_PATTERNS["git"]:
            if pattern in lower:
                return {"action": pattern.replace("git ", "").strip()}
        return {"action": "status"}
    
    if tool == "file_manager":
        # Extract file path or query
        query = msg
        for pattern in TOOL_PATTERNS["file_manager"]:
            if pattern in lower:
                idx = lower.find(pattern)
                query = msg[idx + len(pattern):].strip()
                break
        return {"query": query or msg}
    
    if tool == "database":
        # Extract SQL if present
        sql_keywords = ["select", "insert", "update", "delete", "create", "drop"]
        for kw in sql_keywords:
            if kw in lower:
                return {"has_sql": "true"}
        return {"query": msg}
    
    return {}


def should_use_uplink(msg: str) -> bool:
    """Legacy compatibility - returns True only for uplink-worthy messages."""
    route, _ = neuralai_route(msg)
    return route == "uplink"


def strip_terminal_prefix(msg: str) -> str:
    """Strip terminal prefixes from message to get the actual command."""
    lower = msg.lower()
    for p in TOOL_PATTERNS["terminal"]:
        if lower.startswith(p):
            return msg[len(p):].strip()
    return msg.strip()


# Tool descriptions for UI display
TOOL_DESCRIPTIONS = {
    "terminal": "Execute shell commands",
    "code_exec": "Run Python or JavaScript code safely",
    "file_manager": "Read, write, search, and manage files",
    "web_fetcher": "Fetch and parse web content",
    "database": "Query SQLite databases",
    "git": "Git operations (status, commit, push, branch, PR)"
}


def get_tool_help() -> str:
    """Get help text for all available tools."""
    help_text = "Available tools:\n\n"
    for tool, desc in TOOL_DESCRIPTIONS.items():
        help_text += f"• {tool}: {desc}\n"
    return help_text
