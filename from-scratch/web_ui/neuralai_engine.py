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
import sys
from pathlib import Path
import json
import time

# Import tools
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from tools.code_sandbox import CodeSandbox
    from tools.file_manager import FileManager
    from tools.web_fetcher import WebFetcher
    from tools.db_connector import DatabaseConnector
    from tools.git_assistant import GitAssistant
    
    code_sandbox = CodeSandbox()
    file_manager = FileManager()
    web_fetcher = WebFetcher()
    db_connector = DatabaseConnector()
    git_assistant = GitAssistant()
except ImportError as e:
    print(f"[NeuralAI Engine] Import Error: {e}")
    code_sandbox = None
    file_manager = None
    web_fetcher = None
    db_connector = None
    git_assistant = None

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
        adapter_path = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "final_model"
        
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
    
    async def generate(self, prompt: str, max_new_tokens: int = 256, stream: bool = True) -> AsyncGenerator[str, None]:
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
    Uses the clean router from neuralai_router.py if available.
    Returns (route_type, tool_name)
    - route_type: "local" | "uplink" | "tool"
    - tool_name: "terminal" | "code_exec" | "file_manager" | "web_fetcher" | "database" | "git" | None
    """
    try:
        from neuralai_router import neuralai_route as _route
        return _route(msg)
    except ImportError:
        # Fallback routing
        lower = msg.lower()
        if any(k in lower for k in ["research", "analyze", "debug", "explain deeply"]):
            return ("uplink", None)
        if len(msg) > 200:
            return ("uplink", None)
        return ("local", None)


async def neuralai_local(prompt: str) -> AsyncGenerator[str, None]:
    """Generate response using local model."""
    async for token in local_model.generate(prompt, stream=True):
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
    Handle tool calls using the specialized tool classes.
    """
    from neuralai_router import extract_tool_params
    params = extract_tool_params(msg, tool)
    
    if tool == "terminal":
        yield "```bash\n"
        async for line in terminal_execute(msg):
            yield line
        yield "```\n"
        return
        
    if tool == "code_exec":
        if not code_sandbox:
            yield "[Error] Code sandbox not available."
            return
        
        language = params.get("language", "python")
        code = params.get("code", "")
        
        if not code:
            yield "[Error] No code found to execute."
            return
            
        yield f"[Sandbox] Running {language} code...\n\n"
        
        import asyncio
        loop = asyncio.get_event_loop()
        if language == "python":
            result = await loop.run_in_executor(None, code_sandbox.run_python, code)
        else:
            result = await loop.run_in_executor(None, code_sandbox.run_javascript, code)
            
        if result["success"]:
            yield "```\n"
            yield result["output"]
            yield "\n```\n"
        else:
            yield "```\n"
            yield f"Error: {result['error']}\n"
            if result["output"]:
                yield f"Output: {result['output']}\n"
            yield "```\n"
        return

    if tool == "code_gen":
        if not code_sandbox:
            yield "[Error] Code sandbox not available."
            return
            
        yield "[NeuralAI] Writing and preparing code execution...\n\n"
        
        gen_prompt = f"Write a clean, single-file Python script to solve this: {msg}. Return ONLY the code inside a ```python ... ``` block."
        
        code_text = ""
        async for chunk in local_model.generate(gen_prompt, max_new_tokens=512, stream=True):
            code_text += chunk
            
        import re
        code_match = re.search(r'```python\s*([\s\S]*?)```', code_text, re.IGNORECASE)
        if not code_match:
            code_match = re.search(r'([\s\S]+)', code_text)
            
        if code_match:
            extracted_code = code_match.group(1).strip()
            yield "### Generated Code:\n"
            yield f"```python\n{extracted_code}\n```\n\n"
            
            yield "[Sandbox] Executing...\n\n"
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, code_sandbox.run_python, extracted_code)
            
            if result["success"]:
                yield "### Output:\n"
                yield "```\n"
                yield result["output"] or "(No output)"
                yield "\n```\n"
            else:
                yield "### Execution Error:\n"
                yield f"```\n{result['error']}\n```\n"
        else:
            yield f"[Error] Could not generate executable code: {code_text}"
        return

    if tool == "file_manager":
        if not file_manager:
            yield "[Error] File manager not available."
            return
            
        query = params.get("query", "")
        yield f"[FileManager] Searching for: {query}...\n\n"
        
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, file_manager.search, query)
        
        if result["success"]:
            if not result["results"]:
                yield f"No files found matching '{query}'."
            else:
                yield f"Found {result['total']} results:\n\n"
                for r in result["results"][:10]:
                    yield f"- {r['path']} (line {r['line']}): {r['match']}\n"
        else:
            yield f"Error searching files: {result['error']}"
        return

    if tool == "web_fetcher":
        if not web_fetcher:
            yield "[Error] Web fetcher not available."
            return
            
        url = params.get("url", "")
        if not url:
            yield "[Error] No URL provided."
            return
            
        yield f"[WebFetcher] Fetching {url}...\n\n"
        
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, web_fetcher.fetch, url)
        
        if result["success"]:
            yield f"### {result['title']}\n\n"
            yield result["text"][:2000] + ("..." if len(result["text"]) > 2000 else "")
        else:
            yield f"Error fetching URL: {result['error']}"
        return

    if tool == "database":
        if not db_connector:
            yield "[Error] Database connector not available."
            return
            
        query_text = params.get("query", msg)
        yield f"[Database] Processing request: {query_text[:100]}...\n\n"
        
        import asyncio
        loop = asyncio.get_event_loop()
        db_path = "/home/workspace/Projects/NeuralAI/from-scratch/web_ui/neuralai.db"
        await loop.run_in_executor(None, db_connector.connect_sqlite, db_path, "neuralai")
        
        import re
        sql_match = re.search(r'```sql\s*([\s\S]*?)```', msg, re.IGNORECASE)
        sql = sql_match.group(1).strip() if sql_match else None
        
        if not sql:
            lower = msg.lower()
            if "show tables" in lower or "list tables" in lower:
                result = await loop.run_in_executor(None, db_connector.tables)
                if result["success"]:
                    yield f"Tables in NeuralAI database:\n"
                    for t in result["tables"]:
                        yield f"- {t}\n"
                else:
                    yield f"Error: {result['error']}"
                return
            elif "schema" in lower:
                table_name = None
                words = lower.split()
                if "table" in words:
                    idx = words.index("table")
                    if idx + 1 < len(words):
                        table_name = words[idx+1].strip('?,.;')
                
                result = await loop.run_in_executor(None, db_connector.schema, table_name)
                if result["success"]:
                    for table in result["tables"]:
                        yield f"### Table: {table['name']}\n"
                        yield "| Column | Type | Nullable | PK |\n"
                        yield "| --- | --- | --- | --- |\n"
                        for col in table["columns"]:
                            yield f"| {col['name']} | {col['type']} | {'Yes' if col['nullable'] else 'No'} | {'✓' if col['primary_key'] else ''} |\n"
                        yield "\n"
                else:
                    yield f"Error: {result['error']}"
                return
            else:
                yield "I can query your database. Please provide a SQL command inside a ```sql ... ``` block.\n"
                return

        result = await loop.run_in_executor(None, db_connector.query, sql)
        if result["success"]:
            if result["rows"]:
                yield f"Query result ({result['row_count']} rows):\n\n"
                cols = result["columns"]
                yield "| " + " | ".join(cols) + " |\n"
                yield "| " + " | ".join(["---"] * len(cols)) + " |\n"
                for row in result["rows"][:20]:
                    yield "| " + " | ".join([str(row[c]) for c in cols]) + " |\n"
            else:
                yield f"Query successful. Rows affected: {result['row_count']}"
        else:
            yield f"SQL Error: {result['error']}"
        return

    if tool == "git":
        if not git_assistant:
            yield "[Error] Git assistant not available."
            return
            
        action = params.get("action", "status")
        yield f"[Git] Executing: {action}...\n\n"
        
        import asyncio
        loop = asyncio.get_event_loop()
        git_assistant.repo_path = Path("/home/workspace/Projects/NeuralAI")
        
        if action == "status":
            result = await loop.run_in_executor(None, git_assistant.status)
            if result["success"]:
                yield f"**Branch**: {result['branch']}\n"
                if result["staged"]: yield f"**Staged**: {', '.join(result['staged'])}\n"
                if result["modified"]: yield f"**Modified**: {', '.join(result['modified'])}\n"
                if result["untracked"]: yield f"**Untracked**: {', '.join(result['untracked'])}\n"
                if not (result["staged"] or result["modified"] or result["untracked"]):
                    yield "Working directory clean."
            else:
                yield f"Error: {result['error']}"
        elif action == "log":
            result = await loop.run_in_executor(None, git_assistant.log, 5)
            if result["success"]:
                for c in result["commits"]:
                    yield f"- `{c['hash']}` **{c['message']}** ({c['author']}, {c['date']})\n"
            else:
                yield f"Error: {result['error']}"
        elif action == "diff":
            result = await loop.run_in_executor(None, git_assistant.diff)
            if result["success"]:
                if result["diff"]:
                    yield "```diff\n"
                    yield result["diff"][:2000] + ("..." if len(result["diff"]) > 2000 else "")
                    yield "\n```"
                else:
                    yield "No changes to show."
            else:
                yield f"Error: {result['error']}"
        else:
            result = await loop.run_in_executor(None, git_assistant._run_git, [action])
            if result["success"]:
                yield "```\n"
                yield result["output"][:2000]
                yield "\n```"
            else:
                yield f"Git Error: {result['error']}"
        return

    yield f"[TOOL] Tool '{tool}' integration pending."


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
