# tools_api.py
#
# Flask API endpoints for NeuralAI tools
# - Code execution (Python, JavaScript)
# - File management
# - Web fetching
# - Database operations
# - Git operations

import json
import sys
import os
from pathlib import Path
from flask import Blueprint, jsonify, request

# Add tools to path
TOOLS_PATH = Path(__file__).resolve().parent.parent.parent / "tools"
if str(TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(TOOLS_PATH))

try:
    from code_sandbox import CodeSandbox
    from file_manager import FileManager
    from web_fetcher import WebFetcher
    from db_connector import DatabaseConnector
    from git_assistant import GitAssistant
    TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"[Tools API] Import error: {e}")
    TOOLS_AVAILABLE = False
    CodeSandbox = None
    FileManager = None
    WebFetcher = None
    DatabaseConnector = None
    GitAssistant = None

# Create blueprint
tools_bp = Blueprint('tools', __name__, url_prefix='/api/tools')

# Initialize tools
if TOOLS_AVAILABLE:
    sandbox = CodeSandbox()
    file_mgr = FileManager(base_dir=str(Path(__file__).resolve().parent.parent.parent.parent))
    web_fetcher = WebFetcher()
    db_conn = DatabaseConnector()
    git_helper = GitAssistant()
else:
    sandbox = None
    file_mgr = None
    web_fetcher = None
    db_conn = None
    git_helper = None


# ========================================
# CODE EXECUTION
# ========================================

@tools_bp.route('/execute', methods=['POST'])
def execute_code():
    """Execute Python or JavaScript code."""
    if not sandbox:
        return jsonify({"error": "Code sandbox not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    code = data.get('code', '')
    language = data.get('language', 'python').lower()
    timeout = data.get('timeout', 30)
    
    if not code:
        return jsonify({"error": "No code provided", "success": False}), 400
    
    if language == 'python':
        result = sandbox.run_python(code, timeout=timeout)
    elif language in ('javascript', 'js'):
        result = sandbox.run_javascript(code, timeout=timeout)
    else:
        return jsonify({"error": f"Unsupported language: {language}", "success": False}), 400
    
    return jsonify({
        "success": result.get("success", False),
        "output": result.get("output", ""),
        "error": result.get("error", ""),
        "exit_code": result.get("exit_code", -1),
        "execution_time": result.get("execution_time", 0),
        "language": language
    })


@tools_bp.route('/execute/python', methods=['POST'])
def execute_python():
    """Execute Python code specifically."""
    data = request.get_json(silent=True) or {}
    data['language'] = 'python'
    return execute_code()


@tools_bp.route('/execute/javascript', methods=['POST'])
def execute_javascript():
    """Execute JavaScript code specifically."""
    data = request.get_json(silent=True) or {}
    data['language'] = 'javascript'
    return execute_code()


# ========================================
# FILE MANAGEMENT
# ========================================

@tools_bp.route('/files/list', methods=['POST'])
def list_files_tool():
    """List files in a directory."""
    if not file_mgr:
        return jsonify({"error": "File manager not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    path = data.get('path', '.')
    
    result = file_mgr.list_dir(path)
    
    return jsonify({
        "success": result.get("success", True),
        "path": path,
        "directories": result.get("directories", []),
        "files": result.get("files", [])
    })


@tools_bp.route('/files/read', methods=['POST'])
def read_file_tool():
    """Read a file."""
    if not file_mgr:
        return jsonify({"error": "File manager not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    filepath = data.get('path', '')
    
    if not filepath:
        return jsonify({"error": "No path provided", "success": False}), 400
    
    result = file_mgr.read_file(filepath)
    
    return jsonify({
        "success": result.get("success", False),
        "content": result.get("content", ""),
        "error": result.get("error", ""),
        "path": filepath
    })


@tools_bp.route('/files/write', methods=['POST'])
def write_file_tool():
    """Write content to a file."""
    if not file_mgr:
        return jsonify({"error": "File manager not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    filepath = data.get('path', '')
    content = data.get('content', '')
    
    if not filepath:
        return jsonify({"error": "No path provided", "success": False}), 400
    
    result = file_mgr.write_file(filepath, content)
    
    return jsonify({
        "success": result.get("success", False),
        "path": filepath,
        "bytes_written": result.get("bytes_written", 0),
        "error": result.get("error", "")
    })


@tools_bp.route('/files/search', methods=['POST'])
def search_files_tool():
    """Search for files by name or content."""
    if not file_mgr:
        return jsonify({"error": "File manager not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    query = data.get('query', '')
    path = data.get('path', '.')
    search_type = data.get('type', 'filename')  # filename or content
    
    if not query:
        return jsonify({"error": "No search query provided", "success": False}), 400
    
    if search_type == 'content':
        result = file_mgr.search(query, path, search_content=True)
    else:
        result = file_mgr.search(query, path, search_content=False)
    
    return jsonify({
        "success": result.get("success", True),
        "query": query,
        "matches": result.get("matches", []),
        "count": len(result.get("matches", []))
    })


# ========================================
# WEB FETCHING
# ========================================

@tools_bp.route('/web/fetch', methods=['POST'])
def fetch_url_tool():
    """Fetch content from a URL."""
    if not web_fetcher:
        return jsonify({"error": "Web fetcher not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    url = data.get('url', '')
    timeout = data.get('timeout', 30)
    
    if not url:
        return jsonify({"error": "No URL provided", "success": False}), 400
    
    result = web_fetcher.fetch(url, timeout=timeout)
    
    return jsonify({
        "success": result.get("success", False),
        "url": url,
        "content": result.get("content", ""),
        "title": result.get("title", ""),
        "links": result.get("links", [])[:20],  # Limit links
        "error": result.get("error", "")
    })


@tools_bp.route('/web/extract', methods=['POST'])
def extract_from_url():
    """Extract specific data from a URL."""
    if not web_fetcher:
        return jsonify({"error": "Web fetcher not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    url = data.get('url', '')
    extract_type = data.get('extract', 'text')  # text, links, meta, all
    
    if not url:
        return jsonify({"error": "No URL provided", "success": False}), 400
    
    result = web_fetcher.fetch(url)
    
    if not result.get("success"):
        return jsonify({"success": False, "error": result.get("error", "Fetch failed")})
    
    response = {"success": True, "url": url}
    
    if extract_type in ('text', 'all'):
        response['text'] = result.get('content', '')
    if extract_type in ('links', 'all'):
        response['links'] = result.get('links', [])
    if extract_type in ('meta', 'all'):
        response['meta'] = result.get('meta', {})
    
    return jsonify(response)


# ========================================
# DATABASE OPERATIONS
# ========================================

@tools_bp.route('/db/query', methods=['POST'])
def query_database():
    """Execute a SQL query."""
    if not db_conn:
        return jsonify({"error": "Database connector not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    db_path = data.get('database', '')
    query = data.get('query', '')
    params = data.get('params', [])
    
    if not query:
        return jsonify({"error": "No query provided", "success": False}), 400
    
    # Default to NeuralAI's database if not specified
    if not db_path:
        db_path = str(Path(__file__).resolve().parent / "neuralai.db")
    
    result = db_conn.execute_query(db_path, query, params)
    
    return jsonify({
        "success": result.get("success", False),
        "rows": result.get("rows", []),
        "row_count": len(result.get("rows", [])),
        "columns": result.get("columns", []),
        "error": result.get("error", "")
    })


@tools_bp.route('/db/schema', methods=['POST'])
def get_database_schema():
    """Get database schema."""
    if not db_conn:
        return jsonify({"error": "Database connector not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    db_path = data.get('database', '')
    
    if not db_path:
        db_path = str(Path(__file__).resolve().parent / "neuralai.db")
    
    result = db_conn.get_schema(db_path)
    
    return jsonify({
        "success": result.get("success", False),
        "tables": result.get("tables", []),
        "error": result.get("error", "")
    })


@tools_bp.route('/db/tables', methods=['POST'])
def list_tables():
    """List all tables in database."""
    if not db_conn:
        return jsonify({"error": "Database connector not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    db_path = data.get('database', '')
    
    if not db_path:
        db_path = str(Path(__file__).resolve().parent / "neuralai.db")
    
    result = db_conn.list_tables(db_path)
    
    return jsonify({
        "success": result.get("success", False),
        "tables": result.get("tables", []),
        "error": result.get("error", "")
    })


# ========================================
# GIT OPERATIONS
# ========================================

@tools_bp.route('/git/status', methods=['POST'])
def git_status():
    """Get git status."""
    if not git_helper:
        return jsonify({"error": "Git assistant not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    repo_path = data.get('path', '.')
    
    result = git_helper.status(repo_path)
    
    return jsonify({
        "success": result.get("success", False),
        "branch": result.get("branch", ""),
        "staged": result.get("staged", []),
        "modified": result.get("modified", []),
        "untracked": result.get("untracked", []),
        "ahead": result.get("ahead", 0),
        "behind": result.get("behind", 0),
        "error": result.get("error", "")
    })


@tools_bp.route('/git/log', methods=['POST'])
def git_log():
    """Get git log."""
    if not git_helper:
        return jsonify({"error": "Git assistant not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    repo_path = data.get('path', '.')
    limit = data.get('limit', 10)
    
    result = git_helper.log(repo_path, limit=limit)
    
    return jsonify({
        "success": result.get("success", False),
        "commits": result.get("commits", []),
        "count": len(result.get("commits", [])),
        "error": result.get("error", "")
    })


@tools_bp.route('/git/diff', methods=['POST'])
def git_diff():
    """Get git diff."""
    if not git_helper:
        return jsonify({"error": "Git assistant not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    repo_path = data.get('path', '.')
    staged = data.get('staged', False)
    
    result = git_helper.diff(repo_path, staged=staged)
    
    return jsonify({
        "success": result.get("success", False),
        "diff": result.get("diff", ""),
        "error": result.get("error", "")
    })


@tools_bp.route('/git/commit', methods=['POST'])
def git_commit():
    """Create a git commit."""
    if not git_helper:
        return jsonify({"error": "Git assistant not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    repo_path = data.get('path', '.')
    message = data.get('message', '')
    add_all = data.get('add_all', True)
    
    if not message:
        return jsonify({"error": "Commit message required", "success": False}), 400
    
    result = git_helper.commit(repo_path, message, add_all=add_all)
    
    return jsonify({
        "success": result.get("success", False),
        "commit_hash": result.get("commit_hash", ""),
        "message": message,
        "error": result.get("error", "")
    })


@tools_bp.route('/git/push', methods=['POST'])
def git_push():
    """Push to remote."""
    if not git_helper:
        return jsonify({"error": "Git assistant not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    repo_path = data.get('path', '.')
    remote = data.get('remote', 'origin')
    branch = data.get('branch', None)
    
    result = git_helper.push(repo_path, remote=remote, branch=branch)
    
    return jsonify({
        "success": result.get("success", False),
        "output": result.get("output", ""),
        "error": result.get("error", "")
    })


@tools_bp.route('/git/branch', methods=['POST'])
def git_branch():
    """List or create branches."""
    if not git_helper:
        return jsonify({"error": "Git assistant not available", "success": False}), 503
    
    data = request.get_json(silent=True) or {}
    repo_path = data.get('path', '.')
    action = data.get('action', 'list')  # list, create, switch
    branch_name = data.get('branch', '')
    
    if action == 'list':
        result = git_helper.list_branches(repo_path)
        return jsonify({
            "success": result.get("success", False),
            "branches": result.get("branches", []),
            "current": result.get("current", ""),
            "error": result.get("error", "")
        })
    
    if action == 'create' and branch_name:
        result = git_helper.create_branch(repo_path, branch_name)
        return jsonify({
            "success": result.get("success", False),
            "branch": branch_name,
            "error": result.get("error", "")
        })
    
    return jsonify({"error": "Invalid action or missing branch name", "success": False}), 400


# ========================================
# TOOL STATUS
# ========================================

@tools_bp.route('/status', methods=['GET'])
def tools_status():
    """Get status of all tools."""
    return jsonify({
        "available": TOOLS_AVAILABLE,
        "tools": {
            "code_sandbox": sandbox is not None,
            "file_manager": file_mgr is not None,
            "web_fetcher": web_fetcher is not None,
            "database": db_conn is not None,
            "git": git_helper is not None
        }
    })


# ========================================
# HELPER: Execute tool by name
# ========================================

def generate_code_with_model(prompt: str, language: str = "python") -> str:
    """
    Use the local model to generate code based on a prompt.
    Returns the generated code string.
    """
    try:
        # Import the model from neuralai_engine
        from neuralai_engine import local_model, tokenizer
        import torch
        
        if local_model is None or tokenizer is None:
            return None
        
        # Build a code generation prompt
        if language == "python":
            system_prompt = f"""You are a Python code generator. Write clean, working Python code.
Rules:
- Only output the code, no explanations
- Use standard library when possible
- Handle edge cases
- The code should be complete and runnable

User request: {prompt}

Write the Python code:"""
        else:
            system_prompt = f"""You are a JavaScript code generator. Write clean, working JavaScript code.
Rules:
- Only output the code, no explanations
- Use modern JavaScript (ES6+)
- Handle edge cases
- The code should be complete and runnable

User request: {prompt}

Write the JavaScript code:"""
        
        # Tokenize and generate
        inputs = tokenizer(system_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = local_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the code part (after the prompt)
        if "Write the " in generated:
            code_start = generated.rfind("Write the ")
            code = generated[code_start + len(f"Write the {language} code:"):]
        else:
            code = generated
        
        # Clean up - remove markdown code blocks if present
        if "```" in code:
            import re
            code_match = re.search(r'```(?:python|javascript|js)?\s*([\s\S]*?)```', code)
            if code_match:
                code = code_match.group(1).strip()
        
        return code.strip()
    
    except Exception as e:
        print(f"[Code Gen] Error: {e}")
        return None


def execute_tool(tool_name: str, params: dict) -> dict:
    """
    Execute a tool by name with given parameters.
    Used by the chat router to dispatch tool calls.
    """
    if not TOOLS_AVAILABLE:
        return {"success": False, "error": "Tools not available"}
    
    try:
        if tool_name == "terminal":
            # Terminal is handled separately via WebSocket
            return {"success": False, "error": "Use Terminal tab for shell commands"}
        
        # NEW: Code generation + execution
        if tool_name == "code_gen":
            prompt = params.get("prompt", params.get("query", "write a simple program"))
            language = params.get("language", "python")
            
            # First, generate code using the model
            generated_code = generate_code_with_model(prompt, language)
            
            if not generated_code:
                # Fallback to template code if model fails
                if language == "python":
                    generated_code = f'''# Generated Python code
# Request: {prompt}

def main():
    print("Hello from NeuralAI!")
    # Add your logic here
    pass

if __name__ == "__main__":
    main()
'''
                else:
                    generated_code = f'''// Generated JavaScript code
// Request: {prompt}

function main() {{
    console.log("Hello from NeuralAI!");
    // Add your logic here
}}

main();
'''
            
            # Now execute the generated code
            if language == "python":
                result = sandbox.run_python(generated_code, timeout=params.get("timeout", 30))
            elif language in ("javascript", "js"):
                result = sandbox.run_javascript(generated_code, timeout=params.get("timeout", 30))
            else:
                return {"success": False, "error": f"Unsupported language: {language}"}
            
            # Return both the generated code and execution result
            return {
                "success": result.get("success", False),
                "generated_code": generated_code,
                "output": result.get("output", ""),
                "error": result.get("error", ""),
                "execution_time": result.get("execution_time", 0),
                "language": language
            }
        
        if tool_name == "code_exec":
            code = params.get("code", "")
            language = params.get("language", "python")
            timeout = params.get("timeout", 30)
            if language == "python":
                return sandbox.run_python(code, timeout=timeout)
            elif language in ("javascript", "js"):
                return sandbox.run_javascript(code, timeout=timeout)
            else:
                return {"success": False, "error": f"Unsupported language: {language}"}
        
        if tool_name == "file_manager":
            action = params.get("action", "list")
            if action == "list":
                return file_mgr.list_dir(params.get("path", "."))
            elif action == "read":
                return file_mgr.read_file(params.get("path", ""))
            elif action == "write":
                return file_mgr.write_file(params.get("path", ""), params.get("content", ""))
            elif action == "search":
                return file_mgr.search(params.get("query", ""), params.get("path", "."), search_content=False)
            elif action == "search_content":
                return file_mgr.search(params.get("query", ""), params.get("path", "."), search_content=True)
            return {"success": False, "error": f"Unknown file action: {action}"}
        
        if tool_name == "web_fetcher":
            url = params.get("url", "")
            return web_fetcher.fetch(url, timeout=params.get("timeout", 30))
        
        if tool_name == "database":
            db_path = params.get("database", str(Path(__file__).resolve().parent / "neuralai.db"))
            query = params.get("query", "")
            if query.lower().startswith(("select", "pragma")):
                return db_conn.execute_query(db_path, query)
            elif "schema" in params.get("action", "").lower():
                return db_conn.get_schema(db_path)
            elif "tables" in params.get("action", "").lower():
                return db_conn.list_tables(db_path)
            return db_conn.execute_query(db_path, query)
        
        if tool_name == "git":
            action = params.get("action", "status")
            repo_path = params.get("path", ".")
            
            # Create a new GitAssistant with the specified path
            from git_assistant import GitAssistant as _GitAssistant
            git = _GitAssistant(repo_path)
            
            if action == "status":
                return git.status()
            elif action == "log":
                return git.log(limit=params.get("limit", 10))
            elif action == "diff":
                return git.diff(staged=params.get("staged", False))
            elif action == "commit":
                return git.commit(params.get("message", ""), add_all=params.get("add_all", True))
            elif action == "push":
                return git.push(remote=params.get("remote", "origin"))
            elif action == "branch":
                return git.list_branches()
            
            return {"success": False, "error": f"Unknown git action: {action}"}
        
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}
