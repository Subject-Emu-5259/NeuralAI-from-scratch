#!/usr/bin/env python3
"""
NeuralAI Terminal 2.0 — Advanced Integrated Terminal
- PTY-based shell sessions
- Code execution with syntax checking
- File operations (read, write, edit, browse)
- Git integration
- Python/Node execution
- Package management (pip, npm)
- Process management
- Command history & autocomplete
- Code snippets
- Syntax highlighting
"""
import os
import pty
import select
import struct
import fcntl
import termios
import signal
import threading
import uuid
import time
import json
import subprocess
import re
from pathlib import Path
from flask import Blueprint, request, Response, jsonify

terminal_bp = Blueprint("terminal", __name__)

# In-memory session store
sessions = {}
command_history = {}
snippets = {
    "python": {
        "fib": "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "class": "class MyClass:\n    def __init__(self):\n        pass\n\n    def method(self):\n        pass",
        "main": 'if __name__ == "__main__":\n    pass',
        "args": "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--input', type=str)\nargs = parser.parse_args()",
        "http": "import requests\nresponse = requests.get('https://api.example.com')\nprint(response.json())",
        "json": "import json\nwith open('data.json', 'r') as f:\n    data = json.load(f)",
        "file": "with open('file.txt', 'r') as f:\n    content = f.read()",
        "async": "import asyncio\n\nasync def main():\n    pass\n\nasyncio.run(main())",
    },
    "bash": {
        "loop": "for i in {1..10}; do\n    echo $i\ndone",
        "if": "if [ condition ]; then\n    # code\nfi",
        "function": "function_name() {\n    # code\n}",
        "grep": "grep -r 'pattern' .",
        "find": "find . -name '*.py'",
        "tar": "tar -czvf archive.tar.gz directory/",
        "gitignore": "echo '*.pyc\\n__pycache__/\\n.env' > .gitignore",
    },
    "javascript": {
        "func": "function name(params) {\n    return;\n}",
        "arrow": "const func = (params) => {\n    return;\n};",
        "async": "async function fetchData() {\n    try {\n        const response = await fetch(url);\n        const data = await response.json();\n        return data;\n    } catch (error) {\n        console.error(error);\n    }\n}",
        "class": "class MyClass {\n    constructor() {}\n\n    method() {}\n}",
        "export": "export default function() {\n    // code\n}",
        "import": "import { module } from 'package';",
    }
}

class TerminalSession:
    def __init__(self, session_id: str, rows: int = 24, cols: int = 80):
        self.session_id = session_id
        self.rows = rows
        self.cols = cols
        self.master_fd = None
        self.pid = None
        self.started = False
        self.last_activity = time.time()
        self.env = os.environ.copy()
        self.env["TERM"] = "xterm-256color"
        self.env["HOME"] = os.path.expanduser("~")
        self.env["USER"] = os.environ.get("USER", "root")
        self.env["EDITOR"] = "nano"
        self.env["PAGER"] = "less"
        self.buffer = []  # scrollback buffer
        self.buffer_size = 5000
        self.history = []
        self.history_index = -1
        self.cwd = os.environ.get("HOME", "/root")
        self.env_vars = {}
        self.running_processes = {}
        self.last_command = ""
        self.output_handlers = {}

    def start(self):
        if self.started:
            return
        self.master_fd, slave_fd = pty.openpty()
        self.pid = os.fork()
        if self.pid == 0:
            # Child
            os.setsid()
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", self.rows, self.cols, 0, 0))
            os.dup2(slave_fd, 0)
            os.dup2(slave_fd, 1)
            os.dup2(slave_fd, 2)
            if slave_fd > 2:
                os.close(slave_fd)
            os.chdir(self.cwd)
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(sig, signal.SIG_DFL)
            os.execvp(os.environ.get("SHELL", "/bin/bash"), ["-bash", "--login"])
            os._exit(1)
        else:
            os.close(slave_fd)
            flags = fcntl.fcntl(self.master_fd, fcntl.F_GETFL)
            fcntl.fcntl(self.master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            self.started = True
            self.resize(self.rows, self.cols)

    def write(self, data: str):
        if not self.started or self.master_fd is None:
            return
        self.last_activity = time.time()
        try:
            os.write(self.master_fd, data.encode("utf-8"))
        except OSError:
            pass

    def read(self, timeout: float = 0.05) -> str:
        if not self.started or self.master_fd is None:
            return ""
        self.last_activity = time.time()
        output = []
        try:
            while True:
                r, _, _ = select.select([self.master_fd], [], [], timeout)
                if not r:
                    break
                try:
                    data = os.read(self.master_fd, 4096)
                    if not data:
                        break
                    decoded = data.decode("utf-8", errors="replace")
                    output.append(decoded)
                    self._buffer_add(decoded)
                except OSError:
                    break
        except Exception:
            pass
        return "".join(output)

    def _buffer_add(self, text: str):
        self.buffer.append(text)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]

    def resize(self, rows: int, cols: int):
        if self.master_fd is None:
            return
        self.rows = rows
        self.cols = cols
        try:
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        except Exception:
            pass

    def is_alive(self) -> bool:
        if not self.started:
            return False
        if self.pid is None:
            return False
        try:
            os.kill(self.pid, 0)
            return True
        except OSError:
            return False

    def stop(self):
        if self.pid is not None:
            try:
                os.kill(self.pid, signal.SIGTERM)
                time.sleep(0.2)
                os.kill(self.pid, signal.SIGKILL)
            except OSError:
                pass
            self.pid = None
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None

    def get_buffer(self) -> str:
        return "".join(self.buffer)

    def add_to_history(self, cmd: str):
        if cmd.strip() and (not self.history or self.history[-1] != cmd):
            self.history.append(cmd)
            if len(self.history) > 1000:
                self.history = self.history[-1000:]
            self.history_index = len(self.history)

    def get_history(self, index: int) -> str:
        if 0 <= index < len(self.history):
            return self.history[index]
        return ""


def cleanup_dead_sessions():
    dead = [sid for sid, sess in sessions.items() if not sess.is_alive()]
    for sid in dead:
        sessions[sid].stop()
        del sessions[sid]


def execute_python(code: str) -> dict:
    """Execute Python code and return result."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": "Execution timed out"}
    except Exception as e:
        return {"error": str(e)}


def execute_javascript(code: str) -> dict:
    """Execute JavaScript code using node."""
    try:
        result = subprocess.run(
            ["node", "-e", code],
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": "Execution timed out"}
    except FileNotFoundError:
        return {"error": "Node.js not installed"}
    except Exception as e:
        return {"error": str(e)}


def run_file(filepath: str) -> dict:
    """Run a file based on its extension."""
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}
    
    ext = path.suffix.lower()
    
    if ext == ".py":
        result = subprocess.run(
            ["python3", str(path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    
    elif ext == ".js":
        result = subprocess.run(
            ["node", str(path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    
    elif ext == ".sh":
        result = subprocess.run(
            ["bash", str(path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    
    else:
        return {"error": f"Unsupported file type: {ext}"}


def get_file_info(filepath: str) -> dict:
    """Get file information."""
    path = Path(filepath)
    if not path.exists():
        return {"error": "File not found"}
    
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "modified": time.ctime(stat.st_mtime),
        "is_dir": path.is_dir(),
        "is_file": path.is_file(),
        "extension": path.suffix,
        "permissions": oct(stat.st_mode)[-3:],
    }


def list_directory(dirpath: str = ".") -> dict:
    """List directory contents."""
    path = Path(dirpath)
    if not path.exists():
        return {"error": "Directory not found"}
    
    if not path.is_dir():
        return {"error": "Not a directory"}
    
    items = []
    for item in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        items.append({
            "name": item.name,
            "is_dir": item.is_dir(),
            "size": item.stat().st_size if item.is_file() else None,
        })
    
    return {"path": str(path.absolute()), "items": items}


def search_files(pattern: str, directory: str = ".") -> dict:
    """Search for files matching pattern."""
    path = Path(directory)
    matches = list(path.glob(pattern))
    return {
        "pattern": pattern,
        "directory": str(path.absolute()),
        "matches": [str(m.relative_to(path)) for m in matches[:50]]
    }


def check_syntax(code: str, language: str) -> dict:
    """Check code syntax without executing."""
    if language == "python":
        try:
            compile(code, "<string>", "exec")
            return {"valid": True, "message": "Syntax OK"}
        except SyntaxError as e:
            return {"valid": False, "line": e.lineno, "message": str(e)}
    
    elif language == "javascript":
        try:
            result = subprocess.run(
                ["node", "--check", "-e", code],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return {"valid": True, "message": "Syntax OK"}
            return {"valid": False, "message": result.stderr}
        except Exception as e:
            return {"valid": False, "message": str(e)}
    
    return {"error": f"Unsupported language: {language}"}


def get_git_status(directory: str = ".") -> dict:
    """Get git status."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=directory
        )
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=directory
        )
        return {
            "branch": branch.stdout.strip(),
            "status": result.stdout,
            "modified": len([l for l in result.stdout.split("\n") if l.strip()])
        }
    except Exception as e:
        return {"error": str(e)}


def get_process_list() -> dict:
    """Get list of running processes."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        processes = []
        for line in result.stdout.split("\n")[1:20]:  # Top 20
            if line.strip():
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    processes.append({
                        "user": parts[0],
                        "pid": parts[1],
                        "cpu": parts[2],
                        "mem": parts[3],
                        "command": parts[10][:50]
                    })
        return {"processes": processes}
    except Exception as e:
        return {"error": str(e)}


# Routes

@terminal_bp.route("/api/terminal/create", methods=["POST"])
def create_session():
    cleanup_dead_sessions()
    data = request.json or {}
    rows = int(data.get("rows", 24))
    cols = int(data.get("cols", 100))
    session_id = str(uuid.uuid4())[:8]
    sess = TerminalSession(session_id, rows=rows, cols=cols)
    sess.start()
    sessions[session_id] = sess
    command_history[session_id] = []
    return {
        "session_id": session_id, 
        "rows": rows, 
        "cols": cols,
        "cwd": sess.cwd,
        "welcome": """NeuralAI Terminal 2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ready. Type 'help' for commands.

Built-in Commands:
  help          - Show this help
  clear         - Clear terminal
  ls [path]     - List directory
  cd <path>     - Change directory
  cat <file>    - View file
  edit <file>   - Edit file
  run <file>    - Execute file (.py, .js, .sh)
  
Code Execution:
  python <code> - Execute Python
  js <code>    - Execute JavaScript
  check <lang> <code> - Syntax check
  
Git Commands:
  git status    - Repository status
  git log       - Commit history
  git diff      - Show changes
  
Snippets:
  snippet <lang> <name> - Insert code snippet
  snippets <lang> - List available snippets
  
Tools:
  ps            - List processes
  kill <pid>    - Kill process
  find <pattern> - Search files
  info <file>   - File details
  
Neural Integration:
  neural <msg>  - Ask NeuralAI
  uplink        - Check uplink status
  model         - Show model info
  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    }


@terminal_bp.route("/api/terminal/<session_id>/read", methods=["GET"])
def read_session(session_id):
    if session_id not in sessions:
        return {"error": "Session not found"}, 404
    sess = sessions[session_id]
    output = sess.read(timeout=0.1)
    return {
        "output": output,
        "alive": sess.is_alive(),
        "rows": sess.rows,
        "cols": sess.cols,
        "cwd": sess.cwd
    }


@terminal_bp.route("/api/terminal/<session_id>/write", methods=["POST"])
def write_session(session_id):
    if session_id not in sessions:
        return {"error": "Session not found"}, 404
    data = request.json or {}
    cmd = data.get("input", "")
    if cmd:
        # Handle special commands
        cmd_lower = cmd.strip().lower()
        
        # Track command history
        if session_id in command_history:
            command_history[session_id].append(cmd.strip())
        
        sessions[session_id].add_to_history(cmd)
        sessions[session_id].write(cmd)
    return {"ok": True}


@terminal_bp.route("/api/terminal/<session_id>/command", methods=["POST"])
def execute_command(session_id):
    """Execute a command with enhanced handling."""
    if session_id not in sessions:
        return {"error": "Session not found"}, 404
    
    data = request.json or {}
    cmd = data.get("command", "").strip()
    
    if not cmd:
        return {"error": "No command provided"}
    
    sess = sessions[session_id]
    parts = cmd.split()
    cmd_name = parts[0].lower() if parts else ""
    args = parts[1:] if len(parts) > 1 else []
    
    # Built-in command handlers
    if cmd_name == "help":
        return {"output": sess.get_buffer().split("Built-in Commands:")[1].split("━")[0] if "Built-in Commands" in sess.get_buffer() else "Type 'help' for commands"}
    
    elif cmd_name == "ls":
        path = args[0] if args else "."
        return list_directory(path)
    
    elif cmd_name == "cd":
        if not args:
            return {"error": "Usage: cd <path>"}
        new_path = Path(sess.cwd) / args[0]
        if new_path.exists() and new_path.is_dir():
            sess.cwd = str(new_path.resolve())
            return {"cwd": sess.cwd}
        return {"error": f"Directory not found: {args[0]}"}
    
    elif cmd_name == "cat":
        if not args:
            return {"error": "Usage: cat <file>"}
        filepath = Path(sess.cwd) / args[0]
        if filepath.exists() and filepath.is_file():
            try:
                content = filepath.read_text()
                return {"output": content[:10000]}  # Limit output
            except Exception as e:
                return {"error": str(e)}
        return {"error": f"File not found: {args[0]}"}
    
    elif cmd_name == "run":
        if not args:
            return {"error": "Usage: run <file>"}
        filepath = Path(sess.cwd) / args[0]
        return run_file(str(filepath))
    
    elif cmd_name == "python":
        code = " ".join(args)
        if not code:
            return {"error": "Usage: python <code>"}
        return execute_python(code)
    
    elif cmd_name == "js" or cmd_name == "javascript":
        code = " ".join(args)
        if not code:
            return {"error": "Usage: js <code>"}
        return execute_javascript(code)
    
    elif cmd_name == "check":
        if len(args) < 2:
            return {"error": "Usage: check <python|js> <code>"}
        lang = args[0]
        code = " ".join(args[1:])
        return check_syntax(code, lang)
    
    elif cmd_name == "snippet":
        if len(args) < 2:
            return {"error": "Usage: snippet <lang> <name>"}
        lang = args[0]
        name = args[1]
        if lang in snippets and name in snippets[lang]:
            return {"code": snippets[lang][name]}
        return {"error": f"Snippet not found: {lang}/{name}"}
    
    elif cmd_name == "snippets":
        lang = args[0] if args else None
        if lang and lang in snippets:
            return {"snippets": list(snippets[lang].keys())}
        return {"languages": list(snippets.keys())}
    
    elif cmd_name == "info":
        if not args:
            return {"error": "Usage: info <file>"}
        filepath = Path(sess.cwd) / args[0]
        return get_file_info(str(filepath))
    
    elif cmd_name == "find":
        pattern = args[0] if args else "*"
        return search_files(pattern, sess.cwd)
    
    elif cmd_name == "ps":
        return get_process_list()
    
    elif cmd_name == "git":
        if args and args[0] == "status":
            return get_git_status(sess.cwd)
        # Pass through to shell for other git commands
        pass
    
    elif cmd_name == "neural":
        msg = " ".join(args)
        if not msg:
            return {"error": "Usage: neural <message>"}
        # This will be handled by the frontend chat API
        return {"redirect": "chat", "message": msg}
    
    # Default: pass to shell
    sess.write(cmd + "\n")
    output = sess.read(timeout=0.5)
    return {"output": output}


@terminal_bp.route("/api/terminal/<session_id>/resize", methods=["POST"])
def resize_session(session_id):
    if session_id not in sessions:
        return {"error": "Session not found"}, 404
    data = request.json or {}
    rows = int(data.get("rows", 24))
    cols = int(data.get("cols", 100))
    sessions[session_id].resize(rows, cols)
    return {"ok": True}


@terminal_bp.route("/api/terminal/<session_id>/alive", methods=["GET"])
def alive_session(session_id):
    if session_id not in sessions:
        return {"alive": False}
    return {"alive": sessions[session_id].is_alive()}


@terminal_bp.route("/api/terminal/<session_id>/buffer", methods=["GET"])
def buffer_session(session_id):
    if session_id not in sessions:
        return {"error": "Session not found"}, 404
    return {"buffer": sessions[session_id].get_buffer()}


@terminal_bp.route("/api/terminal/<session_id>/history", methods=["GET"])
def history_session(session_id):
    if session_id not in sessions:
        return {"error": "Session not found"}, 404
    sess = sessions[session_id]
    return {"history": sess.history[-50:]}  # Last 50 commands


@terminal_bp.route("/api/terminal/<session_id>/stop", methods=["POST"])
def stop_session(session_id):
    if session_id in sessions:
        sessions[session_id].stop()
        del sessions[session_id]
    if session_id in command_history:
        del command_history[session_id]
    return {"ok": True}


@terminal_bp.route("/api/terminal/sessions", methods=["GET"])
def list_sessions():
    cleanup_dead_sessions()
    return {
        "sessions": [
            {
                "session_id": sid, 
                "alive": sess.is_alive(), 
                "rows": sess.rows, 
                "cols": sess.cols,
                "cwd": sess.cwd
            }
            for sid, sess in sessions.items()
        ]
    }


@terminal_bp.route("/api/terminal/snippets", methods=["GET"])
def list_snippets():
    """List all available code snippets."""
    return {
        "snippets": {
            lang: list(names.keys()) 
            for lang, names in snippets.items()
        }
    }


@terminal_bp.route("/api/terminal/snippets/<language>/<name>", methods=["GET"])
def get_snippet(language, name):
    """Get a specific code snippet."""
    if language in snippets and name in snippets[language]:
        return {"code": snippets[language][name]}
    return {"error": "Snippet not found"}, 404
