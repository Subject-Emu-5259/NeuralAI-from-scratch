# tools/tool_handler.py
#
# Central handler for all tool execution
# - Routes tool calls to appropriate tool class
# - Returns formatted results for chat display

import sys
import os
from typing import Dict, Any, Optional

# Add tools to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.code_sandbox import CodeSandbox, execute
from tools.file_manager import FileManager
from tools.web_fetcher import WebFetcher
from tools.db_connector import DatabaseConnector
from tools.git_assistant import GitAssistant


class ToolHandler:
    """Central handler for tool execution."""

    def __init__(self, workspace: str = "/home/workspace"):
        self.workspace = workspace
        self.code_sandbox = CodeSandbox()
        self.file_manager = FileManager(base_dir=workspace)
        self.web_fetcher = WebFetcher()
        self.db_connector = DatabaseConnector()
        self.git_assistant = GitAssistant(repo_path=workspace)

    def execute(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool: Tool name (terminal, code_exec, file_manager, web_fetcher, database, git)
            params: Tool-specific parameters
        
        Returns:
            {
                "success": bool,
                "output": str,
                "error": str,
                "data": dict (tool-specific)
            }
        """
        handlers = {
            "terminal": self._handle_terminal,
            "code_exec": self._handle_code_exec,
            "file_manager": self._handle_file_manager,
            "web_fetcher": self._handle_web_fetcher,
            "database": self._handle_database,
            "git": self._handle_git,
        }
        
        handler = handlers.get(tool)
        if not handler:
            return {
                "success": False,
                "output": "",
                "error": f"Unknown tool: {tool}",
                "data": {}
            }
        
        try:
            return handler(params)
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Tool execution error: {str(e)}",
                "data": {}
            }

    def _handle_terminal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle terminal/shell commands."""
        command = params.get("command", "")
        if not command:
            return {
                "success": False,
                "output": "",
                "error": "No command provided",
                "data": {}
            }
        
        result = self.code_sandbox.run_bash(command)
        
        output = result["output"]
        if result["error"]:
            output += f"\n[stderr]\n{result['error']}"
        
        return {
            "success": result["success"],
            "output": output,
            "error": "",
            "data": {
                "exit_code": result["exit_code"],
                "execution_time": result["execution_time"]
            }
        }

    def _handle_code_exec(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code execution requests."""
        code = params.get("code", "")
        language = params.get("language", "python")
        
        if not code:
            return {
                "success": False,
                "output": "",
                "error": "No code provided",
                "data": {}
            }
        
        # If code doesn't look like actual code, it might be a message
        # asking to run something mentioned elsewhere
        if not any(kw in code for kw in ["def ", "function ", "print(", "console.log", "import "]):
            return {
                "success": False,
                "output": "",
                "error": "No executable code detected. Provide code to run.",
                "data": {}
            }
        
        result = execute(code, language=language)
        
        output = result["output"]
        if result["error"]:
            output += f"\n[error]\n{result['error']}"
        
        return {
            "success": result["success"],
            "output": output,
            "error": "",
            "data": {
                "exit_code": result["exit_code"],
                "execution_time": result["execution_time"],
                "language": language
            }
        }

    def _handle_file_manager(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file operations."""
        query = params.get("query", "").lower()
        
        # Determine operation from query
        if "list" in query or "show files" in query or "what files" in query:
            path = params.get("path", ".")
            result = self.file_manager.list_dir(path)
            
            if result["success"]:
                output = f"Directory: {result['path']}\n\n"
                output += f"Directories ({result['total_dirs']}):\n"
                for d in result["directories"]:
                    output += f"  📁 {d['name']}\n"
                output += f"\nFiles ({result['total_files']}):\n"
                for f in result["files"]:
                    size = self._format_size(f["size"])
                    output += f"  📄 {f['name']} ({size})\n"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
            return {
                "success": False,
                "output": "",
                "error": result.get("error", "Failed to list directory"),
                "data": result
            }
        
        elif "read" in query:
            # Extract path from query
            path = params.get("path", "")
            if not path:
                # Try to extract from query
                import re
                match = re.search(r"read (?:file )?['\"]?([^\s'\"]+)['\"]?", query)
                if match:
                    path = match.group(1)
            
            result = self.file_manager.read_file(path)
            
            if result["success"]:
                output = f"File: {result['path']}\n"
                output += f"Size: {result['size']} bytes | Lines: {result['lines']}\n\n"
                output += f"```\n{result['content']}\n```"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
            return {
                "success": False,
                "output": "",
                "error": result.get("error", "Failed to read file"),
                "data": result
            }
        
        elif "search" in query or "find" in query:
            # Extract search pattern
            pattern = params.get("pattern", "")
            if not pattern:
                # Try to extract from query
                import re
                match = re.search(r"(?:search|find) (?:for )?['\"]?([^\s'\"]+)['\"]?", query)
                if match:
                    pattern = match.group(1)
                else:
                    pattern = query.split()[-1]  # Use last word as fallback
            
            search_content = "content" in query or "in files" in query
            result = self.file_manager.search(pattern, search_content=search_content)
            
            if result["success"]:
                output = f"Search for '{pattern}': {result['total']} results\n\n"
                for r in result["results"][:20]:
                    if r.get("line"):
                        output += f"📄 {r['path']}:{r['line']}\n  {r['match']}\n\n"
                    else:
                        output += f"📄 {r['path']}\n"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
            return {
                "success": False,
                "output": "",
                "error": result.get("error", "Search failed"),
                "data": result
            }
        
        # Default: list current directory
        result = self.file_manager.list_dir()
        
        if result["success"]:
            output = f"Directory: {result['path']}\n\n"
            output += f"Directories ({result['total_dirs']}):\n"
            for d in result["directories"][:10]:
                output += f"  📁 {d['name']}\n"
            output += f"\nFiles ({result['total_files']}):\n"
            for f in result["files"][:10]:
                size = self._format_size(f["size"])
                output += f"  📄 {f['name']} ({size})\n"
            
            return {
                "success": True,
                "output": output,
                "error": "",
                "data": result
            }
        return {
            "success": False,
            "output": "",
            "error": result.get("error", "Failed to list directory"),
            "data": result
        }

    def _handle_web_fetcher(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle web fetching operations."""
        url = params.get("url", "")
        
        if not url:
            return {
                "success": False,
                "output": "",
                "error": "No URL provided",
                "data": {}
            }
        
        # Fetch and parse the URL
        result = self.web_fetcher.fetch(url)
        
        if result["success"]:
            output = f"URL: {result['url']}\n"
            output += f"Title: {result['title']}\n"
            output += f"Status: {result['status']}\n\n"
            output += f"Content Preview:\n{result['text'][:1500]}...\n\n"
            output += f"Links: {len(result['links'])} found\n"
            output += f"Images: {len(result['images'])} found"
            
            return {
                "success": True,
                "output": output,
                "error": "",
                "data": result
            }
        return {
            "success": False,
            "output": "",
            "error": result.get("error", "Failed to fetch URL"),
            "data": result
        }

    def _handle_database(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle database operations."""
        query = params.get("query", "").lower()
        sql = params.get("sql", "")
        
        # Default to NeuralAI's database if not connected
        if not self.db_connector.active_db:
            db_path = os.path.join(self.workspace, "Projects/NeuralAI/from-scratch/web_ui/neuralai.db")
            if os.path.exists(db_path):
                self.db_connector.connect_sqlite(db_path, "neuralai")
            else:
                # Create in-memory DB for testing
                self.db_connector.connect_sqlite(":memory:", "memory")
        
        if "show tables" in query or "list tables" in query:
            result = self.db_connector.tables()
            
            if result["success"]:
                output = f"Tables ({result['count']}):\n"
                for t in result["tables"]:
                    output += f"  📊 {t}\n"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
            return {
                "success": False,
                "output": "",
                "error": result.get("error", "Failed to list tables"),
                "data": result
            }
        
        elif "schema" in query:
            result = self.db_connector.schema()
            
            if result["success"]:
                output = "Database Schema:\n\n"
                for table in result["tables"]:
                    output += f"📊 {table['name']}:\n"
                    for col in table["columns"]:
                        pk = " 🔑" if col["primary_key"] else ""
                        output += f"  - {col['name']}: {col['type']}{pk}\n"
                    output += "\n"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
            return {
                "success": False,
                "output": "",
                "error": result.get("error", "Failed to get schema"),
                "data": result
            }
        
        elif sql:
            result = self.db_connector.query(sql)
            
            if result["success"]:
                output = f"Query: {sql}\n\n"
                if result["rows"]:
                    output += f"Results ({result['row_count']} rows):\n"
                    # Format as table
                    if result["columns"]:
                        output += "| " + " | ".join(result["columns"]) + " |\n"
                        output += "|" + "|".join(["---" for _ in result["columns"]]) + "|\n"
                    for row in result["rows"][:20]:
                        values = [str(v) for v in row.values()]
                        output += "| " + " | ".join(values) + " |\n"
                else:
                    output += f"Affected {result['row_count']} rows"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
            return {
                "success": False,
                "output": "",
                "error": result.get("error", "Query failed"),
                "data": result
            }
        
        # Default: show tables
        result = self.db_connector.tables()
        output = f"Connected to: {self.db_connector.active_db}\n\n"
        output += f"Tables ({result.get('count', 0)}):\n"
        for t in result.get("tables", []):
            output += f"  📊 {t}\n"
        
        return {
            "success": True,
            "output": output,
            "error": "",
            "data": result
        }

    def _handle_git(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle git operations."""
        action = params.get("action", "status").lower()
        
        # Check if in a git repo
        if not self.git_assistant.is_repo()["is_repo"]:
            return {
                "success": False,
                "output": "",
                "error": "Not a git repository",
                "data": {}
            }
        
        if "status" in action:
            result = self.git_assistant.status()
            
            if result["success"]:
                output = f"Branch: {result['branch']}\n"
                output += f"Ahead: {result['ahead']} | Behind: {result['behind']}\n\n"
                
                if result["staged"]:
                    output += "Staged:\n"
                    for f in result["staged"]:
                        output += f"  ✅ {f}\n"
                
                if result["modified"]:
                    output += "Modified:\n"
                    for f in result["modified"]:
                        output += f"  📝 {f}\n"
                
                if result["untracked"]:
                    output += "Untracked:\n"
                    for f in result["untracked"]:
                        output += f"  ❓ {f}\n"
                
                if not any([result["staged"], result["modified"], result["untracked"]]):
                    output += "Working directory clean ✨"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
        
        elif "log" in action:
            result = self.git_assistant.log(count=10)
            
            if result["success"]:
                output = f"Recent commits ({result['count']}):\n\n"
                for c in result["commits"]:
                    output += f"📝 {c['hash']} - {c['message']}\n"
                    output += f"   {c['author']} • {c['date']}\n\n"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
        
        elif "branch" in action:
            result = self.git_assistant.branch(list_all=True)
            
            if result["success"]:
                output = f"Current: {result['current']}\n\n"
                output += "Branches:\n"
                for b in result["branches"]:
                    marker = "→ " if b == result["current"] else "  "
                    output += f"{marker}{b}\n"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
        
        elif "diff" in action:
            result = self.git_assistant.diff()
            
            if result["success"]:
                output = "Git Diff:\n\n"
                output += f"```diff\n{result['diff']}\n```"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
        
        elif "remote" in action:
            result = self.git_assistant.remote()
            
            if result["success"]:
                output = "Remotes:\n"
                for name, url in result["remotes"].items():
                    output += f"  {name}: {url}\n"
                
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "data": result
                }
        
        # Default: show status
        result = self.git_assistant.status()
        
        if result["success"]:
            output = f"Branch: {result['branch']}\n"
            output += f"Ahead: {result['ahead']} | Behind: {result['behind']}\n\n"
            
            if result["staged"]:
                output += "Staged:\n"
                for f in result["staged"]:
                    output += f"  ✅ {f}\n"
            
            if result["modified"]:
                output += "Modified:\n"
                for f in result["modified"]:
                    output += f"  📝 {f}\n"
            
            if result["untracked"]:
                output += "Untracked:\n"
                for f in result["untracked"]:
                    output += f"  ❓ {f}\n"
            
            if not any([result["staged"], result["modified"], result["untracked"]]):
                output += "Working directory clean ✨"
            
            return {
                "success": True,
                "output": output,
                "error": "",
                "data": result
            }
        
        return {
            "success": False,
            "output": "",
            "error": "Git operation failed",
            "data": {}
        }

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


# Global handler instance
_handler: Optional[ToolHandler] = None


def get_handler(workspace: str = "/home/workspace") -> ToolHandler:
    """Get or create the global tool handler."""
    global _handler
    if _handler is None:
        _handler = ToolHandler(workspace=workspace)
    return _handler


def run_tool(tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool and return the result."""
    handler = get_handler()
    return handler.execute(tool, params)


if __name__ == "__main__":
    # Test the tool handler
    handler = ToolHandler()
    
    print("Testing file manager:")
    result = handler.execute("file_manager", {"query": "list files"})
    print(result["output"])
    
    print("\nTesting git:")
    result = handler.execute("git", {"action": "status"})
    print(result["output"])
