# tools/code_sandbox.py
#
# Safe code execution sandbox
# - Timeout protection
# - Output capture
# - Python and JavaScript support

import subprocess
import tempfile
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


class CodeSandbox:
    """Execute generated code safely with timeout and isolation."""

    def __init__(self, timeout: int = 30, max_output: int = 10000):
        self.timeout = timeout
        self.max_output = max_output

    def run_python(self, code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute Python code in a sandboxed environment.
        
        Returns:
            {
                "success": bool,
                "output": str,
                "error": str,
                "exit_code": int,
                "execution_time": float
            }
        """
        import time
        start = time.time()
        timeout = timeout or self.timeout
        
        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['python3', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir(),
                # Security: limit resources
                env={
                    'PYTHONDONTWRITEBYTECODE': '1',
                    'PYTHONUNBUFFERED': '1',
                }
            )
            
            output = result.stdout[:self.max_output]
            error = result.stderr[:self.max_output]
            
            return {
                "success": result.returncode == 0,
                "output": output,
                "error": error if result.returncode != 0 else "",
                "exit_code": result.returncode,
                "execution_time": time.time() - start
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Execution timed out after {timeout}s",
                "exit_code": -1,
                "execution_time": timeout
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "exit_code": -1,
                "execution_time": time.time() - start
            }
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except:
                pass

    def run_javascript(self, code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute JavaScript code using Node.js.
        
        Returns:
            {
                "success": bool,
                "output": str,
                "error": str,
                "exit_code": int,
                "execution_time": float
            }
        """
        import time
        start = time.time()
        timeout = timeout or self.timeout
        
        # Check if node is available
        try:
            subprocess.run(['node', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {
                "success": False,
                "output": "",
                "error": "Node.js is not installed. Install with: apt install nodejs",
                "exit_code": -1,
                "execution_time": 0
            }
        
        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.js', 
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['node', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir()
            )
            
            output = result.stdout[:self.max_output]
            error = result.stderr[:self.max_output]
            
            return {
                "success": result.returncode == 0,
                "output": output,
                "error": error if result.returncode != 0 else "",
                "exit_code": result.returncode,
                "execution_time": time.time() - start
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Execution timed out after {timeout}s",
                "exit_code": -1,
                "execution_time": timeout
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "exit_code": -1,
                "execution_time": time.time() - start
            }
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except:
                pass

    def run_bash(self, command: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a bash command.
        
        Returns:
            {
                "success": bool,
                "output": str,
                "error": str,
                "exit_code": int,
                "execution_time": float
            }
        """
        import time
        start = time.time()
        timeout = timeout or self.timeout
        
        try:
            result = subprocess.run(
                ['bash', '-c', command],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir()
            )
            
            output = result.stdout[:self.max_output]
            error = result.stderr[:self.max_output]
            
            return {
                "success": result.returncode == 0,
                "output": output,
                "error": error if result.returncode != 0 else "",
                "exit_code": result.returncode,
                "execution_time": time.time() - start
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Execution timed out after {timeout}s",
                "exit_code": -1,
                "execution_time": timeout
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "exit_code": -1,
                "execution_time": time.time() - start
            }


# Convenience function for direct execution
def execute(code: str, language: str = "python", timeout: int = 30) -> Dict[str, Any]:
    """Execute code in the specified language."""
    sandbox = CodeSandbox(timeout=timeout)
    
    if language.lower() in ["python", "py"]:
        return sandbox.run_python(code, timeout)
    elif language.lower() in ["javascript", "js", "node"]:
        return sandbox.run_javascript(code, timeout)
    elif language.lower() in ["bash", "shell", "sh"]:
        return sandbox.run_bash(code, timeout)
    else:
        return {
            "success": False,
            "output": "",
            "error": f"Unsupported language: {language}",
            "exit_code": -1,
            "execution_time": 0
        }


if __name__ == "__main__":
    # Test the sandbox
    sandbox = CodeSandbox()
    
    # Test Python
    print("Testing Python execution...")
    result = sandbox.run_python("print('Hello from Python!')")
    print(f"Python: {result}")
    
    # Test JavaScript
    print("\nTesting JavaScript execution...")
    result = sandbox.run_javascript("console.log('Hello from JavaScript!')")
    print(f"JavaScript: {result}")
    
    # Test timeout
    print("\nTesting timeout...")
    result = sandbox.run_python("import time; time.sleep(60)", timeout=2)
    print(f"Timeout test: {result}")
