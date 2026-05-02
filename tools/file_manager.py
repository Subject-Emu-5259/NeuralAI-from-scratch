# tools/file_manager.py
#
# Advanced file operations
# - Search by content or filename
# - Read/write/create/delete
# - Batch operations
# - Directory listing

import os
import re
import shutil
import fnmatch
from typing import Dict, Any, List, Optional
from pathlib import Path


class FileManager:
    """Manage files and directories with advanced operations."""

    def __init__(self, base_dir: str = "/home/workspace"):
        self.base_dir = Path(base_dir)

    def list_dir(self, path: str = ".", show_hidden: bool = False) -> Dict[str, Any]:
        """
        List contents of a directory.
        
        Returns:
            {
                "success": bool,
                "path": str,
                "files": [{"name": str, "type": str, "size": int, "modified": str}],
                "directories": [{"name": str, "modified": str}],
                "total_files": int,
                "total_dirs": int
            }
        """
        full_path = self.base_dir / path
        
        if not full_path.exists():
            return {
                "success": False,
                "error": f"Path does not exist: {path}",
                "path": str(full_path),
                "files": [],
                "directories": [],
                "total_files": 0,
                "total_dirs": 0
            }
        
        if not full_path.is_dir():
            return {
                "success": False,
                "error": f"Path is not a directory: {path}",
                "path": str(full_path),
                "files": [],
                "directories": [],
                "total_files": 0,
                "total_dirs": 0
            }
        
        files = []
        directories = []
        
        try:
            for item in full_path.iterdir():
                # Skip hidden files unless requested
                if not show_hidden and item.name.startswith('.'):
                    continue
                
                modified = os.path.getmtime(item)
                modified_str = str(modified)
                
                if item.is_file():
                    size = os.path.getsize(item)
                    files.append({
                        "name": item.name,
                        "type": item.suffix.lower() or "unknown",
                        "size": size,
                        "modified": modified_str
                    })
                elif item.is_dir():
                    directories.append({
                        "name": item.name,
                        "modified": modified_str
                    })
            
            # Sort by name
            files.sort(key=lambda x: x["name"])
            directories.sort(key=lambda x: x["name"])
            
            return {
                "success": True,
                "path": str(full_path),
                "files": files,
                "directories": directories,
                "total_files": len(files),
                "total_dirs": len(directories)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": str(full_path),
                "files": [],
                "directories": [],
                "total_files": 0,
                "total_dirs": 0
            }

    def read_file(self, path: str, max_size: int = 100000) -> Dict[str, Any]:
        """
        Read contents of a file.
        
        Returns:
            {
                "success": bool,
                "path": str,
                "content": str,
                "size": int,
                "lines": int
            }
        """
        full_path = self.base_dir / path
        
        if not full_path.exists():
            return {
                "success": False,
                "error": f"File does not exist: {path}",
                "path": str(full_path),
                "content": "",
                "size": 0,
                "lines": 0
            }
        
        if not full_path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {path}",
                "path": str(full_path),
                "content": "",
                "size": 0,
                "lines": 0
            }
        
        try:
            size = os.path.getsize(full_path)
            
            # Check if file is too large
            if size > max_size:
                return {
                    "success": False,
                    "error": f"File too large: {size} bytes (max: {max_size})",
                    "path": str(full_path),
                    "content": "",
                    "size": size,
                    "lines": 0
                }
            
            # Try to read as text
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                lines = content.count('\n') + 1
                
                return {
                    "success": True,
                    "path": str(full_path),
                    "content": content,
                    "size": size,
                    "lines": lines
                }
            except UnicodeDecodeError:
                # Binary file
                return {
                    "success": False,
                    "error": "File is binary, cannot read as text",
                    "path": str(full_path),
                    "content": "",
                    "size": size,
                    "lines": 0,
                    "is_binary": True
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": str(full_path),
                "content": "",
                "size": 0,
                "lines": 0
            }

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """
        Write content to a file (creates or overwrites).
        
        Returns:
            {
                "success": bool,
                "path": str,
                "size": int
            }
        """
        full_path = self.base_dir / path
        
        try:
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            size = os.path.getsize(full_path)
            
            return {
                "success": True,
                "path": str(full_path),
                "size": size
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": str(full_path),
                "size": 0
            }

    def delete(self, path: str) -> Dict[str, Any]:
        """
        Delete a file or directory.
        
        Returns:
            {
                "success": bool,
                "path": str,
                "type": str ("file" or "directory")
            }
        """
        full_path = self.base_dir / path
        
        if not full_path.exists():
            return {
                "success": False,
                "error": f"Path does not exist: {path}",
                "path": str(full_path),
                "type": None
            }
        
        try:
            if full_path.is_file():
                os.unlink(full_path)
                return {
                    "success": True,
                    "path": str(full_path),
                    "type": "file"
                }
            elif full_path.is_dir():
                shutil.rmtree(full_path)
                return {
                    "success": True,
                    "path": str(full_path),
                    "type": "directory"
                }
            else:
                return {
                    "success": False,
                    "error": "Unknown path type",
                    "path": str(full_path),
                    "type": None
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": str(full_path),
                "type": None
            }

    def search(self, pattern: str, path: str = ".", search_content: bool = False, 
               max_results: int = 50) -> Dict[str, Any]:
        """
        Search for files by name or content.
        
        Returns:
            {
                "success": bool,
                "query": str,
                "results": [{"path": str, "line": int, "match": str}],
                "total": int
            }
        """
        full_path = self.base_dir / path
        results = []
        
        if not full_path.exists():
            return {
                "success": False,
                "error": f"Path does not exist: {path}",
                "query": pattern,
                "results": [],
                "total": 0
            }
        
        try:
            if search_content:
                # Search within file contents
                for root, dirs, files in os.walk(full_path):
                    # Skip hidden directories and common ignore patterns
                    dirs[:] = [d for d in dirs if not d.startswith('.') and 
                              d not in ['node_modules', '__pycache__', 'venv', '.git']]
                    
                    for filename in files:
                        if filename.startswith('.'):
                            continue
                        
                        filepath = Path(root) / filename
                        
                        # Skip binary files
                        if filepath.suffix.lower() in ['.pyc', '.so', '.dll', '.exe', '.bin']:
                            continue
                        
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                for line_num, line in enumerate(f, 1):
                                    if pattern.lower() in line.lower():
                                        rel_path = filepath.relative_to(self.base_dir)
                                        results.append({
                                            "path": str(rel_path),
                                            "line": line_num,
                                            "match": line.strip()[:100]
                                        })
                                        
                                        if len(results) >= max_results:
                                            break
                        except:
                            pass
                        
                        if len(results) >= max_results:
                            break
                    if len(results) >= max_results:
                        break
            else:
                # Search by filename
                for root, dirs, files in os.walk(full_path):
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    
                    for filename in files:
                        if fnmatch.fnmatch(filename.lower(), f"*{pattern.lower()}*"):
                            filepath = Path(root) / filename
                            rel_path = filepath.relative_to(self.base_dir)
                            results.append({
                                "path": str(rel_path),
                                "line": 0,
                                "match": filename
                            })
                            
                            if len(results) >= max_results:
                                break
                    if len(results) >= max_results:
                        break
            
            return {
                "success": True,
                "query": pattern,
                "results": results,
                "total": len(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": pattern,
                "results": [],
                "total": 0
            }

    def copy(self, src: str, dst: str) -> Dict[str, Any]:
        """Copy a file or directory."""
        src_path = self.base_dir / src
        dst_path = self.base_dir / dst
        
        if not src_path.exists():
            return {
                "success": False,
                "error": f"Source does not exist: {src}",
                "src": str(src_path),
                "dst": str(dst_path)
            }
        
        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            if src_path.is_file():
                shutil.copy2(src_path, dst_path)
            else:
                shutil.copytree(src_path, dst_path)
            
            return {
                "success": True,
                "src": str(src_path),
                "dst": str(dst_path)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "src": str(src_path),
                "dst": str(dst_path)
            }

    def move(self, src: str, dst: str) -> Dict[str, Any]:
        """Move/rename a file or directory."""
        src_path = self.base_dir / src
        dst_path = self.base_dir / dst
        
        if not src_path.exists():
            return {
                "success": False,
                "error": f"Source does not exist: {src}",
                "src": str(src_path),
                "dst": str(dst_path)
            }
        
        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
            
            return {
                "success": True,
                "src": str(src_path),
                "dst": str(dst_path)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "src": str(src_path),
                "dst": str(dst_path)
            }

    def create_dir(self, path: str) -> Dict[str, Any]:
        """Create a directory."""
        full_path = self.base_dir / path
        
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            return {
                "success": True,
                "path": str(full_path)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": str(full_path)
            }


if __name__ == "__main__":
    # Test the file manager
    fm = FileManager()
    
    print("Listing workspace:")
    result = fm.list_dir()
    print(f"Files: {result['total_files']}, Dirs: {result['total_dirs']}")
    
    print("\nSearching for Python files:")
    result = fm.search(".py", search_content=False)
    print(f"Found {result['total']} files")
