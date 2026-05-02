# tools/git_assistant.py
#
# Git operations assistant
# - Status, log, diff
# - Stage, commit, push
# - Branch management
# - PR creation (via gh CLI)

import subprocess
import os
from typing import Dict, Any, List, Optional
from pathlib import Path


class GitAssistant:
    """Git operations from NeuralAI chat."""

    def __init__(self, repo_path: str = "/home/workspace"):
        self.repo_path = Path(repo_path)

    def _run_git(self, args: List[str], timeout: int = 30) -> Dict[str, Any]:
        """Execute a git command."""
        try:
            result = subprocess.run(
                ['git'] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.repo_path,
                env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout.strip(),
                "error": result.stderr.strip() if result.returncode != 0 else "",
                "exit_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Git command timed out after {timeout}s",
                "output": "",
                "exit_code": -1
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Git is not installed",
                "output": "",
                "exit_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "exit_code": -1
            }

    def _run_gh(self, args: List[str], timeout: int = 30) -> Dict[str, Any]:
        """Execute a GitHub CLI command."""
        try:
            result = subprocess.run(
                ['gh'] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.repo_path
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout.strip(),
                "error": result.stderr.strip() if result.returncode != 0 else "",
                "exit_code": result.returncode
            }
            
        except FileNotFoundError:
            return {
                "success": False,
                "error": "GitHub CLI (gh) is not installed. Install from: https://cli.github.com/",
                "output": "",
                "exit_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "exit_code": -1
            }

    def is_repo(self) -> Dict[str, Any]:
        """Check if current directory is a git repository."""
        result = self._run_git(['rev-parse', '--is-inside-work-tree'])
        
        return {
            "success": True,
            "is_repo": result["success"] and result["output"] == "true",
            "path": str(self.repo_path)
        }

    def status(self) -> Dict[str, Any]:
        """
        Get repository status.
        
        Returns:
            {
                "success": bool,
                "branch": str,
                "staged": list,
                "modified": list,
                "untracked": list,
                "ahead": int,
                "behind": int
            }
        """
        # Check if repo
        if not self.is_repo()["is_repo"]:
            return {
                "success": False,
                "error": "Not a git repository",
                "branch": "",
                "staged": [],
                "modified": [],
                "untracked": [],
                "ahead": 0,
                "behind": 0
            }
        
        # Get branch
        branch_result = self._run_git(['branch', '--show-current'])
        branch = branch_result["output"] or "HEAD"
        
        # Get status
        status_result = self._run_git(['status', '--porcelain'])
        
        staged = []
        modified = []
        untracked = []
        
        for line in status_result["output"].split('\n'):
            if not line:
                continue
            
            status_code = line[:2]
            filename = line[3:]
            
            if status_code[0] != ' ' and status_code[0] != '?':
                staged.append(filename)
            if status_code[1] == 'M':
                modified.append(filename)
            if status_code[0] == '?':
                untracked.append(filename)
        
        # Get ahead/behind
        ahead = 0
        behind = 0
        if branch != "HEAD":
            count_result = self._run_git(['rev-list', '--left-right', '--count', f'origin/{branch}...HEAD'])
            if count_result["success"] and count_result["output"]:
                parts = count_result["output"].split()
                if len(parts) == 2:
                    behind = int(parts[0])
                    ahead = int(parts[1])
        
        return {
            "success": True,
            "branch": branch,
            "staged": staged,
            "modified": modified,
            "untracked": untracked,
            "ahead": ahead,
            "behind": behind
        }

    def log(self, count: int = 10, oneline: bool = True) -> Dict[str, Any]:
        """
        Get commit history.
        
        Returns:
            {
                "success": bool,
                "commits": [{"hash": str, "author": str, "date": str, "message": str}]
            }
        """
        if not self.is_repo()["is_repo"]:
            return {
                "success": False,
                "error": "Not a git repository",
                "commits": []
            }
        
        if oneline:
            result = self._run_git(['log', f'-{count}', '--oneline', '--format=%h %an %ar %s'])
            
            commits = []
            for line in result["output"].split('\n'):
                if not line:
                    continue
                parts = line.split(' ', 3)
                if len(parts) >= 4:
                    commits.append({
                        "hash": parts[0],
                        "author": parts[1],
                        "date": parts[2],
                        "message": parts[3]
                    })
        else:
            result = self._run_git(['log', f'-{count}', '--format=%H%n%an%n%ar%n%s%n---'])
            
            commits = []
            blocks = result["output"].split('---\n')
            for block in blocks:
                if not block.strip():
                    continue
                lines = block.strip().split('\n')
                if len(lines) >= 4:
                    commits.append({
                        "hash": lines[0],
                        "author": lines[1],
                        "date": lines[2],
                        "message": lines[3]
                    })
        
        return {
            "success": True,
            "commits": commits,
            "count": len(commits)
        }

    def diff(self, staged: bool = False, file: Optional[str] = None) -> Dict[str, Any]:
        """Get diff of changes."""
        args = ['diff']
        if staged:
            args.append('--staged')
        if file:
            args.extend(['--', file])
        
        result = self._run_git(args)
        
        return {
            "success": True,
            "diff": result["output"],
            "has_changes": bool(result["output"])
        }

    def add(self, files: List[str] = None) -> Dict[str, Any]:
        """
        Stage files for commit.
        
        Args:
            files: List of files to stage (or ['.'] for all)
        """
        if not self.is_repo()["is_repo"]:
            return {"success": False, "error": "Not a git repository"}
        
        if files is None:
            files = ['.']
        
        result = self._run_git(['add'] + files)
        
        return {
            "success": result["success"],
            "files": files,
            "error": result.get("error", "")
        }

    def commit(self, message: str) -> Dict[str, Any]:
        """Create a commit with the given message."""
        if not self.is_repo()["is_repo"]:
            return {"success": False, "error": "Not a git repository"}
        
        result = self._run_git(['commit', '-m', message])
        
        if result["success"]:
            # Get the new commit hash
            hash_result = self._run_git(['rev-parse', 'HEAD'])
            return {
                "success": True,
                "message": message,
                "commit": hash_result["output"][:7]
            }
        
        return {
            "success": False,
            "error": result.get("error", "Commit failed")
        }

    def push(self, branch: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """Push to remote."""
        if not self.is_repo()["is_repo"]:
            return {"success": False, "error": "Not a git repository"}
        
        args = ['push']
        if force:
            args.append('--force')
        if branch:
            args.extend(['origin', branch])
        
        result = self._run_git(args, timeout=60)
        
        return {
            "success": result["success"],
            "output": result.get("output", ""),
            "error": result.get("error", "")
        }

    def pull(self, branch: Optional[str] = None) -> Dict[str, Any]:
        """Pull from remote."""
        if not self.is_repo()["is_repo"]:
            return {"success": False, "error": "Not a git repository"}
        
        args = ['pull']
        if branch:
            args.extend(['origin', branch])
        
        result = self._run_git(args, timeout=60)
        
        return {
            "success": result["success"],
            "output": result.get("output", ""),
            "error": result.get("error", "")
        }

    def branch(self, name: Optional[str] = None, create: bool = False, 
               delete: bool = False, list_all: bool = False) -> Dict[str, Any]:
        """
        Manage branches.
        
        Args:
            name: Branch name
            create: Create new branch
            delete: Delete branch
            list_all: List all branches
        """
        if not self.is_repo()["is_repo"]:
            return {"success": False, "error": "Not a git repository"}
        
        if list_all or (not name and not create and not delete):
            result = self._run_git(['branch', '-a'])
            branches = [b.strip().lstrip('* ') for b in result["output"].split('\n') if b.strip()]
            return {
                "success": True,
                "branches": branches,
                "current": self._run_git(['branch', '--show-current'])["output"]
            }
        
        if create and name:
            result = self._run_git(['checkout', '-b', name])
            return {
                "success": result["success"],
                "branch": name,
                "error": result.get("error", "")
            }
        
        if delete and name:
            result = self._run_git(['branch', '-D', name])
            return {
                "success": result["success"],
                "deleted": name,
                "error": result.get("error", "")
            }
        
        if name:
            result = self._run_git(['checkout', name])
            return {
                "success": result["success"],
                "branch": name,
                "error": result.get("error", "")
            }
        
        return {"success": False, "error": "Invalid branch operation"}

    def remote(self) -> Dict[str, Any]:
        """Get remote information."""
        result = self._run_git(['remote', '-v'])
        
        remotes = {}
        for line in result["output"].split('\n'):
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                url = parts[1]
                if name not in remotes:
                    remotes[name] = url
        
        return {
            "success": True,
            "remotes": remotes
        }

    def stash(self, message: Optional[str] = None) -> Dict[str, Any]:
        """Stash changes."""
        args = ['stash']
        if message:
            args.extend(['-m', message])
        
        result = self._run_git(args)
        
        return {
            "success": result["success"],
            "output": result.get("output", ""),
            "error": result.get("error", "")
        }

    def stash_pop(self) -> Dict[str, Any]:
        """Apply and remove the latest stash."""
        result = self._run_git(['stash', 'pop'])
        
        return {
            "success": result["success"],
            "output": result.get("output", ""),
            "error": result.get("error", "")
        }

    # GitHub CLI integration
    def pr_create(self, title: str, body: str = "", 
                  base: str = "main", draft: bool = False) -> Dict[str, Any]:
        """
        Create a pull request using GitHub CLI.
        
        Requires: gh CLI installed and authenticated
        """
        args = ['pr', 'create', '--title', title, '--base', base]
        if body:
            args.extend(['--body', body])
        if draft:
            args.append('--draft')
        
        result = self._run_gh(args)
        
        if result["success"]:
            return {
                "success": True,
                "url": result["output"],
                "title": title
            }
        
        return {
            "success": False,
            "error": result.get("error", "Failed to create PR")
        }

    def pr_list(self, state: str = "open") -> Dict[str, Any]:
        """List pull requests."""
        result = self._run_gh(['pr', 'list', '--state', state, '--json', 'number,title,url'])
        
        if result["success"]:
            import json
            try:
                prs = json.loads(result["output"])
                return {
                    "success": True,
                    "prs": prs
                }
            except:
                pass
        
        return {
            "success": False,
            "error": result.get("error", "Failed to list PRs"),
            "prs": []
        }

    def issue_create(self, title: str, body: str = "") -> Dict[str, Any]:
        """Create an issue using GitHub CLI."""
        args = ['issue', 'create', '--title', title]
        if body:
            args.extend(['--body', body])
        
        result = self._run_gh(args)
        
        if result["success"]:
            return {
                "success": True,
                "url": result["output"],
                "title": title
            }
        
        return {
            "success": False,
            "error": result.get("error", "Failed to create issue")
        }


if __name__ == "__main__":
    # Test the git assistant
    git = GitAssistant()
    
    print("Checking if repo...")
    print(git.is_repo())
    
    print("\nGit status:")
    print(git.status())
    
    print("\nGit log:")
    print(git.log(count=5))
    
    print("\nGit branches:")
    print(git.branch(list_all=True))
