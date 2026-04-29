#!/usr/bin/env python3
"""
NeuralAI — Integrated Terminal Module
PTY-based shell sessions with WebSocket streaming.
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
from flask import Blueprint, request, Response

terminal_bp = Blueprint("terminal", __name__)

# In-memory session store
sessions = {}

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
        self.buffer = []  # scrollback buffer
        self.buffer_size = 2000

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
            cwd = os.environ.get("HOME", "/root")
            os.chdir(cwd)
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(sig, signal.SIG_DFL)
            os.execvp(os.environ.get("SHELL", "/bin/bash"), ["-bash", "--login"])
            os._exit(1)
        else:
            os.close(slave_fd)
            flags = fcntl.fcntl(self.master_fd, fcntl.F_GETFL)
            fcntl.fcntl(self.master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            self.started = True
            # Set initial size
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

def cleanup_dead_sessions():
    dead = [sid for sid, sess in sessions.items() if not sess.is_alive()]
    for sid in dead:
        sessions[sid].stop()
        del sessions[sid]

@terminal_bp.route("/api/terminal/create", methods=["POST"])
def create_session():
    cleanup_dead_sessions()
    data = request.json or {}
    rows = int(data.get("rows", 24))
    cols = int(data.get("cols", 80))
    session_id = str(uuid.uuid4())[:8]
    sess = TerminalSession(session_id, rows=rows, cols=cols)
    sess.start()
    sessions[session_id] = sess
    return {"session_id": session_id, "rows": rows, "cols": cols}

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
        "cols": sess.cols
    }

@terminal_bp.route("/api/terminal/<session_id>/write", methods=["POST"])
def write_session(session_id):
    if session_id not in sessions:
        return {"error": "Session not found"}, 404
    data = request.json or {}
    cmd = data.get("input", "")
    if cmd:
        sessions[session_id].write(cmd)
    return {"ok": True}

@terminal_bp.route("/api/terminal/<session_id>/resize", methods=["POST"])
def resize_session(session_id):
    if session_id not in sessions:
        return {"error": "Session not found"}, 404
    data = request.json or {}
    rows = int(data.get("rows", 24))
    cols = int(data.get("cols", 80))
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

@terminal_bp.route("/api/terminal/<session_id>/stop", methods=["POST"])
def stop_session(session_id):
    if session_id in sessions:
        sessions[session_id].stop()
        del sessions[session_id]
    return {"ok": True}

@terminal_bp.route("/api/terminal/sessions", methods=["GET"])
def list_sessions():
    cleanup_dead_sessions()
    return {
        "sessions": [
            {"session_id": sid, "alive": sess.is_alive(), "rows": sess.rows, "cols": sess.cols}
            for sid, sess in sessions.items()
        ]
    }
