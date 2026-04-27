#!/usr/bin/env python3
"""
zo-watchdog — Self-Healing Supervisor
Monitors all processes, restarts on crash, maintains node health.
"""
import os, sys, time, yaml, subprocess, signal
from pathlib import Path

CONFIG_PATH = "./config/node.yaml"
SERVICES = []
RESTART_COUNTS = {}
LAST_START = {}

def load_config():
    global SERVICES
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f)
            SERVICES = data.get("services", [])
            return data
    return {}

def get_process_cmd(service_name: str) -> str:
    for svc in SERVICES:
        if svc["name"] == service_name:
            return svc["cmd"]
    return ""

def is_running(port: int) -> bool:
    import socket
    try:
        sock = socket.socket()
        sock.settimeout(1)
        sock.connect(("localhost", port))
        sock.close()
        return True
    except:
        return False

def start_service(service_name: str) -> bool:
    cmd = get_process_cmd(service_name)
    if not cmd:
        return False
    workdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        proc = subprocess.Popen(
            cmd.split(),
            cwd=workdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
        LAST_START[service_name] = time.time()
        print(f"[watchdog] Started {service_name} (PID: {proc.pid})")
        return True
    except Exception as e:
        print(f"[watchdog] Failed to start {service_name}: {e}")
        return False

def stop_service(service_name: str):
    for svc in SERVICES:
        if svc["name"] == service_name:
            try:
                import socket
                resp = requests.post(f"http://localhost:{svc['port']}/shutdown")
            except:
                pass
    # Kill by name
    subprocess.run(["pkill", "-f", service_name], stderr=subprocess.DEVNULL)

def restart_service(service_name: str):
    stop_service(service_name)
    time.sleep(2)
    start_service(service_name)

def check_services(cfg):
    global SERVICES
    cfg = load_config()
    SERVICES = cfg.get("services", [])
    wd_cfg = cfg.get("watchdog", {})

    max_attempts = wd_cfg.get("max_restart_attempts", 3)
    cooldown = wd_cfg.get("restart_cooldown_secs", 30)

    print(f"[watchdog] Checking {len(SERVICES)} services...")
    for svc in SERVICES:
        name = svc["name"]
        port = svc["port"]
        policy = svc.get("restart_policy", "always")
        healthy = is_running(port)

        if not healthy:
            restarts = RESTART_COUNTS.get(name, 0)
            if restarts >= max_attempts:
                print(f"[watchdog] {name}: MAX RESTARTS REACHED — marking DEGRADED")
                RESTART_COUNTS[name] = 0
            elif policy in ("always", "on_failure"):
                # Check cooldown
                last = LAST_START.get(name, 0)
                if time.time() - last > cooldown:
                    print(f"[watchdog] {name}: DOWN — restarting...")
                    RESTART_COUNTS[name] = restarts + 1
                    start_service(name)
                else:
                    print(f"[watchdog] {name}: DOWN — cooldown active, skipping restart")
            else:
                print(f"[watchdog] {name}: DOWN — no restart policy")
        else:
            if name in RESTART_COUNTS:
                RESTART_COUNTS[name] = 0

def main():
    print("[watchdog] Starting self-healing daemon...")
    load_config()
    interval = 15

    while True:
        try:
            check_services(None)
        except Exception as e:
            print(f"[watchdog] Error in check loop: {e}")
        time.sleep(interval)

if __name__ == "__main__":
    main()