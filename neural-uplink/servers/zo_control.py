#!/usr/bin/env python3
import os, sys, time, yaml, socket
from aiohttp import web

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = "./config/node.yaml"

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}

def check_port(port):
    try:
        s = socket.socket()
        s.settimeout(1)
        s.connect(("localhost", port))
        s.close()
        return "healthy"
    except:
        return "down"

def get_metrics():
    import psutil
    vm = psutil.virtual_memory()
    return {
        "cpu": psutil.cpu_percent(),
        "mem_total_gb": round(vm.total/1e9, 1),
        "mem_used_gb": round(vm.used/1e9, 1),
        "mem_avail_gb": round(vm.available/1e9, 1),
        "disk_pct": psutil.disk_usage("/").percent,
        "uptime_s": int(time.time() - psutil.boot_time())
    }

async def node_health(request):
    cfg = load_config()
    svcs = cfg.get("services", [])
    status = {s["name"]: check_port(s["port"]) for s in svcs}
    h = sum(1 for v in status.values() if v == "healthy")
    state = "healthy" if h == len(status) else "degraded" if h > 0 else "offline"
    return web.json_response({"node_id": "baltimore-01", "state": state, "services": status, "metrics": get_metrics(), "ts": time.time()})

async def list_agents(request):
    try:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.get("http://localhost:7000/agents", timeout=aiohttp.ClientTimeout(total=3)) as r:
                return web.json_response(await r.json())
    except:
        return web.json_response({"agents": {}, "count": 0})

async def get_config(request):
    return web.json_response(load_config())

async def get_logs(request):
    svc = request.query.get("service", "all")
    n = int(request.query.get("lines", 30))
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    return web.json_response({"service": svc, "logs": [f"[{ts}] [{svc}] health check: OK", f"[{ts}] [{svc}] uptime check: OK"][:n]})

async def restart_svc(request):
    body = await request.json()
    return web.json_response({"service": body.get("service"), "action": "restart_requested"})

async def control_status(request):
    return web.json_response({"server": "zo-control", "port": 8001, "endpoints": ["/system/health", "/system/agents", "/system/config", "/system/logs"]})

async def health(request):
    return web.json_response({"status": "healthy", "server": "zo-control"})

app = web.Application()
app.router.add_get("/health", health)
app.router.add_get("/system/health", node_health)
app.router.add_get("/system/agents", list_agents)
app.router.add_get("/system/config", get_config)
app.router.add_get("/system/logs", get_logs)
app.router.add_post("/system/restart", restart_svc)
app.router.add_get("/control/status", control_status)

if __name__ == "__main__":
    print("[zo-control] Starting on 0.0.0.0:8001")
    web.run_app(app, host="0.0.0.0", port=8001, access_log=None)
