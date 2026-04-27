#!/usr/bin/env python3
import os, sys, time, hashlib
from aiohttp import web

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

API_KEYS = {
    "zo-system": "sk_zo_dev_0001",
    "sphere-prod": "sk_sphere_prod_0002",
    "omnix-prod": "sk_omnix_prod_0003",
}
RATE_LIMIT = {"window": 60, "max_requests": 100}
request_log = {}

def verify_api_key(request):
    key = request.headers.get("X-API-Key", "")
    for tenant, stored_key in API_KEYS.items():
        if key == stored_key:
            return tenant
    return None

def check_rate_limit(client_id):
    now = time.time()
    if client_id not in request_log:
        request_log[client_id] = []
    request_log[client_id] = [t for t in request_log[client_id] if now - t < RATE_LIMIT["window"]]
    if len(request_log[client_id]) >= RATE_LIMIT["max_requests"]:
        return False
    request_log[client_id].append(now)
    return True

async def submit_task(request):
    tenant = verify_api_key(request)
    if not tenant:
        return web.json_response({"error": "Unauthorized"}, status=401)
    if not check_rate_limit(request.headers.get("X-Client-ID", "anonymous")):
        return web.json_response({"error": "Rate limit exceeded"}, status=429)
    body = await request.json()
    task_id = body.get("task_id") or hashlib.md5(f"{time.time()}".encode()).hexdigest()[:12]
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:7000/task", json={**body, "task_id": task_id}, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                return web.json_response({"task_id": task_id, "tenant": tenant, "status": "submitted", "result": data})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=502)

async def get_task(request):
    task_id = request.match_info.get("task_id")
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://localhost:7000/task/{task_id}", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return web.json_response(await resp.json())
    except:
        return web.json_response({"error": "Task not found"}, status=404)

async def gateway_status(request):
    return web.json_response({"gateway": "zo-gateway", "status": "operational", "version": "1.0.0", "tenants": list(API_KEYS.keys())})

async def health(request):
    return web.json_response({"status": "healthy", "server": "zo-gateway"})

app = web.Application()
app.router.add_get("/health", health)
app.router.add_get("/gateway/status", gateway_status)
app.router.add_post("/api/v1/tasks", submit_task)
app.router.add_get("/api/v1/tasks/{task_id}", get_task)
app.router.add_post("/api/v1/zo/tasks", submit_task)
app.router.add_post("/api/v1/sphere/tasks", submit_task)
app.router.add_post("/api/v1/omnix/tasks", submit_task)

if __name__ == "__main__":
    print("[zo-gateway] Starting on 0.0.0.0:8000")
    web.run_app(app, host="0.0.0.0", port=8000, access_log=None)
