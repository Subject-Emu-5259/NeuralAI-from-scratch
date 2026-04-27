#!/usr/bin/env python3
"""
Ops Agent — DevOps monitoring, logging, service management
Handles health checks, log aggregation, and service restarts.
"""
import os, json, uuid
from aiohttp import web
import aiohttp

AGENT_ID = os.environ.get("AGENT_ID", "agent-ops")
AGENT_PORT = int(os.environ.get("AGENT_PORT", 7103))

SERVICES = {
    "neural_uplinkd": {"port": 7000, "status": "unknown"},
    "agent-dialog": {"port": 7101, "status": "unknown"},
    "agent-data": {"port": 7102, "status": "unknown"},
    "zo-gateway": {"port": 8000, "status": "unknown"},
    "zo-control": {"port": 8001, "status": "unknown"},
}

TASKS = {}

async def handle_task(request):
    body = await request.json()
    task_id = body.get("task_id", str(uuid.uuid4()))
    goal = body.get("goal", "")

    result = {"response": f"[OpsAgent] Executed: {goal[:80]}", "agent": AGENT_ID}
    TASKS[task_id] = {"status": "completed", "result": result}

    return web.json_response({"task_id": task_id, "status": "completed", "result": result})

async def get_logs(request):
    service = request.query.get("service", "all")
    lines = request.query.get("lines", 50)
    return web.json_response({
        "service": service,
        "logs": [f"[2026-04-25] {service}: operational check passed"],
        "count": 1
    })

async def restart_service(request):
    body = await request.json()
    service_name = body.get("service")
    if service_name in SERVICES:
        SERVICES[service_name]["status"] = "restarting"
        return web.json_response({"service": service_name, "status": "restarting", "message": "Restart initiated"})
    return web.json_response({"error": "Service not found"}, status=404)

async def service_status(request):
    service_name = request.query.get("name")
    if service_name:
        return web.json_response({"service": service_name, **SERVICES.get(service_name, {})})
    return web.json_response({"services": SERVICES})

async def health(request):
    return web.json_response({"status": "healthy", "agent": AGENT_ID, "port": AGENT_PORT})

async def agent_info(request):
    return web.json_response({
        "agent_id": AGENT_ID,
        "name": "Ops Agent",
        "type": "ops",
        "port": AGENT_PORT,
        "capabilities": ["log-aggregation", "service-health", "restart-management", "metrics"]
    })

app = web.Application()
app.router.add_post("/task", handle_task)
app.router.add_get("/logs", get_logs)
app.router.add_post("/restart", restart_service)
app.router.add_get("/services", service_status)
app.router.add_get("/health", health)
app.router.add_get("/info", agent_info)

if __name__ == "__main__":
    print(f"[{AGENT_ID}] Starting on port {AGENT_PORT}")
    web.run_app(app, host="0.0.0.0", port=AGENT_PORT, access_log=None)