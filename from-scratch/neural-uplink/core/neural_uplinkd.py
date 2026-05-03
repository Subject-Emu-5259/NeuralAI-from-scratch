#!/usr/bin/env python3
"""
Neural Uplink Core — Orchestrator Daemon
Central brain for multi-agent AI OS. Routes tasks to specialist agents.
"""
import os, json, uuid, asyncio, logging
from datetime import datetime
from aiohttp import web, ClientSession
import aiohttp

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger('neural_uplinkd')

AGENT_REGISTRY = {}
TASKS = {}

class Task:
    def __init__(self, task_id: str, goal: str, context: dict = None, constraints: dict = None, callback_url: str = ""):
        self.task_id = task_id
        self.goal = goal
        self.context = context or {}
        self.constraints = constraints or {}
        self.callback_url = callback_url
        self.status = "pending"
        self.result = None
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow().isoformat()
        self.subtasks = []
        self.agent_id = None

    def to_dict(self):
        return {
            "task_id": self.task_id, "goal": self.goal, "context": self.context,
            "constraints": self.constraints, "callback_url": self.callback_url,
            "status": self.status, "result": self.result,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "subtasks": self.subtasks, "agent_id": self.agent_id
        }

def register_agent(agent_id: str, port: int, name: str):
    url = f"http://localhost:{port}"
    AGENT_REGISTRY[agent_id] = {"name": name, "port": port, "url": url, "last_seen": datetime.utcnow().isoformat()}
    log.info(f"Registered agent: {name} ({agent_id}) at {url}")

def get_agents_by_capability(capability: str):
    cap_map = {"llm": "agent-dialog", "data": "agent-data", "ops": "agent-ops", "world": "agent-worldbuilder"}
    target = cap_map.get(capability, "agent-dialog")
    return [aid for aid, info in AGENT_REGISTRY.items() if info["name"] == target]

def route_task_to_agent(task: Task):
    goal = task.goal.lower()
    if any(w in goal for w in ["world", "city", "simulate", "build", "render", "environment"]):
        agents = get_agents_by_capability("world")
    elif any(w in goal for w in ["data", "search", "index", "embed", "query", "database"]):
        agents = get_agents_by_capability("data")
    elif any(w in goal for w in ["ops", "restart", "health", "monitor", "log", "deploy"]):
        agents = get_agents_by_capability("ops")
    else:
        agents = get_agents_by_capability("llm")
    return agents[0] if agents else None

async def forward_to_agent(agent_id: str, task: Task) -> dict:
    info = AGENT_REGISTRY.get(agent_id)
    if not info:
        return {"error": f"Agent {agent_id} not found"}
    url = f"{info['url']}/task"
    payload = {
        "task_id": task.task_id, "goal": task.goal, "context": task.context,
        "constraints": task.constraints, "callback_url": f"http://localhost:7000/callback/{task.task_id}"
    }
    try:
        async with ClientSession() as sess:
            async with sess.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    task.status = data.get("status", "forwarded")
                    task.agent_id = agent_id
                    return data
                return {"error": f"Agent returned {resp.status}"}
    except Exception as e:
        return {"error": str(e)}

async def callback_handler(request):
    data = await request.json()
    tid = data.get("task_id")
    if tid in TASKS:
        TASKS[tid].status = data.get("status", "completed")
        TASKS[tid].result = data.get("result", {})
        TASKS[tid].updated_at = datetime.utcnow().isoformat()
        log.info(f"Callback: {tid} = {data.get('status')}")
    return web.json_response({"received": True})

async def submit_task(request):
    body = await request.json()
    task_id = body.get("task_id") or str(uuid.uuid4())
    task = Task(task_id=task_id, goal=body.get("goal", ""), context=body.get("context", {}),
                constraints=body.get("constraints", {}), callback_url=body.get("callback_url", ""))
    TASKS[task_id] = task
    log.info(f"Task: {task_id} — {task.goal[:60]}")
    agent_id = route_task_to_agent(task)
    if agent_id:
        result = await forward_to_agent(agent_id, task)
        return web.json_response({"task_id": task_id, "status": "forwarded", "agent": agent_id, "result": result})
    task.status = "no_agent_available"
    return web.json_response({"task_id": task_id, "status": "no_agent_available", "message": "No agent capable"})

async def sync_task(request):
    """Sync endpoint: accepts {task, context} → returns {result} directly."""
    body = await request.json()
    task_text = body.get("task", body.get("goal", ""))
    context = body.get("context", {})
    task_id = str(uuid.uuid4())
    task_obj = Task(task_id=task_id, goal=task_text, context=context)
    TASKS[task_id] = task_obj
    agent_id = route_task_to_agent(task_obj)
    if not agent_id:
        task_obj.status = "no_agent_available"
        return web.json_response({"error": "No agent capable", "task_id": task_id}, status=503)
    info = AGENT_REGISTRY.get(agent_id)
    url = f"{info['url']}/task"
    payload = {
        "task_id": task_id, "goal": task_text, "context": context,
        "constraints": {}, "callback_url": f"http://localhost:7000/callback/{task_id}"
    }
    try:
        async with ClientSession() as sess:
            async with sess.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    task_obj.status = data.get("status", "forwarded")
                    task_obj.agent_id = agent_id
                    result = data.get("result", data.get("response", ""))
                    return web.json_response({"result": result, "task_id": task_id, "agent": agent_id})
                return web.json_response({"error": f"Agent returned {resp.status}"}, status=502)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def get_task(request):
    task_id = request.match_info.get("task_id")
    task = TASKS.get(task_id)
    if not task:
        return web.json_response({"error": "Task not found"}, status=404)
    return web.json_response(task.to_dict())

async def list_tasks(request):
    status_filter = request.query.get("status")
    tasks = list(TASKS.values())
    if status_filter:
        tasks = [t for t in tasks if t.status == status_filter]
    return web.json_response({"tasks": [t.to_dict() for t in tasks], "count": len(tasks)})

async def agent_register(request):
    try:
        body = await request.json()
    except Exception as e:
        print(f"Error parsing registration JSON: {e}")
        return web.json_response({"error": "Invalid JSON"}, status=400)
        
    agent_id = body.get("agent_id") or str(uuid.uuid4())
    port = body.get("port", 7100)
    name = body.get("name", "unknown")
    register_agent(agent_id, port, name)
    return web.json_response({"agent_id": agent_id, "registered": True})

async def list_agents(request):
    return web.json_response({"agents": AGENT_REGISTRY, "count": len(AGENT_REGISTRY)})

async def health(request):
    return web.json_response({
        "status": "healthy", "node": "baltimore-01",
        "services": len(AGENT_REGISTRY),
        "active_tasks": sum(1 for t in TASKS.values() if t.status == "processing")
    })

async def shutdown_node(request):
    log.warning("Shutdown requested")
    return web.json_response({"ok": True})

async def node_info(request):
    return web.json_response({
        "node_id": "baltimore-01", "name": "Baltimore Node",
        "location": "Baltimore, MD", "termux": True, "uptime": "N/A",
        "capabilities": ["llm-inference", "world-simulation", "data-processing", "api-gateway", "agent-runtime"],
        "registered_agents": list(AGENT_REGISTRY.keys()),
        "active_tasks": len([t for t in TASKS.values() if t.status == "processing"]),
        "version": "1.0.0"
    })

app = web.Application()

app.router.add_post("/task", submit_task)
app.router.add_get("/task/{task_id}", get_task)
app.router.add_get("/tasks", list_tasks)
app.router.add_post("/callback/{task_id}", callback_handler)
app.router.add_post("/agent/register", agent_register)
app.router.add_get("/agents", list_agents)
app.router.add_get("/health", health)
app.router.add_get("/info", node_info)
app.router.add_post("/shutdown", shutdown_node)
app.router.add_post("/api/v1/zo/tasks", sync_task)  # sync wrapper

if __name__ == "__main__":
    log.info("[NeuralUplink] Starting on localhost:7000")
    web.run_app(app, host="0.0.0.0", port=7000, access_log=None)
