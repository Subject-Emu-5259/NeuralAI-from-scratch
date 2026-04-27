#!/usr/bin/env python3
"""
Dialog Agent — User-facing LLM agent
Handles chat, explanations, code generation, and general conversation.
"""
import os, json, uuid, asyncio
from aiohttp import web

AGENT_ID = os.environ.get("AGENT_ID", "agent-dialog")
AGENT_PORT = int(os.environ.get("AGENT_PORT", 7101))

TASKS = {}

async def handle_task(request):
    body = await request.json()
    task_id = body.get("task_id", str(uuid.uuid4()))
    goal = body.get("goal", "")
    context = body.get("context", {})
    callback_url = body.get("callback_url", "")

    log_entry = {
        "task_id": task_id,
        "goal": goal,
        "context": context,
        "status": "processing"
    }
    TASKS[task_id] = log_entry

    # Route to actual LLM via app.py backend
    result = {"response": f"[DialogAgent] Processed: {goal[:100]}", "agent": AGENT_ID}

    # Callback if provided
    if callback_url:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as sess:
                await sess.post(callback_url, json={"task_id": task_id, "status": "completed", "result": result})
        except:
            pass

    TASKS[task_id]["status"] = "completed"
    return web.json_response({"task_id": task_id, "status": "completed", "result": result})

async def health(request):
    return web.json_response({"status": "healthy", "agent": AGENT_ID, "port": AGENT_PORT})

async def status_task(request):
    task_id = request.match_info.get("task_id")
    task = TASKS.get(task_id, {})
    return web.json_response({"task_id": task_id, "status": task.get("status", "unknown")})

async def cancel_task(request):
    task_id = request.match_info.get("task_id")
    if task_id in TASKS:
        TASKS[task_id]["status"] = "cancelled"
    return web.json_response({"task_id": task_id, "cancelled": True})

async def agent_info(request):
    return web.json_response({
        "agent_id": AGENT_ID,
        "name": "Dialog Agent",
        "type": "llm",
        "port": AGENT_PORT,
        "capabilities": ["chat", "code-generation", "explanations", "writing", "analysis"]
    })

app = web.Application()
app.router.add_post("/task", handle_task)
app.router.add_get("/health", health)
app.router.add_get("/status/{task_id}", status_task)
app.router.add_post("/cancel/{task_id}", cancel_task)
app.router.add_get("/info", agent_info)

if __name__ == "__main__":
    print(f"[{AGENT_ID}] Starting on port {AGENT_PORT}")
    web.run_app(app, host="0.0.0.0", port=AGENT_PORT, access_log=None)