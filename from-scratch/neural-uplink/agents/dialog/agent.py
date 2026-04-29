#!/usr/bin/env python3
"""Dialog Agent — LLM chat / reasoning"""
import asyncio, aiohttp, uuid
from aiohttp import web

PORT = 7101
HANDLERS = {}

async def handle_task(request):
    body = await request.json()
    task_id = body.get("task_id", str(uuid.uuid4()))
    goal = body.get("goal", "")
    context = body.get("context", {})
    conversation = context.get("conversation", [])
    conversation.append({"role": "user", "content": goal})
    result = f"[DialogAgent] Processed: {goal[:80]}"
    return web.json_response({"task_id": task_id, "status": "completed", "result": result})

async def health(request):
    return web.json_response({"status": "healthy", "agent": "dialog"})

app = web.Application()
app.router.add_post("/task", handle_task)
app.router.add_get("/health", health)
web.run_app(app, host="0.0.0.0", port=PORT, access_log=None)
