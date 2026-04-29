#!/usr/bin/env python3
"""
Data Agent — Storage, embeddings, indexing
Handles vector operations, document indexing, and data retrieval.
"""
import os, json, uuid, asyncio
from aiohttp import web
import aiohttp

AGENT_ID = os.environ.get("AGENT_ID", "agent-data")
AGENT_PORT = int(os.environ.get("AGENT_PORT", 7102))
VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH", "./vector_db")

TASKS = {}

async def handle_task(request):
    body = await request.json()
    task_id = body.get("task_id", str(uuid.uuid4()))
    goal = body.get("goal", "")
    context = body.get("context", {})

    result = {"response": f"[DataAgent] Processed data task: {goal[:80]}", "agent": AGENT_ID}
    TASKS[task_id] = {"status": "completed", "result": result}

    return web.json_response({"task_id": task_id, "status": "completed", "result": result})

async def query_vector(request):
    body = await request.json()
    query = body.get("query", "")
    top_k = body.get("top_k", 4)
    return web.json_response({
        "query": query,
        "results": [{"content": f"[Data] result for: {query[:40]}", "score": 0.95}],
        "agent": AGENT_ID
    })

async def index_document(request):
    body = await request.json()
    doc_id = body.get("doc_id", str(uuid.uuid4()))
    content = body.get("content", "")
    return web.json_response({
        "doc_id": doc_id,
        "indexed": True,
        "chunks": len(content) // 500,
        "agent": AGENT_ID
    })

async def health(request):
    return web.json_response({"status": "healthy", "agent": AGENT_ID, "port": AGENT_PORT})

async def agent_info(request):
    return web.json_response({
        "agent_id": AGENT_ID,
        "name": "Data Agent",
        "type": "data",
        "port": AGENT_PORT,
        "capabilities": ["vector-search", "document-indexing", "embeddings", "data-retrieval"]
    })

app = web.Application()
app.router.add_post("/task", handle_task)
app.router.add_post("/query", query_vector)
app.router.add_post("/index", index_document)
app.router.add_get("/health", health)
app.router.add_get("/info", agent_info)

if __name__ == "__main__":
    print(f"[{AGENT_ID}] Starting on port {AGENT_PORT}")
    web.run_app(app, host="0.0.0.0", port=AGENT_PORT, access_log=None)