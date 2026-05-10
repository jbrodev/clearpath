"""
ClearPath - Pre-Operative Anesthesia Clearance Agent
FastAPI application serving the A2A v0.3 external agent endpoint.

Endpoints:
  GET  /.well-known/agent-card.json  - A2A discovery
  POST /                             - A2A message/send JSON-RPC handler
  GET  /health                       - Health check
"""

import json
import os
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from clearpath.agent_card import get_agent_card
from clearpath.models.a2a import (
    JSONRPCRequest, JSONRPCResponse, JSONRPCError,
    A2ATask, A2ATaskStatus, A2AArtifact,
    FHIR_CONTEXT_EXTENSION_URI,
)
from clearpath.pipeline import run_clearance_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("ClearPath starting up...")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY not set. LLM reasoning will use fallback mode.")
    yield
    print("ClearPath shutting down.")


app = FastAPI(
    title="ClearPath",
    description="Pre-operative anesthesia clearance triage agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/.well-known/agent-card.json")
async def agent_card():
    return JSONResponse(content=get_agent_card())


@app.get("/health")
async def health():
    return {"status": "ok", "service": "clearpath", "version": "1.0.0"}


@app.post("/")
async def a2a_handler(request: Request):
    raw_body = await request.body()

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error: invalid JSON"}
            },
            status_code=400
        )

    req_id = body.get("id")
    method = body.get("method", "")

    # A2A v1 renamed methods to PascalCase. Accept both old (v0.3) and new (v1)
    # method names so the same handler works against either dialect of caller.
    _SEND_METHODS = {"SendMessage", "message/send", "tasks/send"}
    if method not in _SEND_METHODS:
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        })

    try:
        rpc_request = JSONRPCRequest(**body)
    except Exception as e:
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32600, "message": f"Invalid request: {str(e)}"}
        })

    params = rpc_request.params
    if not params:
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32600, "message": "Missing params"}
        })

    if isinstance(params, dict):
        from clearpath.models.a2a import A2AMessageSendParams
        try:
            params = A2AMessageSendParams(**params)
        except Exception as e:
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": f"Invalid params: {str(e)}"}
            })

    message = params.message
    user_query = message.get_text() or "Review this patient for pre-operative anesthesia clearance"
    fhir_context = message.get_fhir_context()
    task_id = str(uuid.uuid4())

    try:
        clearance_output = await run_clearance_pipeline(fhir_context, user_query)

        result_json = clearance_output.model_dump()
        result_md = clearance_output.to_markdown()

        # Propagate inbound FHIR context back on the response so any agent
        # downstream of ClearPath in a multi-agent chain inherits it.
        echo_metadata = None
        if message.metadata and FHIR_CONTEXT_EXTENSION_URI in message.metadata:
            echo_metadata = {FHIR_CONTEXT_EXTENSION_URI: message.metadata[FHIR_CONTEXT_EXTENSION_URI]}

        task = A2ATask(
            id=task_id,
            status=A2ATaskStatus(state="completed"),
            artifacts=[
                A2AArtifact(
                    name="clearance_assessment",
                    parts=[
                        {"type": "text", "text": result_md},
                        {"type": "data", "data": result_json}
                    ]
                )
            ],
            metadata=echo_metadata,
        )

        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": req_id,
            "result": task.model_dump()
        })

    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"Pipeline error: {error_detail}")
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32000,
                "message": "Internal processing error",
                "data": str(e)
            }
        }, status_code=500)
