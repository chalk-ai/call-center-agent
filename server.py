"""HTTP server for the Halyard Mobile call-center agent demo.

GET  /                        the call page (index.html)
GET  /api/scenarios           caller personas the page offers
POST /api/calls               start a call -> {"call_id"}
POST /api/calls/{id}/say      {"text"} -> NDJSON: "say" and "tool" events, then one "state"
POST /api/calls/{id}/simulate {"scenario"} -> NDJSON for a whole call played by a simulated caller
POST /api/calls/{id}/end      after-call work: summary, disposition, QA scorecard

The model is reached through the Chalk model gateway with a router API key from a Chalk secret.
The page is public, so calls, turns and text are capped per visitor.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import agent

PORT = int(os.environ.get("PORT", "8080"))
INDEX_HTML = Path(__file__).with_name("index.html")
API_SERVER = os.environ.get("CHALK_API_SERVER", "https://api.chalk.ai").rstrip("/")
ROUTER_BASE_URL = os.environ.get("ROUTER_BASE_URL", f"{API_SERVER}/v1/router")
LIVE_MODEL = os.environ.get("LIVE_MODEL", "anthropic/claude-haiku-4-5")
WRAPUP_MODEL = os.environ.get("WRAPUP_MODEL", "anthropic/claude-sonnet-5-5")

MAX_TEXT_CHARS = 400
MAX_TURNS = 24
MAX_SIM_TURNS = 12
CALL_TTL_S = 20 * 60
MAX_ACTIVE_CALLS = 200
CALLS_PER_HOUR = 12
SIMULATIONS_PER_HOUR = 4

log = logging.getLogger("call-center")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def model_config() -> agent.ModelConfig:
    client = anthropic.Anthropic(
        base_url=ROUTER_BASE_URL,
        api_key=os.environ["ROUTER_API_KEY"],
        default_headers={"X-Chalk-Env-Id": os.environ["CHALK_ENVIRONMENT_ID"]},
        timeout=60.0,
        max_retries=2,
    )
    return agent.ModelConfig(client, LIVE_MODEL, WRAPUP_MODEL)


class Limiter:
    """Sliding one-hour window of events per key."""

    def __init__(self, per_hour: int) -> None:
        super().__init__()
        self.per_hour = per_hour
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            q = self.events[key]
            while q and now - q[0] > 3600:
                q.popleft()
            if len(q) >= self.per_hour:
                return False
            q.append(now)
            return True


calls: dict[str, agent.Call] = {}
call_turns: dict[str, int] = defaultdict(int)
call_locks: dict[str, threading.Lock] = {}
call_limiter = Limiter(CALLS_PER_HOUR)
sim_limiter = Limiter(SIMULATIONS_PER_HOUR)
cfg = model_config()
app = FastAPI()


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def reap_expired() -> None:
    now = time.time()
    for cid in [cid for cid, c in calls.items() if now - c.started > CALL_TTL_S]:
        calls.pop(cid, None)
        call_turns.pop(cid, None)
        call_locks.pop(cid, None)


def get_call(call_id: str) -> agent.Call:
    call = calls.get(call_id)
    if call is None:
        raise HTTPException(404, "This call has ended or expired. Start a new one.")
    return call


def state_event(call: agent.Call) -> dict[str, Any]:
    """What the agent desktop shows: who is verified, their account, and what has changed."""
    account = agent._mask(call.crm[call.verified_account]) if call.verified_account else None
    return {
        "type": "state",
        "verified": call.verified_account is not None,
        "account_id": call.verified_account,
        "account": account,
        "credited": call.credited,
        "retention_used": call.retention_used,
        "escalated": call.escalated,
        "ended": call.ended,
        "turns": call_turns[call.id],
    }


def stream(events: Iterator[dict[str, Any]], lock: threading.Lock) -> StreamingResponse:
    """Runs a blocking event generator on a thread and streams each event as one JSON line.

    The per-call lock keeps two requests from interleaving turns on one conversation.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def run() -> None:
        try:
            with lock:
                for event in events:
                    asyncio.run_coroutine_threadsafe(queue.put(json.dumps(event) + "\n"), loop).result()
        except Exception as exc:  # noqa: BLE001 - reported to the page instead of a dropped stream
            log.exception("turn failed")
            error = {"type": "error", "message": f"The agent hit an error: {type(exc).__name__}"}
            asyncio.run_coroutine_threadsafe(queue.put(json.dumps(error) + "\n"), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    threading.Thread(target=run, daemon=True).start()

    async def body() -> AsyncIterator[str]:
        while (line := await queue.get()) is not None:
            yield line

    return StreamingResponse(body(), media_type="application/x-ndjson")


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/api/scenarios")
async def scenarios() -> JSONResponse:
    return JSONResponse({"company": agent.COMPANY, "agent": agent.AGENT_NAME, "scenarios": agent.SCENARIOS})


@app.post("/api/calls")
async def start_call(request: Request) -> JSONResponse:
    reap_expired()
    if len(calls) >= MAX_ACTIVE_CALLS:
        raise HTTPException(503, "All lines are busy. Try again in a few minutes.")
    if not call_limiter.allow(client_ip(request)):
        raise HTTPException(429, f"That is {CALLS_PER_HOUR} calls this hour. Try again later.")
    call = agent.Call(id=uuid.uuid4().hex)
    calls[call.id] = call
    call_locks[call.id] = threading.Lock()
    greeting = f"Thanks for calling {agent.COMPANY}, this is {agent.AGENT_NAME}. How can I help you today?"
    call.transcript.append({"speaker": "agent", "text": greeting})
    # The greeting is part of the conversation the model sees, so it does not greet twice.
    call.messages.append({"role": "user", "content": "(The caller is connected.)"})
    call.messages.append({"role": "assistant", "content": greeting})
    return JSONResponse({"call_id": call.id, "greeting": greeting})


class SayRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


@app.post("/api/calls/{call_id}/say")
async def say(call_id: str, req: SayRequest) -> StreamingResponse:
    call = get_call(call_id)
    if call.ended:
        raise HTTPException(409, "This call has ended.")
    if call_turns[call_id] >= MAX_TURNS:
        raise HTTPException(429, "This call has reached its length limit. End it to see the wrap-up.")
    call_turns[call_id] += 1

    def events() -> Iterator[dict[str, Any]]:
        yield from agent.take_turn(call, req.text.strip(), cfg)
        yield state_event(call)

    return stream(events(), call_locks[call_id])


class SimulateRequest(BaseModel):
    scenario: str


@app.post("/api/calls/{call_id}/simulate")
async def simulate(call_id: str, req: SimulateRequest, request: Request) -> StreamingResponse:
    call = get_call(call_id)
    scenario = next((s for s in agent.SCENARIOS if s["id"] == req.scenario), None)
    if scenario is None:
        raise HTTPException(404, "Unknown scenario.")
    if not sim_limiter.allow(client_ip(request)):
        raise HTTPException(429, f"That is {SIMULATIONS_PER_HOUR} simulated calls this hour. Try again later.")

    def events() -> Iterator[dict[str, Any]]:
        for _ in range(MAX_SIM_TURNS):
            line = agent.caller_turn(scenario, call, cfg)
            if "[HANGUP]" in line or not line:
                break
            yield {"type": "caller", "text": line}
            call_turns[call_id] += 1
            yield from agent.take_turn(call, line, cfg)
            yield state_event(call)
            if call.ended:
                break

    return stream(events(), call_locks[call_id])


@app.post("/api/calls/{call_id}/end")
async def end(call_id: str) -> JSONResponse:
    call = get_call(call_id)
    if len(call.transcript) < 2:
        calls.pop(call_id, None)
        return JSONResponse({"summary": "The caller hung up before saying anything.", "empty": True})
    record = await asyncio.to_thread(agent.wrap_up, call, cfg)
    calls.pop(call_id, None)
    return JSONResponse(record)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
