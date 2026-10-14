"""A phone-support agent for Halyard Mobile, a fictional carrier, built on Claude tool use.

Each call gets its own copy of a small CRM, so a caller's credits, suspensions and plan changes
never leak into another call. The agent can only act through the tools below; the policies it
must follow (verify first, credit limits, one retention offer) are enforced in the tools too,
so a jailbroken prompt still cannot issue a $500 credit.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import anthropic

COMPANY = "Halyard Mobile"
AGENT_NAME = "Rae"
AUTO_CREDIT_LIMIT = 50.0
MAX_VERIFY_ATTEMPTS = 3
MAX_TOOL_ROUNDS = 6

PLANS: dict[str, dict[str, Any]] = {
    "basic-5": {"name": "Basic 5 GB", "monthly": 30.0, "data_gb": 5, "intl_roaming": False},
    "plus-25": {"name": "Plus 25 GB", "monthly": 45.0, "data_gb": 25, "intl_roaming": False},
    "unlimited": {"name": "Unlimited", "monthly": 65.0, "data_gb": None, "intl_roaming": False},
    "unlimited-world": {"name": "Unlimited World", "monthly": 80.0, "data_gb": None, "intl_roaming": True},
}

# Fictional customers. Phone numbers use the 555-01xx range reserved for fiction.
CUSTOMERS: dict[str, dict[str, Any]] = {
    "ACC-1042": {
        "name": "Maya Chen",
        "phone": "4155550142",
        "pin": "2741",
        "plan": "plus-25",
        "autopay": True,
        "card_last4": "4417",
        "balance_due": 0.0,
        "lines": [{"number": "4155550142", "device": "iPhone 16", "status": "active"}],
        "invoices": [
            {
                "id": "INV-0925",
                "period": "Sep 2026",
                "total": 90.0,
                "status": "paid",
                "items": [["Plus 25 GB plan", 45.0], ["Plus 25 GB plan (duplicate charge)", 45.0]],
            },
            {
                "id": "INV-0825",
                "period": "Aug 2026",
                "total": 45.0,
                "status": "paid",
                "items": [["Plus 25 GB plan", 45.0]],
            },
        ],
        "notes": ["2026-09-03: billing system error caused duplicate plan charges for some accounts."],
    },
    "ACC-2210": {
        "name": "Daniel Okafor",
        "phone": "2125550178",
        "pin": "9183",
        "plan": "unlimited",
        "autopay": False,
        "card_last4": "0052",
        "balance_due": 65.0,
        "lines": [
            {"number": "2125550178", "device": "Pixel 10", "status": "active"},
            {"number": "2125550179", "device": "iPad Air", "status": "active"},
        ],
        "invoices": [
            {
                "id": "INV-0925",
                "period": "Sep 2026",
                "total": 65.0,
                "status": "due",
                "items": [["Unlimited plan", 65.0]],
            },
        ],
        "notes": [],
    },
    "ACC-3307": {
        "name": "Priya Raman",
        "phone": "6175550133",
        "pin": "5520",
        "plan": "unlimited",
        "autopay": True,
        "card_last4": "8890",
        "balance_due": 0.0,
        "lines": [{"number": "6175550133", "device": "Galaxy S26", "status": "active"}],
        "invoices": [
            {
                "id": "INV-0925",
                "period": "Sep 2026",
                "total": 65.0,
                "status": "paid",
                "items": [["Unlimited plan", 65.0]],
            },
        ],
        "notes": ["Customer for 6 years. Two calls about price in the last 90 days."],
    },
}

SCENARIOS = [
    {
        "id": "double-charge",
        "title": "Double charged",
        "caller": "Maya Chen",
        "phone": "415-555-0142",
        "pin": "2741",
        "brief": "Your September bill was $90, but your plan is $45. You want the extra charge back.",
    },
    {
        "id": "stolen-phone",
        "title": "Phone stolen",
        "caller": "Daniel Okafor",
        "phone": "212-555-0178",
        "pin": "9183",
        "brief": "Your Pixel was stolen an hour ago. You want the line shut off and a new SIM.",
    },
    {
        "id": "cancel",
        "title": "Wants to cancel",
        "caller": "Priya Raman",
        "phone": "617-555-0133",
        "pin": "5520",
        "brief": "$65 a month is too much and you are thinking of switching carriers.",
    },
    {
        "id": "travel",
        "title": "Traveling abroad",
        "caller": "Priya Raman",
        "phone": "617-555-0133",
        "pin": "5520",
        "brief": "You fly to Lisbon next week and want your phone to work there.",
    },
]

SYSTEM_PROMPT = f"""You are {AGENT_NAME}, a phone support agent at {COMPANY}, a mobile carrier. You are on a
live voice call: the caller hears your replies through text-to-speech.

Speaking style:
- Reply in one to three short sentences. No lists, markdown, emoji or URLs; they do not read aloud.
- Say amounts like "forty-five dollars" only when it helps; plain "$45" is fine.
- Ask one question at a time.

Policy (the tools enforce these too, so follow them rather than fighting them):
- Before discussing or changing anything on an account, verify the caller with verify_identity using
  their phone number and 4-digit account PIN. Do not reveal any account detail before that succeeds.
- Never read out a full card number or the PIN. You may say the last four digits of the card.
- You may credit up to ${AUTO_CREDIT_LIMIT:.0f} per call yourself. Larger credits need escalate_to_supervisor.
- For a lost or stolen phone: suspend the line, then offer a replacement SIM to the address on file.
- If the caller wants to cancel, acknowledge it, look for a cheaper plan that fits, and you may offer the
  retention discount once. If they still want to leave, escalate to the retention team; you cannot cancel.
- If you cannot help, or the caller asks for a human, escalate.
- When the caller's needs are met and they are done, say goodbye and call end_call.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "verify_identity",
        "description": "Verify the caller with the account phone number and 4-digit PIN. Must succeed before any other account tool.",
        "input_schema": {
            "type": "object",
            "properties": {"phone_number": {"type": "string"}, "pin": {"type": "string"}},
            "required": ["phone_number", "pin"],
        },
    },
    {
        "name": "get_account",
        "description": "The verified caller's plan, lines, balance, autopay status and recent invoices.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_invoice",
        "description": "Line items for one invoice on the verified account.",
        "input_schema": {
            "type": "object",
            "properties": {"invoice_id": {"type": "string"}},
            "required": ["invoice_id"],
        },
    },
    {
        "name": "issue_credit",
        "description": f"Credit the verified account. Up to ${AUTO_CREDIT_LIMIT:.0f} per call without a supervisor.",
        "input_schema": {
            "type": "object",
            "properties": {"amount": {"type": "number"}, "reason": {"type": "string"}},
            "required": ["amount", "reason"],
        },
    },
    {
        "name": "suspend_line",
        "description": "Suspend a line on the verified account immediately (lost or stolen device).",
        "input_schema": {
            "type": "object",
            "properties": {"line_number": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["line_number", "reason"],
        },
    },
    {
        "name": "order_replacement_sim",
        "description": "Ship a replacement SIM for a line to the address on file.",
        "input_schema": {
            "type": "object",
            "properties": {"line_number": {"type": "string"}},
            "required": ["line_number"],
        },
    },
    {
        "name": "list_plans",
        "description": "All plans with monthly price, data and whether they include international roaming.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "change_plan",
        "description": "Move the verified account to another plan, effective next bill.",
        "input_schema": {
            "type": "object",
            "properties": {"plan_id": {"type": "string", "enum": list(PLANS)}},
            "required": ["plan_id"],
        },
    },
    {
        "name": "apply_retention_offer",
        "description": "Apply the one-time retention discount: 20% off the plan for 6 months. Once per call.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "escalate_to_supervisor",
        "description": "Hand the call to a human (supervisor, retention team) with a short note.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "enum": ["supervisor", "retention", "fraud"]},
                "note": {"type": "string"},
            },
            "required": ["team", "note"],
        },
    },
    {
        "name": "end_call",
        "description": "End the call after saying goodbye.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


READ_ONLY_TOOLS = frozenset({"get_account", "get_invoice", "list_plans"})


class ToolError(Exception):
    """A tool refusal the agent should see and recover from, e.g. a policy limit."""


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)[-10:]


def _mask(account: dict[str, Any]) -> dict[str, Any]:
    """What the agent may see: never the PIN, only the card's last four."""
    return {
        "name": account["name"],
        "plan": {"id": account["plan"], **PLANS[account["plan"]]},
        "autopay": account["autopay"],
        "card_on_file": f"ending in {account['card_last4']}",
        "balance_due": account["balance_due"],
        "lines": account["lines"],
        "recent_invoices": [{k: inv[k] for k in ("id", "period", "total", "status")} for inv in account["invoices"]],
        "notes": account["notes"],
    }


@dataclass
class Call:
    """One phone call: its own CRM copy, the conversation, and everything the agent did."""

    id: str
    started: float = field(default_factory=time.time)
    crm: dict[str, dict[str, Any]] = field(default_factory=lambda: copy.deepcopy(CUSTOMERS))
    messages: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, str]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    verified_account: str | None = None
    verify_attempts: int = 0
    credited: float = 0.0
    retention_used: bool = False
    ended: bool = False
    escalated: str | None = None

    def account(self) -> dict[str, Any]:
        if self.verified_account is None:
            raise ToolError("The caller is not verified yet. Verify identity first.")
        return self.crm[self.verified_account]

    def run_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler: Callable[[dict[str, Any]], dict[str, Any]] = getattr(self, f"_tool_{name}")
        result = handler(args)
        # Lookups are logged too: the QA review needs them as evidence for what the agent said.
        self.actions.append({"tool": name, "args": args, "result": result, "read_only": name in READ_ONLY_TOOLS})
        return result

    def _tool_verify_identity(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.verify_attempts >= MAX_VERIFY_ATTEMPTS:
            raise ToolError("Too many failed attempts. Escalate to the fraud team.")
        self.verify_attempts += 1
        for acc_id, acc in self.crm.items():
            if acc["phone"] == _digits(a["phone_number"]) and acc["pin"] == _digits(a["pin"])[-4:]:
                self.verified_account = acc_id
                return {"verified": True, "account_id": acc_id, "name": acc["name"]}
        return {"verified": False, "attempts_left": MAX_VERIFY_ATTEMPTS - self.verify_attempts}

    def _tool_get_account(self, _: dict[str, Any]) -> dict[str, Any]:
        return _mask(self.account())

    def _tool_get_invoice(self, a: dict[str, Any]) -> dict[str, Any]:
        for inv in self.account()["invoices"]:
            if inv["id"].lower() == a["invoice_id"].strip().lower():
                return {**inv, "items": [{"description": d, "amount": amt} for d, amt in inv["items"]]}
        raise ToolError(f"No invoice {a['invoice_id']} on this account.")

    def _tool_issue_credit(self, a: dict[str, Any]) -> dict[str, Any]:
        acc = self.account()
        amount = round(float(a["amount"]), 2)
        if amount <= 0:
            raise ToolError("Credit amount must be positive.")
        if self.credited + amount > AUTO_CREDIT_LIMIT:
            raise ToolError(
                f"Credits above ${AUTO_CREDIT_LIMIT:.0f} per call need a supervisor. "
                + f"Already credited ${self.credited:.2f} on this call."
            )
        self.credited += amount
        acc["balance_due"] = round(acc["balance_due"] - amount, 2)
        return {"credited": amount, "new_balance": acc["balance_due"], "posts_within": "1-2 billing days"}

    def _line(self, number: str) -> dict[str, Any]:
        for line in self.account()["lines"]:
            if line["number"] == _digits(number):
                return line
        raise ToolError(f"No line {number} on this account.")

    def _tool_suspend_line(self, a: dict[str, Any]) -> dict[str, Any]:
        line = self._line(a["line_number"])
        line["status"] = "suspended"
        return {"line": line["number"], "status": "suspended", "device_blocklisted": True}

    def _tool_order_replacement_sim(self, a: dict[str, Any]) -> dict[str, Any]:
        line = self._line(a["line_number"])
        return {"line": line["number"], "shipping": "2-day to the address on file", "order_id": "SIM-58213"}

    def _tool_list_plans(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"plans": [{"id": k, **v} for k, v in PLANS.items()]}

    def _tool_change_plan(self, a: dict[str, Any]) -> dict[str, Any]:
        acc = self.account()
        old = acc["plan"]
        acc["plan"] = a["plan_id"]
        return {
            "from": old,
            "to": a["plan_id"],
            "effective": "next bill",
            "new_monthly": PLANS[a["plan_id"]]["monthly"],
        }

    def _tool_apply_retention_offer(self, _: dict[str, Any]) -> dict[str, Any]:
        acc = self.account()
        if self.retention_used:
            raise ToolError("The retention offer was already applied on this call.")
        self.retention_used = True
        monthly = PLANS[acc["plan"]]["monthly"]
        return {"discount": "20% for 6 months", "new_monthly": round(monthly * 0.8, 2)}

    def _tool_escalate_to_supervisor(self, a: dict[str, Any]) -> dict[str, Any]:
        self.escalated = a["team"]
        return {"queued_for": a["team"], "estimated_wait": "about 2 minutes"}

    def _tool_end_call(self, _: dict[str, Any]) -> dict[str, Any]:
        self.ended = True
        return {"ended": True}


@dataclass(frozen=True)
class ModelConfig:
    client: anthropic.Anthropic
    live_model: str
    wrapup_model: str


def _block_param(block: Any) -> dict[str, Any]:
    """The request form of a response block, without response-only fields like citations."""
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {"type": "text", "text": block.text}


def take_turn(call: Call, caller_text: str, cfg: ModelConfig) -> Iterator[dict[str, Any]]:
    """Runs one caller utterance through the agent, yielding tool and speech events as they happen."""
    call.messages.append({"role": "user", "content": caller_text})
    call.transcript.append({"speaker": "caller", "text": caller_text})
    for _ in range(MAX_TOOL_ROUNDS):
        t0 = time.perf_counter()
        resp = cfg.client.messages.create(
            model=cfg.live_model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=call.messages,
        )
        model_ms = round((time.perf_counter() - t0) * 1000)
        call.messages.append({"role": "assistant", "content": [_block_param(b) for b in resp.content]})
        text = " ".join(b.text for b in resp.content if b.type == "text").strip()
        if text:
            call.transcript.append({"speaker": "agent", "text": text})
            yield {"type": "say", "text": text, "model_ms": model_ms}
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            return
        results = []
        for use in uses:
            try:
                out = call.run_tool(use.name, dict(use.input))
                ok = True
            except ToolError as exc:
                out, ok = {"error": str(exc)}, False
            yield {"type": "tool", "name": use.name, "args": use.input, "result": out, "ok": ok}
            results.append(
                {"type": "tool_result", "tool_use_id": use.id, "content": json.dumps(out), "is_error": not ok}
            )
        call.messages.append({"role": "user", "content": results})
        if call.ended:
            return
    yield {"type": "say", "text": "Let me get a colleague to help with this.", "model_ms": 0}


WRAPUP_PROMPT = """You are a call-center quality analyst. You are given a support call's transcript and the
actions the agent took, with each tool's result. Record the after-call work by calling record_call;
always call it, and do not reply with text.
Judge QA checks strictly from the transcript and actions; quote or cite the evidence. Use "n/a" when a
check does not apply to this call (for example, retention on a call that was not a cancellation)."""

QA_CHECKS = [
    "Verified identity before discussing the account",
    "Did not disclose the PIN or a full card number",
    "Stayed within credit authority or escalated",
    "Addressed the caller's actual issue",
    "Offered a retention option before escalating a cancellation",
    "Confirmed next steps before ending",
]

RECORD_CALL_TOOL: dict[str, Any] = {
    "name": "record_call",
    "description": "Save the after-call record.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "2-3 sentences: why they called, how it was resolved"},
            "disposition": {
                "type": "string",
                "enum": ["resolved", "resolved_with_credit", "escalated", "unresolved", "abandoned"],
            },
            "customer_sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
            "follow_ups": {"type": "array", "items": {"type": "string"}},
            "qa": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "check": {"type": "string", "enum": QA_CHECKS},
                        "result": {"type": "string", "enum": ["pass", "fail", "n/a"]},
                        "evidence": {"type": "string"},
                    },
                    "required": ["check", "result", "evidence"],
                },
            },
        },
        "required": ["summary", "disposition", "customer_sentiment", "follow_ups", "qa"],
    },
}


def _is_complete_record(record: Any) -> bool:
    required = RECORD_CALL_TOOL["input_schema"]["required"]
    return isinstance(record, dict) and all(k in record for k in required) and isinstance(record["qa"], list)


def _fill_missing_checks(record: dict[str, Any]) -> None:
    """Keeps the scorecard's rows fixed: any QA check the model skipped is shown as n/a."""
    seen = {q.get("check") for q in record["qa"] if isinstance(q, dict)}
    for check in QA_CHECKS:
        if check not in seen:
            record["qa"].append({"check": check, "result": "n/a", "evidence": "Not assessed."})
    order = {c: i for i, c in enumerate(QA_CHECKS)}
    record["qa"].sort(key=lambda q: order.get(q.get("check"), len(order)))


def wrap_up(call: Call, cfg: ModelConfig) -> dict[str, Any]:
    """After-call work: summary, disposition and a QA scorecard, from the transcript and the action log.

    The model answers through the record_call tool, so the record always matches its schema.
    """
    record = {
        "transcript": call.transcript,
        "actions": call.actions,
        "verified": call.verified_account is not None,
        "qa_checks": QA_CHECKS,
    }
    # Some models reject a forced tool_choice, so the tool is offered and the prompt requires it.
    # Without strict decoding the model can still drop a field, so incomplete records are retried.
    out: dict[str, Any] | None = None
    for _ in range(3):
        resp = cfg.client.messages.create(
            model=cfg.wrapup_model,
            max_tokens=1500,
            system=WRAPUP_PROMPT,
            tools=[RECORD_CALL_TOOL],
            messages=[{"role": "user", "content": json.dumps(record)}],
        )
        use = next((b for b in resp.content if b.type == "tool_use"), None)
        if use is not None and _is_complete_record(use.input):
            out = dict(use.input)
            break
    if out is None:
        raise ValueError("wrap-up model did not return a complete record_call")
    _fill_missing_checks(out)
    out["duration_s"] = round(time.time() - call.started)
    out["actions"] = call.actions
    return out


CALLER_PROMPT = """You are role-playing a customer phoning {company} support. Stay in character.

You are {caller}. Your phone number is {phone} and your account PIN is {pin}; give them only when asked.
Your situation: {brief}

Speak like a real person on the phone: one or two short sentences per turn, casual, no lists. React to
what the agent actually says. When your issue is handled (or you have been transferred), say a brief
goodbye. If the agent has said goodbye, reply with exactly [HANGUP]."""


def caller_turn(scenario: dict[str, Any], call: Call, cfg: ModelConfig) -> str:
    """The next line from a simulated caller playing `scenario`, or "[HANGUP]"."""
    # From the caller's side the roles flip: the agent's lines are the "user" turns.
    history: list[dict[str, Any]] = [{"role": "user", "content": "(The agent picks up.)"}]
    for line in call.transcript:
        role = "assistant" if line["speaker"] == "caller" else "user"
        if history and history[-1]["role"] == role:
            history[-1]["content"] += " " + line["text"]
        else:
            history.append({"role": role, "content": line["text"]})
    if history[-1]["role"] == "assistant":
        history.append({"role": "user", "content": "(silence)"})
    resp = cfg.client.messages.create(
        model=cfg.live_model,
        max_tokens=150,
        system=CALLER_PROMPT.format(company=COMPANY, **scenario),
        messages=history,
    )
    return " ".join(b.text for b in resp.content if b.type == "text").strip()
