# Call-center agent on Chalk

A phone-support agent for **Halyard Mobile**, a fictional carrier. Call it from your browser,
or watch a simulated caller. The agent verifies you, reads your account, and acts through tools:
it can issue credits, suspend a stolen phone's line, ship a SIM, and change plans. It hands off to
a human when policy says so. Hang up and a second model writes the after-call record: summary,
disposition, follow-ups, and a QA scorecard that quotes its evidence.

```
browser mic ─ speech recognition ─▶ /say ─▶ Claude (live model) ⇄ tools on a per-call CRM ─▶ reply ─ speech synthesis ─▶ speaker
                                                                       │
                                                         hang up ─▶ Claude (wrap-up model) ─▶ summary · disposition · QA
```

| File | Role |
| --- | --- |
| `agent.py` | The fake CRM, the tools and the policies they enforce, the agent loop, the simulated caller, and the wrap-up. |
| `server.py` | FastAPI app: starts calls, streams each turn as NDJSON, runs simulations and wrap-ups, rate-limits visitors. |
| `index.html` | The call page: scenarios, live transcript with tool calls, agent desktop, after-call work. |
| `deploy.py` | Deploys it as a `chalkcompute.ScalingGroup` that reaches Claude through the Chalk model gateway. |

## Guardrails live in the tools

The system prompt describes the policy, but each tool enforces it as well, so a caller who talks
the model into something still hits a wall:

- Every account tool refuses until `verify_identity` succeeds, and verification locks after three
  failed attempts.
- The agent sees the account through a mask: never the PIN, only the card's last four digits.
- `issue_credit` refuses more than $50 per call, which pushes the agent to escalate.
- `apply_retention_offer` works once per call.
- Each call gets its own copy of the CRM, so one caller's credits and suspensions never show up
  in another call.

The QA review sees the transcript *and* every tool call with its result. So it can fail an agent
that claimed something it never looked up, or flag a fix that won't land in time. In testing it
flagged that a plan change "effective next bill" might not cover a trip next week.

## Run it

You need [`uv`](https://docs.astral.sh/uv/) and a Chalk environment with the model gateway
enabled (`chalk router status`). The environment also needs a router API key stored as the
secret `CHALK_ROUTER_API_KEY` (`chalk router api-key create`).

```bash
./deploy.py # CPU only: the models run behind the gateway
./deploy.py
```

Open the printed `web_url`. **Call** uses your microphone through the browser's speech
recognition (Chrome, Edge or Safari), and the agent answers out loud. You can always type instead.
**Watch** plays the scenario with a simulated caller, which is another model given the persona.

## Models

Live turns use `anthropic/claude-haiku-4-5` by default. A turn with one tool call takes about
1–2 s per model round trip, which keeps the conversation feeling like a phone call. The wrap-up
runs once per call on `anthropic/claude-sonnet-5-5`. Both go through the Chalk model gateway, so
the provider key, usage and tracing stay in your Chalk environment.

## Limits

The page is public, so the server caps visitors at 12 calls and 4 simulated calls per hour. A
call is capped at 24 turns and 400 characters per utterance, and calls expire after 20 minutes.
Calls live in the replica's memory, which is why the scaling group runs one replica.

## License

Apache-2.0. Halyard Mobile, its customers and their phone numbers (in the 555-01xx range) are
fictional.

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported.
The examples use the current environment selected by your Chalk login.
