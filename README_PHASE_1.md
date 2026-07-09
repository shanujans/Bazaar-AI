# Bazaar-AI — Phase 1: Compute-Saving Cascade Router

Track 3 (LocalFirst) · Zero-Cost Edition

This is the async FastAPI backend that intercepts WhatsApp-style
messages and routes each one to the cheapest model that can handle it:

| Traffic type | Example | Routed to |
|---|---|---|
| Greetings, single-item lookups | "hi", "do you have this in stock?" | `gemma-4-26b-a4b-it` (edge, quantized, low-VRAM) |
| Multi-turn negotiation, invoice/tool generation, deep code-switched vernacular | "let's negotiate a bulk discount…", "generate an invoice for order #4521" | `gemma-4-31b-it` (heavy, AMD MI300X Dev Cloud) |

Every response includes routing metadata (`model_used`, `routing_reason`,
`api_cost`, `compute_saved_vs_heavy`) so the judges' dashboard can show
live "compute saved" numbers instead of dollar savings.

## Files

- `main.py` — the FastAPI app, `ComputeRouter`, and the two downstream model callers
- `mock_servers.py` — optional local stand-ins for Ollama + the AMD vLLM endpoint, so you can demo the whole pipeline with no real GPU hardware
- `requirements.txt` — pinned dependencies

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run with the built-in mocked model endpoints (fastest path — no GPUs, no Ollama)

The app already fails **gracefully** if the real endpoints aren't reachable:
it returns a clearly-labeled simulated reply (`simulated_fallback: true`)
instead of crashing, so `uvicorn main:app --port 8080` will demo fine
completely on its own.

For a more realistic demo — actual round-trip HTTP calls with slightly
different latencies per tier — spin up the included mock cluster first:

**Terminal 1 — mock model cluster (simulates Ollama + the AMD vLLM node):**
```bash
uvicorn mock_servers:app --port 9000
```

**Terminal 2 — the router app, pointed at the mocks:**
```bash
export EDGE_ENDPOINT_URL="http://localhost:9000/api/chat"
export HEAVY_ENDPOINT_URL="http://localhost:9000/v1/chat/completions"
uvicorn main:app --port 8080 --reload
```

## 3. Run against real infrastructure

Once you have Ollama serving `gemma-4-26b-a4b-it` locally and vLLM
serving `gemma-4-31b-it` on the AMD MI300X node, just point the env vars
at them (defaults already assume this setup):

```bash
export EDGE_ENDPOINT_URL="http://localhost:11434/api/chat"           # Ollama default
export HEAVY_ENDPOINT_URL="http://amd-cluster:8000/v1/chat/completions"  # your vLLM host
uvicorn main:app --port 8080
```

No API keys are required beyond the dummy bearer token — both endpoints
are internal, self-hosted, and free.

## 4. Try it

```bash
curl -X POST http://localhost:8080/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "shopkeeper_001",
    "message": "Do you have blue sarees in stock?",
    "conversation_history": []
  }'
```

Expect `model_used: gemma-4-26b-a4b-it` and `routing_reason: routine
greeting/single-item lookup`.

```bash
curl -X POST http://localhost:8080/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "shopkeeper_001",
    "message": "I want to negotiate a bulk discount for 500 units, whats your best wholesale price and payment terms?",
    "conversation_history": []
  }'
```

Expect `model_used: gemma-4-31b-it` and `routing_reason: negotiation/
financial intent detected`.

## 5. Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat/` | Main chat endpoint — routes + calls the appropriate model |
| `GET` | `/api/health` | Confirms which endpoints the router is currently pointed at |
| `GET` | `/api/metrics` | Cumulative edge-vs-heavy call counts — feeds the pitch dashboard |
| `GET` | `/docs` | Auto-generated Swagger UI (FastAPI default) |

## 6. How `ComputeRouter` decides

It scores each message on a few zero-shot heuristic signals:

- **Negotiation/financial intent** — keyword bag (`negotiate`, `bulk`,
  `wholesale`, `payment terms`, …)
- **Tool-generation intent** — keyword bag (`invoice`, `quotation`,
  `purchase order`, …)
- **Deep code-switched vernacular** — Hinglish/Taglish marker words
  combined with message length
- **Multi-turn negotiation** — negotiation keywords appearing earlier in
  `conversation_history`, not just the current message
- **Token count** — a minor tiebreaker signal

Any one of the first three signals is, by design, enough on its own to
route to `heavy-compute` — each represents exactly the kind of task the
brief defines as "complex." Everything else defaults to `edge-compute`.

This is intentionally a transparent, inspectable heuristic rather than a
black-box classifier — easy to explain to judges in 30 seconds, and easy
to swap for a distilled intent model post-hackathon without touching the
FastAPI plumbing around it.
