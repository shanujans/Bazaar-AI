"""
Bazaar-AI — Track 3: LocalFirst
Phase 1: Compute-Saving Cascade Router
=======================================

HACKATHON NARRATIVE
--------------------
Small businesses on WhatsApp send mostly trivial traffic — "hi", "is this
in stock?", "what's the price?" — mixed in with genuinely hard traffic:
multi-turn bulk-order negotiations, invoice/quote generation, and deep
code-switched vernacular (Hinglish/Taglish) reasoning.

Spinning up a full-precision LLM on our AMD MI300X Dev Cloud node for
every "hi 👋" is wasteful. So this router acts as a bouncer in front of
the cluster: it classifies each incoming WhatsApp-style message and only
wakes the expensive `gemma-4-31b-it` expert model when the task actually
justifies the VRAM. Everything routine gets served instantly by the
4-bit-quantized `gemma-4-26b-a4b-it` edge model.

Zero-cost principle: both models are self-hosted (Ollama/vLLM), so
`api_cost` is always $0.00 — the dashboard metric that matters here is
compute/VRAM avoided, not dollars saved.
"""

import os
import time
import logging
from typing import List, Dict

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Configuration (override via env vars — handy for pointing at mock_servers.py)
# ---------------------------------------------------------------------------
EDGE_MODEL_NAME = "gemma-4-26b-a4b-it"
HEAVY_MODEL_NAME = "gemma-4-31b-it"

# Local low-VRAM edge node (Ollama-style API)
EDGE_ENDPOINT_URL = os.getenv("EDGE_ENDPOINT_URL", "http://localhost:11434/api/chat")

# AMD MI300X Dev Cloud node, exposed as an OpenAI-compatible vLLM endpoint
HEAVY_ENDPOINT_URL = os.getenv("HEAVY_ENDPOINT_URL", "http://amd-cluster:8000/v1/chat/completions")

# Internal open-source network — no billing attached, dummy key only.
HEAVY_ENDPOINT_API_KEY = os.getenv("HEAVY_ENDPOINT_API_KEY", "sk-local-dummy-key-no-cost")

REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))

# If the mocked/self-hosted endpoints aren't reachable (e.g. Ollama not
# pulled, AMD cluster mock not running), fall back to a clearly-labeled
# simulated reply instead of crashing the live judge demo.
ALLOW_SIMULATED_FALLBACK = os.getenv("ALLOW_SIMULATED_FALLBACK", "true").lower() == "true"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("bazaar-ai.router")

# ---------------------------------------------------------------------------
# In-memory metrics store — powers the "Compute Saved" dashboard tile.
# (Swap for Redis/Postgres if you take this past the hackathon.)
# ---------------------------------------------------------------------------
METRICS = {
    "edge_compute_calls": 0,
    "heavy_compute_calls": 0,
    "fallback_calls": 0,
    "total_calls": 0,
}

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class ConversationTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    user_id: str
    message: str
    conversation_history: List[ConversationTurn] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    target: str  # "edge-compute" | "heavy-compute"
    model_name: str
    routing_reason: str
    complexity_score: int
    signals: Dict[str, bool]


class ChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    user_id: str
    reply: str
    model_used: str
    routing_reason: str
    api_cost: str
    compute_saved_vs_heavy: str
    latency_ms: int
    simulated_fallback: bool


# ---------------------------------------------------------------------------
# ComputeRouter — the heart of Phase 1
# ---------------------------------------------------------------------------
class ComputeRouter:
    """
    Heuristic / zero-shot semantic router.

    This is the "bouncer" standing in front of the AMD MI300X Dev Cloud.
    It keeps the expensive GPUs asleep for the large fraction of
    real-world small-business chatter (greetings, "do you have X in
    stock", single-item lookups) and only wakes gemma-4-31b-it on the
    AMD cluster when a conversation genuinely needs deep reasoning: bulk
    negotiation, invoice/tool generation, or code-switched vernacular
    that needs wider context to disambiguate.

    In production, swap the keyword bags below for a small fastText /
    langid classifier or a distilled intent model — the keyword-bag
    version is intentionally cheap-to-run and easy to demo live.
    """

    NEGOTIATION_KEYWORDS = {
        "negotiate", "negotiable", "discount", "bulk", "wholesale",
        "bargain", "counter offer", "counteroffer", "best price",
        "final price", "price break", "deal", "moq", "minimum order",
        "credit terms", "payment terms", "installment", "quotation",
    }

    TOOL_GENERATION_KEYWORDS = {
        "invoice", "generate invoice", "create quote", "quotation",
        "purchase order", "receipt", "billing", "gst", "tax invoice",
    }

    GREETING_KEYWORDS = {
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "namaste", "kumusta", "vanakkam", "salam",
    }

    SIMPLE_LOOKUP_PATTERNS = {
        "price of", "do you have", "in stock", "available", "how much is",
        "kitna hai", "magkano",
    }

    # A small sample of code-switch markers, just enough to demonstrate
    # routing *behavior* live. Not a real vernacular classifier.
    VERNACULAR_MARKERS = {
        # Hinglish
        "hai", "kya", "chahiye", "kitna", "bhai", "paisa", "acha", "theek",
        # Taglish
        "po", "opo", "pwede", "salamat", "magkano", "paano",
    }

    EDGE_TOKEN_THRESHOLD = 25          # words — beyond this, lean heavy
    HEAVY_HISTORY_TURNS_THRESHOLD = 4  # multi-turn negotiation signal
    COMPLEXITY_SCORE_THRESHOLD = 3     # >= this score routes to heavy-compute

    @staticmethod
    def _contains_any(text: str, vocabulary: set) -> bool:
        return any(term in text for term in vocabulary)

    def classify(self, message: str, history: List[ConversationTurn]) -> RoutingDecision:
        text = message.lower().strip()
        word_count = len(text.split())

        signals = {
            "has_negotiation_intent": self._contains_any(text, self.NEGOTIATION_KEYWORDS),
            "has_tool_generation_intent": self._contains_any(text, self.TOOL_GENERATION_KEYWORDS),
            "is_greeting_or_simple_lookup": (
                self._contains_any(text, self.GREETING_KEYWORDS)
                or self._contains_any(text, self.SIMPLE_LOOKUP_PATTERNS)
            ),
            "is_long_query": word_count > self.EDGE_TOKEN_THRESHOLD,
            "has_deep_vernacular_mix": (
                self._contains_any(text, self.VERNACULAR_MARKERS) and word_count > 8
            ),
            "is_multi_turn_negotiation": (
                len(history) >= self.HEAVY_HISTORY_TURNS_THRESHOLD
                and any(
                    self._contains_any(t.content.lower(), self.NEGOTIATION_KEYWORDS)
                    for t in history
                )
            ),
        }

        score = 0
        reasons = []

        # Each of these three is, on its own, exactly the kind of task the
        # brief calls out as "complex" — so each independently clears the
        # heavy-compute threshold rather than needing to stack with others.
        if signals["has_negotiation_intent"]:
            score += 3
            reasons.append("negotiation/financial intent detected")
        if signals["has_tool_generation_intent"]:
            score += 3
            reasons.append("invoice/quote tool-call required")
        if signals["has_deep_vernacular_mix"]:
            score += 3
            reasons.append("code-switched vernacular requiring deeper context")
        if signals["is_multi_turn_negotiation"]:
            score += 2
            reasons.append("sustained multi-turn negotiation thread")
        if signals["is_long_query"]:
            score += 1
            reasons.append(f"long query ({word_count} tokens)")
        if signals["is_greeting_or_simple_lookup"] and score == 0:
            score -= 2
            reasons.append("routine greeting/single-item lookup")

        if score >= self.COMPLEXITY_SCORE_THRESHOLD:
            target = "heavy-compute"
            model_name = HEAVY_MODEL_NAME
        else:
            target = "edge-compute"
            model_name = EDGE_MODEL_NAME

        reason_str = "; ".join(reasons) if reasons else "low token count & routine intent"

        return RoutingDecision(
            target=target,
            model_name=model_name,
            routing_reason=reason_str,
            complexity_score=score,
            signals=signals,
        )


router = ComputeRouter()

# ---------------------------------------------------------------------------
# Downstream model callers
# ---------------------------------------------------------------------------
def _history_to_messages(history: List[ConversationTurn], message: str) -> List[Dict[str, str]]:
    messages = [{"role": t.role, "content": t.content} for t in history]
    messages.append({"role": "user", "content": message})
    return messages


async def call_edge_model(message: str, history: List[ConversationTurn]) -> str:
    """
    Calls a local Ollama/vLLM-style endpoint serving the 4-bit-quantized
    gemma-4-26b-a4b-it edge model. Designed to run on commodity, low-VRAM
    hardware — the whole point of the zero-cost architecture.
    """
    payload = {
        "model": EDGE_MODEL_NAME,
        "messages": _history_to_messages(history, message),
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        resp = await client.post(EDGE_ENDPOINT_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # Ollama /api/chat response shape: {"message": {"role": .., "content": ..}, ...}
        return data.get("message", {}).get("content", "").strip()


async def call_heavy_model(message: str, history: List[ConversationTurn]) -> str:
    """
    Calls the AMD MI300X Dev Cloud node, exposed as an OpenAI-compatible
    vLLM endpoint serving the full-precision gemma-4-31b-it expert model.
    This is where the real reasoning horsepower lives — the router only
    wakes it up when the complexity score justifies it.
    """
    payload = {
        "model": HEAVY_MODEL_NAME,
        "messages": _history_to_messages(history, message),
        "temperature": 0.4,
    }
    headers = {
        # Dummy key — internal open-source cluster, no billing attached.
        "Authorization": f"Bearer {HEAVY_ENDPOINT_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        resp = await client.post(HEAVY_ENDPOINT_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # OpenAI-compatible response shape
        return data["choices"][0]["message"]["content"].strip()


def _simulated_reply(target: str, message: str) -> str:
    """
    Demo safety net. If Ollama isn't pulled yet, or the AMD cluster mock
    isn't running, we still want a *live, working* demo in front of
    judges — this returns a clearly-labeled simulated response instead
    of a 500 error. See `simulated_fallback` in the API response.
    """
    snippet = message[:80]
    if target == "edge-compute":
        return (
            f"[simulated edge reply] Quick answer for: \"{snippet}\". "
            "(gemma-4-26b-a4b-it endpoint unreachable — showing fallback response.)"
        )
    return (
        f"[simulated heavy reply] Working through this in detail: \"{snippet}\". "
        "(gemma-4-31b-it AMD cluster endpoint unreachable — showing fallback response.)"
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bazaar-AI — Compute-Saving Cascade Router",
    description=(
        "Phase 1 of Bazaar-AI (Track 3: LocalFirst). Routes WhatsApp-style "
        "business chatter between a quantized edge model and a full "
        "gemma-4-31b-it expert model on an AMD MI300X Dev Cloud node — "
        "zero commercial API cost, 100% open-source stack."
    ),
    version="0.1.0",
)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "edge_model": EDGE_MODEL_NAME,
        "edge_endpoint": EDGE_ENDPOINT_URL,
        "heavy_model": HEAVY_MODEL_NAME,
        "heavy_endpoint": HEAVY_ENDPOINT_URL,
    }


@app.get("/api/metrics")
async def metrics():
    """
    Powers the hackathon dashboard's "Compute Saved" / "VRAM avoided"
    tiles: what fraction of traffic never touched the expensive AMD
    MI300X cluster.
    """
    total = METRICS["total_calls"] or 1
    edge_ratio = METRICS["edge_compute_calls"] / total
    return {
        **METRICS,
        "edge_compute_ratio": round(edge_ratio, 3),
        "estimated_vram_avoided": f"{round(edge_ratio * 100, 1)}% of requests never spun up MI300X",
        "total_api_cost": "$0.00",
    }


@app.post("/api/chat/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.perf_counter()

    decision = router.classify(req.message, req.conversation_history)
    simulated_fallback = False

    try:
        if decision.target == "edge-compute":
            reply = await call_edge_model(req.message, req.conversation_history)
        else:
            reply = await call_heavy_model(req.message, req.conversation_history)
        if not reply:
            raise ValueError("empty response from model endpoint")
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.warning("Downstream model call failed (%s); falling back.", exc)
        if not ALLOW_SIMULATED_FALLBACK:
            raise HTTPException(status_code=502, detail=f"Model endpoint unreachable: {exc}")
        reply = _simulated_reply(decision.target, req.message)
        simulated_fallback = True
        METRICS["fallback_calls"] += 1

    METRICS["total_calls"] += 1
    if decision.target == "edge-compute":
        METRICS["edge_compute_calls"] += 1
        compute_saved = "high"
    else:
        METRICS["heavy_compute_calls"] += 1
        compute_saved = "n/a — this request needed the full AMD MI300X cluster"

    latency_ms = int((time.perf_counter() - start) * 1000)

    return ChatResponse(
        user_id=req.user_id,
        reply=reply,
        model_used=decision.model_name,
        routing_reason=decision.routing_reason,
        api_cost="$0.00",
        compute_saved_vs_heavy=compute_saved,
        latency_ms=latency_ms,
        simulated_fallback=simulated_fallback,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
