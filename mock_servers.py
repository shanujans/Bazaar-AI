"""
Bazaar-AI — Local Mock Model Endpoints
=======================================
Simulates BOTH the Ollama edge endpoint and the AMD-cluster vLLM
endpoint on a single local process, so the whole Phase 1 demo can run
end-to-end without any real GPU hardware, Ollama install, or AMD Dev
Cloud access. Purely for hackathon rehearsal / offline demoing.

Run:
    uvicorn mock_servers:app --port 9000

Then point main.py at it via env vars before starting it:
    export EDGE_ENDPOINT_URL="http://localhost:9000/api/chat"
    export HEAVY_ENDPOINT_URL="http://localhost:9000/v1/chat/completions"
"""

import json
import time

from fastapi import FastAPI, Request

app = FastAPI(title="Bazaar-AI Mock Model Cluster")


def _mock_invoice_json(last_user_msg: str) -> dict:
    """
    Phase 3 rehearsal helper: the real gemma-4-31b-it will actually read
    INVOICE_SYSTEM_PROMPT and return schema-matching JSON on its own. This
    mock doesn't run a real model, so it can't extract real fields from
    last_user_msg -- it just returns a static, schema-correct payload so
    main.py's json.loads() -> receipt-formatting path can be rehearsed
    end-to-end offline. Swap for the real AMD endpoint before trusting the
    output of the parse itself (see README_PHASE_3.md).
    """
    return {
        "customer_name": "Ravi Kumar",
        "items": [
            {"item": "rice bags", "quantity": 10, "price_guess": 45.5},
            {"item": "cooking oil", "quantity": 5, "price_guess": 12.0},
        ],
        "total_discount_requested": "10% bulk discount",
    }


@app.post("/api/chat")
async def mock_ollama_chat(request: Request):
    """Mimics Ollama's /api/chat response shape for gemma-4-26b-a4b-it."""
    body = await request.json()
    messages = body.get("messages", [])
    last_user_msg = messages[-1]["content"] if messages else ""

    time.sleep(0.15)  # pretend this is fast, low-VRAM inference

    return {
        "model": body.get("model", "gemma-4-26b-a4b-it"),
        "message": {
            "role": "assistant",
            "content": (
                f"[edge-node mock] Quick answer for '{last_user_msg[:60]}'. "
                "Running light and fast on the quantized edge model."
            ),
        },
        "done": True,
    }


@app.post("/v1/chat/completions")
async def mock_vllm_completions(request: Request):
    """Mimics an OpenAI-compatible vLLM response for gemma-4-31b-it on AMD MI300X."""
    body = await request.json()
    messages = body.get("messages", [])
    last_user_msg = messages[-1]["content"] if messages else ""
    system_msg = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")

    time.sleep(0.6)  # pretend this is the heavier expert pass

    # Phase 3: main.py sends a distinct system prompt when it wants structured
    # invoice JSON back (see INVOICE_SYSTEM_PROMPT in main.py). A real vLLM-hosted
    # gemma-4-31b-it would read that instruction and comply on its own; this mock
    # doesn't run a real model, so it fakes compliance instead, purely so the
    # json.loads() -> receipt-formatting path can be rehearsed offline.
    if "structured data extraction engine" in system_msg:
        content = json.dumps(_mock_invoice_json(last_user_msg))
    else:
        content = (
            f"[AMD MI300X mock] Detailed reasoning for '{last_user_msg[:60]}'. "
            "This response simulates the full-precision expert model handling "
            "a complex negotiation/invoice task."
        )

    return {
        "id": "chatcmpl-mock-amd-mi300x",
        "object": "chat.completion",
        "model": body.get("model", "gemma-4-31b-it"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mock_servers:app", host="0.0.0.0", port=9000, reload=True)
