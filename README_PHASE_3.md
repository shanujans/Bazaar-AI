# Phase 3 — AMD MI300X Deployment Guide (vLLM, ROCm)

This is the exact sequence to stand up `gemma-4-31b-it` as an OpenAI-compatible
endpoint on the AMD Dev Cloud, so `HEAVY_ENDPOINT_URL` in `main.py` can point
at real silicon instead of `mock_servers.py`.

---

## 0. Prerequisites

- ROCm 6.x drivers installed and visible:
  ```bash
  rocm-smi
  ```
  You should see your MI300X GPU(s) listed with memory/utilization stats. If
  this fails, stop here — the deploy commands below will fail too.
- Python 3.11+ available on the node.
- Model weights for `gemma-4-31b-it` staged locally (e.g. under `/data/models/gemma-4-31b-it`)
  or accessible via your model registry / Hugging Face cache.

---

## 1. Option A — Docker (recommended)

AMD ships a ROCm-prebuilt vLLM image, which avoids the usual pain of building
vLLM's ROCm kernels from source.

```bash
docker pull rocm/vllm:latest

docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --ipc=host \
  --shm-size 16g \
  --network host \
  -v /data/models:/models \
  rocm/vllm:latest \
  vllm serve /models/gemma-4-31b-it \
    --served-model-name gemma-4-31b-it \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code
```

**Why these flags:**
- `--dtype bfloat16` — matches the "Unquantized / Full Context" spec for the heavy tier; MI300X has native bf16 support.
- `--tensor-parallel-size 1` — a single MI300X (192GB HBM3e) comfortably holds a 31B bf16 model (~62GB weights + KV cache headroom), so no sharding is needed by default. Bump this only if you're handed multiple GPUs and want more headroom (see §3).
- `--max-model-len 8192` — safe default context window; raise it if your prompts (long negotiation histories, multi-turn WhatsApp threads) need more room and you've confirmed VRAM headroom.
- `--gpu-memory-utilization 0.90` — leaves ~10% headroom so the process doesn't OOM under bursty concurrent requests during the judges' demo.
- `--served-model-name gemma-4-31b-it` — **must** match `HEAVY_MODEL_NAME` in `main.py` exactly, since the client sends this string in the `model` field of every request.
- `--trust-remote-code` — required if the Gemma-4 checkpoint ships a custom modeling file.

---

## 2. Option B — Bare metal (no Docker)

```bash
# one-time setup
python3.11 -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install --upgrade pip

# Install the ROCm-flavored vLLM build (check AMD Dev Cloud's provided
# index URL / wheel — it varies by ROCm version; ask your Dev Cloud docs
# or environment onboarding for the exact --extra-index-url to use)
pip install vllm --extra-index-url <AMD-Dev-Cloud-provided-ROCm-wheel-index>

export HIP_VISIBLE_DEVICES=0
export PYTORCH_ROCM_ARCH=gfx942   # MI300X architecture target

vllm serve /data/models/gemma-4-31b-it \
  --served-model-name gemma-4-31b-it \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
```

---

## 3. Scaling to multiple MI300X GPUs

If the Dev Cloud allocation includes more than one MI300X and you want extra
throughput or a longer context window:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3

vllm serve /data/models/gemma-4-31b-it \
  --served-model-name gemma-4-31b-it \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
```

`--tensor-parallel-size` must evenly divide the model's attention-head count —
`1`, `2`, `4`, and `8` are the safe values to try.

---

## 4. Securing the endpoint for the demo

By default vLLM's OpenAI-compatible server accepts any bearer token. For the
judges' demo (endpoint reachable on the Dev Cloud's internal network), that's
fine — it's what the `dummy-internal-no-billing` key in `skill.md` assumes.
If you want a real check:

```bash
vllm serve /data/models/gemma-4-31b-it \
  --served-model-name gemma-4-31b-it \
  --api-key demo-only-not-a-secret \
  ... # other flags as above
```

and set `HEAVY_ENDPOINT_API_KEY=demo-only-not-a-secret` in `main.py`'s environment.

---

## 5. Point Bazaar-AI at the live endpoint

```bash
export HEAVY_ENDPOINT_URL="http://<amd-dev-cloud-hostname>:8000/v1/chat/completions"
export HEAVY_ENDPOINT_API_KEY="dummy-internal-no-billing"   # or your --api-key value

uvicorn main:app --host 0.0.0.0 --port 8080
```

`EDGE_ENDPOINT_URL` stays pointed at your Ollama instance (or `mock_servers.py`
if the edge tier isn't live yet) — the two tiers are configured independently.

---

## 6. Smoke test the raw endpoint

Before wiring it into `main.py`, confirm vLLM itself is healthy:

```bash
curl http://<amd-dev-cloud-hostname>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-31b-it",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}]
  }'
```

You should get back a normal OpenAI-shaped `choices[0].message.content` response.

Then confirm the invoice-JSON path specifically (this is what Phase 3's
knockout feature depends on):

```bash
curl http://<amd-dev-cloud-hostname>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-31b-it",
    "messages": [
      {"role": "system", "content": "You are a structured data extraction engine... (see INVOICE_SYSTEM_PROMPT in main.py)"},
      {"role": "user", "content": "Bhai 10 rice bags aur 5 cooking oil chahiye, customer Ravi Kumar, 10% bulk discount"}
    ],
    "temperature": 0.1
  }'
```

The `content` field should come back as a raw JSON object (no markdown
fences, no prose) matching the invoice schema in `main.py`.

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `rocm-smi` shows no GPUs | Driver/container device mapping issue — recheck `--device=/dev/kfd`, `--device=/dev/dri`, and `video`/`render` group membership |
| OOM on startup | Lower `--gpu-memory-utilization` or `--max-model-len` |
| `vllm serve` flag not recognized | Version mismatch — run `vllm serve --help` inside the container/venv to confirm the flags supported by the installed version |
| `main.py` gets `simulated_fallback: true` even though vLLM is up | Check `HEAVY_ENDPOINT_URL` matches the actual host:port, and that nothing (firewall, ngrok, security group) is blocking the Dev Cloud → Bazaar-AI backend path |
| Invoice replies aren't parsing as JSON (`invoice_data` missing from the response) | Model added prose despite instructions — `main.py` already tries to salvage a `{...}` block from stray text, but if it still fails, tighten `INVOICE_SYSTEM_PROMPT` or lower `temperature` further |

---

## 8. What's still open for post-hackathon hardening

(Carried over from the Phase 3 candidates list in `skill.md` — not blockers for the demo.)

- Structured JSON-schema output enforcement (vLLM's guided-decoding / `response_format`) instead of prompt-only JSON instructions, for a harder guarantee than "the model usually complies."
- Persisting `METRICS` / `ROUTING_LOG` / `WHATSAPP_HISTORY` beyond process memory.
- Twilio webhook signature verification on `/api/whatsapp/`.
