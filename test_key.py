# test_key.py
import os
import httpx

api_key = os.getenv("FIREWORKS_API_KEY", "").strip().replace('"', '').replace("'", "")
if not api_key:
    print("\n❌ ERROR: FIREWORKS_API_KEY is empty in this terminal! Run 'set FIREWORKS_API_KEY=yourkey' first.\n")
else:
    print(f"\n🔑 Loaded API Key: {api_key[:8]}...{api_key[-4:]}\n")

url = "https://api.fireworks.ai/inference/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 100% valid dictionary structure using Fireworks' active serverless model
payload = {
    "model": "accounts/fireworks/models/gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.1
}

if api_key:
    try:
        print("Sending request to Fireworks...")
        resp = httpx.post(url, json=payload, headers=headers, timeout=10.0)
        print(f"\n📡 Status Code: {resp.status_code}")
        print(f"💬 Response Body: {resp.text}\n")
    except Exception as e:
        print(f"\n❌ Network Error: {e}\n")