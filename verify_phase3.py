"""
verify_phase3.py — smoke-tests Bazaar-AI's Phase 3 invoice JSON tool-calling
feature (and re-checks Phase 1/2 routing) against a running main.py instance.

Zero extra dependencies — uses only the Python standard library, so it runs
in the same venv as main.py with nothing extra to install.

Usage:
    python verify_phase3.py [base_url]

Defaults to http://localhost:8080 if base_url is omitted.

Before running:
  1. Start the mock cluster:   python mock_servers.py
  2. Start the backend:        uvicorn main:app --host 0.0.0.0 --port 8080
     (with EDGE_ENDPOINT_URL / HEAVY_ENDPOINT_URL pointed at the mock cluster)
"""
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

PASS = "PASS"
FAIL = "FAIL"

results = []


def post_json(path, payload):
    url = BASE_URL + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_form(path, form: dict):
    url = BASE_URL + path
    data = urllib.parse.urlencode(form).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8"), resp.status


def get_json(path):
    url = BASE_URL + path
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status))
    suffix = f"  ({detail})" if detail and not condition else ""
    print(f"[{status}] {name}{suffix}")


def main():
    print(f"Testing Bazaar-AI backend at {BASE_URL}\n")

    try:
        health = get_json("/api/health")
        check("Backend reachable (/api/health)", True)
    except Exception as e:
        check("Backend reachable (/api/health)", False, str(e))
        print("\nBackend not reachable — start main.py first. Aborting.")
        sys.exit(1)

    # --- Edge routing ---
    try:
        r = post_json("/api/chat/", {"user_id": "verify-edge", "message": "Hi there!"})
        check("Simple greeting routes to edge", r.get("model_used") == health["edge_model"], r.get("model_used"))
        check("Edge routing_reason mentions greeting", "greeting" in r.get("routing_reason", "").lower())
    except Exception as e:
        check("Edge routing test", False, str(e))

    # --- Heavy routing (non-invoice) ---
    try:
        r = post_json(
            "/api/chat/",
            {"user_id": "verify-heavy", "message": "I want to negotiate a bulk wholesale discount for 500 units"},
        )
        check("Negotiation message routes to heavy", r.get("model_used") == health["heavy_model"], r.get("model_used"))
        check(
            "No false-positive vernacular flag on plain English negotiation text",
            "vernacular" not in r.get("routing_reason", "").lower(),
            r.get("routing_reason"),
        )
        check("No invoice_data on a non-invoice heavy call", "invoice_data" not in r)
    except Exception as e:
        check("Heavy routing test", False, str(e))

    # --- Invoice JSON tool-calling (the knockout feature) ---
    try:
        r = post_json(
            "/api/chat/",
            {
                "user_id": "verify-invoice",
                "message": (
                    "Please generate an invoice for 10 rice bags and 5 cooking oil "
                    "for customer Ravi Kumar with a 10% bulk discount"
                ),
            },
        )
        check("Invoice message routes to heavy", r.get("model_used") == health["heavy_model"])
        check(
            "routing_reason flags invoice/tool-generation",
            "invoice/tool-generation" in r.get("routing_reason", "").lower(),
        )
        has_invoice_data = "invoice_data" in r
        check("Response includes parsed invoice_data", has_invoice_data)
        if has_invoice_data:
            inv = r["invoice_data"]
            check("invoice_data has customer_name", bool(inv.get("customer_name")))
            check(
                "invoice_data has a non-empty items list",
                isinstance(inv.get("items"), list) and len(inv["items"]) > 0,
            )
            check("reply is formatted as a receipt (contains the invoice emoji)", "🧾" in r.get("reply", ""))
        else:
            check(
                "Fallback: raw text still returned when JSON parse fails",
                isinstance(r.get("reply"), str) and len(r["reply"]) > 0,
            )
    except Exception as e:
        check("Invoice tool-calling test", False, str(e))

    # --- Metrics sanity ---
    try:
        m = get_json("/api/metrics")
        check("/api/metrics total_calls >= 3", m.get("total_calls", 0) >= 3, m.get("total_calls"))
        check("/api/metrics api_cost is $0.00", m.get("total_api_cost") == "$0.00")
    except Exception as e:
        check("Metrics check", False, str(e))

    # --- Routing log sanity ---
    try:
        log = get_json("/api/routing-log?limit=5")
        check("/api/routing-log returns entries", isinstance(log, list) and len(log) > 0)
    except Exception as e:
        check("Routing log check", False, str(e))

    # --- WhatsApp webhook (Twilio-shaped form POST) ---
    try:
        body, status = post_form("/api/whatsapp/", {"Body": "Hi there!", "From": "whatsapp:+10000000000"})
        check("WhatsApp webhook returns 200", status == 200, status)
        check("WhatsApp webhook returns TwiML XML", "<Response>" in body and "<Message>" in body)
    except Exception as e:
        check("WhatsApp webhook test", False, str(e))

    total = len(results)
    passed = sum(1 for _, s in results if s == PASS)
    print(f"\n{passed}/{total} checks passed.")
    if passed < total:
        print("\nCommon causes for failures:")
        print("  - mock_servers.py isn't running, or EDGE_ENDPOINT_URL / HEAVY_ENDPOINT_URL are misconfigured")
        print("  - main.py running is an older version without the Phase 3 invoice logic")
        sys.exit(1)


if __name__ == "__main__":
    main()
