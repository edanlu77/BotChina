# app/whats.py
import requests

def normalize_phone(num: str) -> str:
    n = (num or "").strip().replace("+", "").replace(" ", "")
    for ch in ("-", "(", ")", "."):
        n = n.replace(ch, "")
    return n

def wa_send_text_tenant(tenant: dict, to: str, body: str) -> dict:
    to_norm = normalize_phone(to)
    url = f"https://graph.facebook.com/v22.0/{tenant['wa_phone_id']}/messages"
    headers = {
        "Authorization": f"Bearer {tenant['wa_token']}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_norm,
        "type": "text",
        "text": {"body": body[:4096]},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    try:
        j = r.json()
    except Exception:
        j = {"raw": r.text}
    if r.status_code >= 400:
        raise RuntimeError(f"WhatsApp send error {r.status_code}: {j}")
    return j
