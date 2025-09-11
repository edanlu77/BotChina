# main.py — Chatbot 100% LLM
# Papel: professor bíblico cristocêntrico (foco em Cristo)
# Integração: Webhook do WhatsApp (Meta) + API LLM compatível com OpenAI

import os, json, requests
from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv(override=True)
app = FastAPI()

# -----------------------------
# Webhook Verify (Meta)
# -----------------------------
@app.get("/webhook")
def verify_webhook(
    mode: str = Query("", alias="hub.mode"),
    verify_token: str = Query("", alias="hub.verify_token"),
    challenge: str = Query("", alias="hub.challenge"),
):
    VERIFY = os.getenv("WABA_VERIFY_TOKEN", "botchina-verify")
    if verify_token == VERIFY:
        return PlainTextResponse(challenge or "ok", status_code=200)
    return PlainTextResponse("forbidden", status_code=403)

# -----------------------------
# WhatsApp Send
# -----------------------------
def get_cfg():
    return {
        "GRAPH_VERSION": os.getenv("GRAPH_VERSION", "v23.0"),
        "WABA_TOKEN": os.getenv("WABA_TOKEN", ""),
        "WABA_PHONE_NUMBER_ID": os.getenv("WABA_PHONE_NUMBER_ID", ""),
    }

def send_whatsapp_message(to: str, text: str) -> dict:
    cfg = get_cfg()
    url = f"https://graph.facebook.com/{cfg['GRAPH_VERSION']}/{cfg['WABA_PHONE_NUMBER_ID']}/messages"
    headers = {"Authorization": f"Bearer {cfg['WABA_TOKEN']}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": text},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text}
    return {"status_code": r.status_code, "data": data, "url": url}

# -----------------------------
# LLM (OpenAI-compatible)
# -----------------------------
LLM_API_BASE = os.getenv("LLM_API_BASE", "").rstrip("/")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """
Você é um professor bíblico cristocêntrico, com foco em Cristo.
Fale em PT-BR, com tom pastoral, humilde e encorajador.
Estilo: respostas curtas (3–6 frases), claras, sem jargões.
Baseie-se na Bíblia; quando citar, inclua referências (ex.: João 3:16).
Nunca invente versículos; se não souber, diga que não sabe e sugira passagens.
Evite debates hostis e temas político-partidários; promova paz (Hb 12:14).
Quando o usuário pedir oração, ofereça uma oração breve e respeitosa.
Quando apropriado, inclua 1 aplicação prática para a vida diária.
Finalize, quando fizer sentido, com uma pergunta aberta do tipo: “Quer se aprofundar em algum ponto?”
"""

def llm_chat(messages: list[dict]) -> str:
    if not (LLM_API_BASE and LLM_API_KEY):
        return "Configuração de LLM ausente. Defina LLM_API_BASE e LLM_API_KEY."
    url = f"{LLM_API_BASE}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "temperature": 0.4,
        "max_tokens": 300,
        "messages": messages,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        data = resp.json()
        return (data.get("choices",[{}])[0].get("message",{}).get("content","")).strip() or "Posso ajudar com um estudo bíblico. Qual a sua dúvida?"
    except Exception as e:
        print("[llm error]", e)
        return "Tive um problema para gerar a resposta agora. Pode perguntar de novo?"

def llm_reply(user_text: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text or ""},
    ]
    out = llm_chat(msgs)
    # Evita despedidas automáticas fora de contexto
    if out.lower().strip() in {"tchau", "até logo", "ate logo"}:
        out = "Posso ajudar com uma passagem ou tema específico?"
    return out

# -----------------------------
# Webhook principal (POST)
# -----------------------------
@app.post("/webhook")
async def receive_webhook(request: Request):
    body = await request.json()
    try:
        entry = body.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return JSONResponse({"status": "ignored"}, status_code=200)

        msg = messages[0]
        wa_from = msg.get("from")  # ex.: '55119...'
        msg_type = msg.get("type")
        text_body = msg.get("text", {}).get("body", "") if msg_type == "text" else ""

        to = f"+{wa_from}"

        # Chat 100% LLM (professor bíblico)
        reply = llm_reply(text_body)
        send_res = send_whatsapp_message(to, reply)
        return JSONResponse({"status": "processed", "echo": reply, "send_result": send_res}, status_code=200)

    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=200)

# -----------------------------
# Healthcheck
# -----------------------------
@app.get("/")
def root():
    return {"ok": True}
