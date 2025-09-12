# main.py — Chatbot 100% LLM
# Papel: professor bíblico cristocêntrico (foco em Cristo)
# Integração: Webhook do WhatsApp (Meta) + API LLM compatível com OpenAI

import os
import json
import requests
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
    """Envia mensagem para o WhatsApp Cloud API. 'to' deve ser sem '+'."""
    cfg = get_cfg()
    url = f"https://graph.facebook.com/{cfg['GRAPH_VERSION']}/{cfg['WABA_PHONE_NUMBER_ID']}/messages"
    headers = {
        "Authorization": f"Bearer {cfg['WABA_TOKEN']}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": str(to).lstrip("+"),  # <= importante: sem '+'
        "type": "text",
        "text": {"preview_url": False, "body": text},
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}
        res = {
            "status_code": r.status_code,
            "data": data,
            "debug_used": {
                "url": url,
                "graph_version": cfg["GRAPH_VERSION"],
                "phone_number_id": cfg["WABA_PHONE_NUMBER_ID"],
                "token_tail": cfg["WABA_TOKEN"][-8:] if cfg["WABA_TOKEN"] else "",
            },
        }
        print("[send_whatsapp_message]", json.dumps(res, ensure_ascii=False))
        return res
    except Exception as e:
        err = {"status_code": 0, "error": str(e)}
        print("[send_whatsapp_message][error]", err)
        return err

# -----------------------------
# LLM (OpenAI-compatible)
# -----------------------------
LLM_API_BASE = os.getenv("LLM_API_BASE", "").rstrip("/")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """
Você é um professor bíblico cristocêntrico, com foco em Cristo. Fale em PT-BR, com tom pastoral, humilde e encorajador. 

Objetivo: conversar naturalmente com os usuários, compartilhar a Palavra de Deus e oferecer orientação prática. Adapte a resposta à mensagem do usuário, de forma natural e espontânea.

Estilo:
- Respostas curtas (33-6 frases), claras, sem jargões.
- Baseie-se na Bíblia; inclua referências corretas quando citar versículos. Se não souber, diga honestamente que não sabe e sugira passagens relacionadas.
- Evite debates hostis ou temas político-partidários; promova paz (Hb 12:14).
- Ofereça oração breve e respeitosa quando o usuário pedir.
- Quando fizer sentido, inclua uma aplicação prática para a vida diária.
- Finalize com uma pergunta aberta ocasional, do tipo: “Quer se aprofundar em algum ponto?”, mas apenas se fizer sentido na conversa.

Importante:
- Se a mensagem do usuário for curta ou apenas uma saudação, responda de forma *natural, acolhedora e simples*, sem repetir toda a estrutura de versículo, aplicação e pergunta.
- Mantenha flexibilidade para variar a forma das respostas, de modo que cada interação pareça espontânea ehumana.
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
        content = (data.get("choices",[{}])[0].get("message",{}).get("content","")).strip()
        return content or "Posso ajudar com um estudo bíblico. Qual a sua dúvida?"
    except Exception as e:
        print("[llm_chat][error]", str(e))
        return "Tive um problema para gerar a resposta agora. Pode perguntar de novo?"

def llm_reply(user_text: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text or ""},
    ]
    out = llm_chat(msgs).strip()
    # Evita despedidas automáticas fora de contexto
    low = out.lower()
    if low in {"tchau", "até logo", "ate logo", "até mais", "ate mais"}:
        out = "Posso ajudar com uma passagem ou tema específico?"
    return out

# -----------------------------
# Webhook principal (POST)
# -----------------------------
@app.post("/webhook")
async def receive_webhook(request: Request):
    body = await request.json()
    print("Incoming webhook:", json.dumps(body, ensure_ascii=False))  # log bruto para depurar
    try:
        entry = body.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return JSONResponse({"status": "ignored"}, status_code=200)

        msg = messages[0]
        wa_from = msg.get("from")  # ex.: '5511999999999'
        msg_type = msg.get("type")
        text_body = msg.get("text", {}).get("body", "") if msg_type == "text" else ""

        # WhatsApp Cloud API espera o 'to' sem '+'
        to = str(wa_from).lstrip("+")

        # Chat 100% LLM (professor bíblico)
        reply = llm_reply(text_body)
        send_res = send_whatsapp_message(to, reply)
        return JSONResponse(
            {"status": "processed", "echo": reply, "send_result": send_res},
            status_code=200
        )

    except Exception as e:
        print("[webhook][error]", str(e))
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=200)

# -----------------------------
# Debug / Health
# -----------------------------
@app.get("/debug/send")
def debug_send(to: str, text: str = "ping"):
    """Envio de teste direto via URL (sem depender do LLM)."""
    to_clean = str(to).lstrip("+")
    res = send_whatsapp_message(to_clean, text)
    return {"ok": True, "to": to_clean, "result": res}

@app.get("/")
def root():
    return {"ok": True}
