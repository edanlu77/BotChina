import os
import re
import json
import requests
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv

# Carrega variáveis do .env quando rodar local
load_dotenv(override=True)

app = FastAPI()

# -----------------------------
# Verificação do Webhook (Meta)
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
# Config da API do WhatsApp
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
    return {
        "status_code": r.status_code,
        "data": data,
        "debug_used": {
            "graph_version": cfg["GRAPH_VERSION"],
            "phone_number_id": cfg["WABA_PHONE_NUMBER_ID"],
            "token_tail": cfg["WABA_TOKEN"][-8:] if cfg["WABA_TOKEN"] else "",
            "token_len": len(cfg["WABA_TOKEN"]),
            "url": url,
        },
    }

# -----------------------------
# Catálogo (CSV via Google Sheets)
# -----------------------------
SESSIONS: dict[str, dict] = {}  # estado simples por wa_id
CATALOG = None
CATALOG_URL = os.getenv("CATALOG_URL", "").strip()

def load_catalog() -> None:
    """Carrega o catálogo do CSV público (CATALOG_URL)."""
    global CATALOG
    if not CATALOG_URL:
        CATALOG = None
        return
    try:
        df = pd.read_csv(CATALOG_URL).fillna("")
        # Espera colunas: sku,nome,sinonimos,descricao,cor,preco,moeda,lead_time,estoque,image_url
        CATALOG = df
        print(f"[catalog] loaded {len(CATALOG)} rows from {CATALOG_URL}")
    except Exception as e:
        CATALOG = None
        print(f"[catalog] load error: {e}")

load_catalog()

def search_catalog(query: str, topn: int = 3):
    """Busca simples por palavras em nome/sinonimos/descricao e retorna top N."""
    if CATALOG is None or CATALOG.empty or not (query or "").strip():
        return []
    q_tokens = [w for w in re.split(r"[^a-zA-Z0-9\u00C0-\u017F]+", query.lower()) if w]
    scored = []
    for _, row in CATALOG.iterrows():
        hay = f"{row.get('nome','')} {row.get('sinonimos','')} {row.get('descricao','')}".lower()
        score = sum(1 for tok in q_tokens if tok in hay)
        if score > 0:
            item = row.to_dict()
            item["_score"] = int(score)
            scored.append(item)
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:topn]

# -----------------------------
# Router + helpers de conversa
# -----------------------------
def route_command(text: str) -> str | None:
    t = (text or "").strip()
    if not t.startswith("/"):
        return None
    cmd, *rest = t.split(maxsplit=1)
    arg = rest[0] if rest else ""
    if cmd.lower() == "/help":
        return ("👋 Posso ajudar com:\n"
                "• Descreva o produto e a quantidade (ex: '3 amoladores de faca')\n"
                "• /sku <código>\n"
                "• /humano para falar com atendente")
    if cmd.lower() == "/humano":
        return "✅ Vou acionar um atendente humano. Envie nome/empresa e melhor horário."
    if cmd.lower() == "/sku":
        return f"Digite o código do SKU após /sku. Você enviou: {arg or '(faltou o código)'}"
    return "🙃 Comando não reconhecido. Digite /help."

def is_greeting(t: str) -> bool:
    t = (t or "").strip().lower()
    return any(g in t for g in ["oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "hello", "hi", "hey"])

def extract_items(text: str):
    """Extrai pares quantidade + descrição. Ex.: '3 carrinhos, 5 impressoras'."""
    pats = re.findall(r"(\d+)\s*(x|X)?\s*([a-zA-Z0-9\u00C0-\u017F\s\-_/]+?)(?:,| e |;|$)", text or "")
    items = [{"qty": int(q), "name": n.strip()} for q, _, n in pats if n.strip()]
    return items or [{"qty": 1, "name": (text or '').strip()}]

def format_candidate_line(i, c):
    preco = f"{c.get('moeda','')}{c.get('preco','')}".strip()
    estoque = c.get('estoque','?')
    nome = c.get('nome','').strip()
    sku = c.get('sku','—')
    return f"{i}) {sku} — {nome} • {preco} • estoque: {estoque}"

def show_candidates_text(to: str, query: str, cands: list):
    lines = [f"Encontrei opções para “{query}”:"]
    for i, c in enumerate(cands, start=1):
        lines.append(format_candidate_line(i, c))
    lines.append("Qual você prefere? Responda 1, 2 ou 3.")
    send_whatsapp_message(to, "\n".join(lines))

# -----------------------------
# Webhook (POST): lógica principal
# -----------------------------
@app.post("/webhook")
async def receive_webhook(request: Request):
    body = await request.json()
    print("Incoming webhook:", json.dumps(body, ensure_ascii=False))

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
        sess = SESSIONS.setdefault(wa_from, {})

        # 1) Comandos com "/"
        cmd_reply = route_command(text_body)
        if cmd_reply:
            send_res = send_whatsapp_message(to, cmd_reply)
            return JSONResponse({"status": "processed", "echo": cmd_reply, "send_result": send_res}, status_code=200)

        # 2) Escolha pendente (1/2/3)
        if sess.get("awaiting_choice"):
            choice = (text_body or "").strip()
            if choice in {"1", "2", "3"}:
                idx = int(choice) - 1
                cands = sess["awaiting_choice"]["cands"]
                if idx < len(cands):
                    chosen = cands[idx]
                    sess.pop("awaiting_choice", None)
                    sess["pending_item"] = {"sku": chosen.get("sku"), "nome": chosen.get("nome")}
                    msg_out = (f"Você escolheu {chosen.get('sku')} — {chosen.get('nome')}.\n"
                               f"Quantas unidades deseja?")
                    send_res = send_whatsapp_message(to, msg_out)
                    return JSONResponse({"status": "processed", "echo": msg_out, "send_result": send_res}, status_code=200)
            send_res = send_whatsapp_message(to, "Por favor, responda 1, 2 ou 3.")
            return JSONResponse({"status": "processed", "echo": "aguardando 1/2/3", "send_result": send_res}, status_code=200)

        # 3) Quantidade pendente
        if "pending_item" in sess and not sess.get("awaiting_choice"):
            m = re.search(r"\d+", text_body or "")
            if m:
                qty = int(m.group())
                item = sess.pop("pending_item")
                resumo = f"✅ Adicionado: {qty}× {item['sku']} — {item['nome']}"
                # (Se quiser carrinho multi-itens, aqui seria o lugar para acumular)
                send_res = send_whatsapp_message(to, resumo)
                return JSONResponse({"status": "processed", "echo": resumo, "send_result": send_res}, status_code=200)
            else:
                send_res = send_whatsapp_message(to, "Quantas unidades? (envie um número)")
                return JSONResponse({"status": "processed", "echo": "perguntando quantidade", "send_result": send_res}, status_code=200)

        # 4) Small talk simples
        if is_greeting(text_body):
            intro = ("Olá! 👋 Somos a BotChina (atendimento 24/7).\n"
                     "Para orçamento, diga o **SKU** ou descreva o produto (ex: 'carrinho de controle remoto amarelo 1:24') "
                     "e a **quantidade**. Posso sugerir opções 😉")
            send_res = send_whatsapp_message(to, intro)
            return JSONResponse({"status": "processed", "echo": intro, "send_result": send_res}, status_code=200)

        # 5) Pedido: extrair item e buscar no catálogo
        items = extract_items(text_body or "")
        item = items[0]  # MVP: tratamos o primeiro item
        cands = search_catalog(item["name"])

        if not cands:
            ask = "Não encontrei nada ainda. Pode detalhar um pouco? (ex: cor, modelo, marca)"
            send_res = send_whatsapp_message(to, ask)
            return JSONResponse({"status": "processed", "echo": ask, "send_result": send_res}, status_code=200)

        if len(cands) == 1:
            chosen = cands[0]
            sess["pending_item"] = {"sku": chosen.get("sku"), "nome": chosen.get("nome")}
            msg_out = (f"Encontrei: {chosen.get('sku')} — {chosen.get('nome')}.\n"
                       f"Quantas unidades deseja?")
            send_res = send_whatsapp_message(to, msg_out)
            return JSONResponse({"status": "processed", "echo": msg_out, "send_result": send_res}, status_code=200)

        # >=2 → desambiguação
        sess["awaiting_choice"] = {"q": item["name"], "cands": cands}
        show_candidates_text(to, item["name"], cands)
        return JSONResponse({"status": "processed", "echo": "asking choice", "send_result": {"status_code": 200}}, status_code=200)

    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=200)

# -----------------------------
# Rotas de debug
# -----------------------------
@app.get("/debug/token")
def debug_token():
    t = os.getenv("WABA_TOKEN", "")
    return {"len": len(t), "tail": t[-8:] if t else ""}

@app.get("/debug/cfg")
def debug_cfg():
    cfg = get_cfg()
    return {
        "graph_version": cfg["GRAPH_VERSION"],
        "phone_number_id": cfg["WABA_PHONE_NUMBER_ID"],
        "token_tail": cfg["WABA_TOKEN"][-8:] if cfg["WABA_TOKEN"] else "",
        "token_len": len(cfg["WABA_TOKEN"]),
        "catalog_url": CATALOG_URL,
        "catalog_loaded_rows": (0 if CATALOG is None else len(CATALOG)),
    }

@app.post("/debug/reload")
def debug_reload():
    load_catalog()
    return {"reloaded": True, "rows": 0 if CATALOG is None else len(CATALOG)}

@app.get("/")
def root():
    return {"ok": True}
