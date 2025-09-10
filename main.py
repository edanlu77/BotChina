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
# LLM (Groq) para conversa livre humanizada
# -----------------------------
LLM_API_BASE = os.getenv("LLM_API_BASE", "").rstrip("/")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

def llm_chat(messages: list[dict]) -> str:
    """Chama endpoint OpenAI-compatível /v1/chat/completions."""
    if not (LLM_API_BASE and LLM_API_KEY):
        return ""
    url = f"{LLM_API_BASE}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": LLM_MODEL, "temperature": 0.4, "messages": messages}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("[llm error]", e)
        return ""

def llm_reply(user_text: str) -> str:
    system = (
        "Você é atendente da BotChina (eletrônicos). Seja educado, calmo e objetivo. "
        "Regras: 1) Se houver irritação/ofensa, mantenha a calma, peça respeito e ofereça falar com humano. "
        "2) Em reclamações (não gostei, extraviado, quebrado, faltou), peça nº do pedido, fotos e descrição objetiva. "
        "3) Não prometa reembolso/troca automática; diga que abrirá ticket e verificará a política. "
        "4) Se pedirem itens fora do escopo (alimentos/roupas), explique que vendemos eletrônicos e dê exemplos. "
        "5) Faça perguntas de uma em uma. Respostas curtas, claras, em PT-BR."
    )
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
    out = llm_chat(msgs)
    return out or "Posso te ajudar. Me diga o nº do pedido e o que ocorreu (produto, problema, fotos)."

def looks_like_free_chat(t: str) -> bool:
    t = (t or "").lower()
    gatilhos = [
        "não gostei", "nao gostei", "reclama", "reclamação", "quebrado", "rachado",
        "faltou", "faltando", "extravi", "atrasou", "demora", "rasgado",
        "onde vocês são", "quem são vocês", "atendente", "humano",
        "oi", "tudo bem", "bom dia", "boa tarde", "boa noite",
    ]
    return any(g in t for g in gatilhos)

# -----------------------------
# Catálogo (CSV via Google Sheets)
# -----------------------------
SESSIONS: dict[str, dict] = {}  # estado simples por wa_id
CATALOG = None
CATALOG_URL = os.getenv("CATALOG_URL", "").strip()

EXPECTED_COLS = ["sku","nome","sinonimos","descricao","cor","preco","moeda","lead_time","estoque","image_url"]

def normalize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    # tira espaços e padroniza para minúsculas
    df = df.rename(columns=lambda c: str(c).strip())
    lower_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower_map)
    # sinônimos comuns
    synonyms = {
        "sku": "sku", "código": "sku", "codigo": "sku",
        "nome": "nome", "produto": "nome", "nome do produto": "nome",
        "sinonimos": "sinonimos", "sinônimos": "sinonimos", "sinonimo": "sinonimos",
        "descricao": "descricao", "descrição": "descricao", "description": "descricao",
        "cor": "cor", "color": "cor",
        "preco": "preco", "preço": "preco", "price": "preco", "valor": "preco",
        "moeda": "moeda", "currency": "moeda",
        "lead_time": "lead_time", "lead time": "lead_time", "prazo": "lead_time",
        "estoque": "estoque", "stock": "estoque", "qtd": "estoque",
        "image_url": "image_url", "imagem": "image_url", "foto": "image_url", "url_imagem": "image_url",
    }
    df = df.rename(columns={c: synonyms.get(c, c) for c in df.columns})
    # garante colunas e limpa
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("").str.strip()
    return df[EXPECTED_COLS]

def load_catalog() -> None:
    """Carrega o catálogo do CSV público (CATALOG_URL)."""
    global CATALOG
    if not CATALOG_URL:
        CATALOG = None
        return
    try:
        df = pd.read_csv(CATALOG_URL).fillna("")
        df = normalize_catalog(df)
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
                "• Descreva o produto e a **quantidade** (ex: '3 amoladores de faca')\n"
                "• /sku <código>\n"
                "• /humano para falar com atendente\n"
                "• /cancel para limpar e voltar ao menu")
    if cmd.lower() == "/humano":
        return "✅ Vou acionar um atendente humano. Envie nome/empresa e melhor horário."
    if cmd.lower() == "/sku":
        return f"Digite o código do SKU após /sku. Você enviou: {arg or '(faltou o código)'}"
    if cmd.lower() == "/cancel":
        return "__CANCEL__"  # sinal interno
    return "🙃 Comando não reconhecido. Digite /help."

def is_greeting(t: str) -> bool:
    t = (t or "").strip().lower()
    return any(g in t for g in ["oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "hello", "hi", "hey"])

def is_cancel(t: str) -> bool:
    t = (t or "").strip().lower()
    gatilhos = ["cancel", "/cancel", "cancelar", "voltar", "menu", "início", "inicio", "voltar ao menu", "menu inicial"]
    return any(g in t for g in gatilhos)

OFFTOPIC_WORDS = {
    "alimentos": {"açaí", "acai", "pizza", "hamburguer", "hambúrguer", "sorvete", "lanche", "suco", "café", "cafe"},
    "roupas": {"camiseta", "calça", "sapato", "tênis", "tenis", "jaqueta", "vestido"},
    "serviços": {"passagem", "hotel", "viagem", "frete internacional"},
}
def detect_offtopic(text: str) -> str | None:
    t = (text or "").lower()
    for _, palavras in OFFTOPIC_WORDS.items():
        for p in palavras:
            if p in t:
                return p
    return None

def extract_items(text: str):
    """Extrai pares quantidade + descrição. Ex.: '3 carrinhos, 5 impressoras'."""
    pats = re.findall(r"(\d+)\s*(x|X)?\s*([a-zA-Z0-9\u00C0-\u017F\s\-_/]+?)(?:,| e |;|$)", text or "")
    items = [{"qty": int(q), "name": n.strip()} for q, _, n in pats if n.strip()]
    return items or [{"qty": 1, "name": (text or '').strip()}]

def format_candidate_line(i, c):
    preco = f"{c.get('moeda','')}{c.get('preco','')}".strip()
    estoque = c.get('estoque','?')
    nome = (c.get('nome','') or '').strip()
    sku = (c.get('sku') or '—').strip()
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
            if cmd_reply == "__CANCEL__":
                sess.clear()
                intro = ("Menu reiniciado ✅\n"
                         "Somos a BotChina (eletrônicos). Diga o **SKU** ou descreva o produto (ex: 'impressora térmica 80mm' "
                         "ou 'carrinho RC amarelo 1:24') e a **quantidade**. Digite /help para ver opções.")
                send_res = send_whatsapp_message(to, intro)
                return JSONResponse({"status":"processed","echo":intro,"send_result":send_res}, status_code=200)
            send_res = send_whatsapp_message(to, cmd_reply)
            return JSONResponse({"status": "processed", "echo": cmd_reply, "send_result": send_res}, status_code=200)

        # 1.1) Cancelar por linguagem natural
        if is_cancel(text_body):
            sess.clear()
            intro = ("Menu reiniciado ✅\n"
                     "Somos a BotChina (eletrônicos). Diga o **SKU** ou descreva o produto (ex: 'impressora térmica 80mm' "
                     "ou 'carrinho RC amarelo 1:24') e a **quantidade**. Digite /help para ver opções.")
            send_res = send_whatsapp_message(to, intro)
            return JSONResponse({"status":"processed","echo":intro,"send_result":send_res}, status_code=200)

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
                send_res = send_whatsapp_message(to, resumo)
                return JSONResponse({"status": "processed", "echo": resumo, "send_result": send_res}, status_code=200)
            else:
                send_res = send_whatsapp_message(to, "Quantas unidades? (envie um número)")
                return JSONResponse({"status": "processed", "echo": "perguntando quantidade", "send_result": send_res}, status_code=200)

        # 4) Conversa livre humanizada via LLM (se não estiver em fluxo de escolha/quantidade)
        if looks_like_free_chat(text_body):
            reply = llm_reply(text_body)
            if reply:
                send_res = send_whatsapp_message(to, reply)
                return JSONResponse({"status":"processed","echo":reply,"send_result":send_res}, status_code=200)

        # 4.5) Off-topic (fora do nosso domínio)
        off = detect_offtopic(text_body)
        if off:
            msg_out = (f"Entendi “{off}”, mas nós trabalhamos com **eletrônicos** "
                       f"(ex.: amoladores de faca, impressoras térmicas 58/80mm, carrinhos RC, etiquetas). "
                       f"Me diga o eletrônico e a quantidade 🙂")
            send_res = send_whatsapp_message(to, msg_out)
            return JSONResponse({"status":"processed","echo":msg_out,"send_result":send_res}, status_code=200)

        # 5) Small talk simples (saudação)
        if is_greeting(text_body):
            intro = ("Olá! 👋 Somos a BotChina (atendimento 24/7).\n"
                     "Para orçamento, diga o **SKU** ou descreva o produto (ex: 'carrinho de controle remoto amarelo 1:24') "
                     "e a **quantidade**. Posso sugerir opções 😉")
            send_res = send_whatsapp_message(to, intro)
            return JSONResponse({"status": "processed", "echo": intro, "send_result": send_res}, status_code=200)

        # 6) Pedido: extrair item e buscar no catálogo
        items = extract_items(text_body or "")
        item = items[0]  # MVP: tratamos o primeiro item
        cands = search_catalog(item["name"])

        if not cands:
            # Fallback humanizado: LLM tenta ajudar
            reply = llm_reply(text_body)
            send_res = send_whatsapp_message(to, reply or "Ainda não encontrei. Pode detalhar cor/modelo/marca ou informar o SKU?")
            return JSONResponse({"status": "processed", "echo": reply, "send_result": send_res}, status_code=200)

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
        "llm_base": LLM_API_BASE,
        "llm_model": LLM_MODEL,
        "llm_enabled": bool(LLM_API_BASE and LLM_API_KEY),
    }

@app.post("/debug/reload")
def debug_reload():
    load_catalog()
    return {"reloaded": True, "rows": 0 if CATALOG is None else len(CATALOG)}

@app.get("/debug/llm")
def debug_llm(q: str = "teste de atendimento humano"):
    return {"sample": llm_reply(q)}

@app.get("/")
def root():
    return {"ok": True}
