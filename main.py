import os
import re
import json
import time
import io
import requests
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv
from difflib import SequenceMatcher

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

HUMAN_SYSTEM_PROMPT = """
Você é o BotChina, atendente de loja de eletrônicos.
Fale em PT-BR, tom humano, curto e educado.
Converse naturalmente e só consulte o catálogo quando:
- o cliente pedir preço/estoque/sku/modelo/comprar
- o cliente pedir para procurar produto
Regras:
- Pergunte 1 coisa por vez; ofereça /cancel e opção de falar com atendente.
- Se ambíguo, pergunte “Quer que eu pesquise no catálogo?” antes de buscar.
- Nunca invente preço/estoque; só retorne se vier do catálogo.
- Em reclamações: peça nº do pedido, fotos e descrição objetiva.
"""

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
    msgs = [{"role": "system", "content": HUMAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_text}]
    out = llm_chat(msgs)
    return out or "Posso te ajudar. Me diga o que precisa 🙂"

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
# Catálogo (CSV via Google Sheets) com cache
# -----------------------------
SESSIONS: dict[str, dict] = {}  # estado simples por wa_id
CATALOG = None
CATALOG_URL = os.getenv("CATALOG_URL", "").strip()
CACHE = {"df": None, "ts": 0}
CACHE_TTL = 300  # 5 min

EXPECTED_COLS = ["sku","nome","sinonimos","descricao","cor","preco","moeda","lead_time","estoque","image_url"]

def normalize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip())
    lower_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower_map)
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
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("").str.strip()
    # colunas normalizadas auxiliares
    df["sku_norm"] = df["sku"].str.upper().str.strip()
    df["name_norm"] = (df["nome"] + " " + df["sinonimos"] + " " + df["descricao"]).str.lower()
    return df[EXPECTED_COLS + ["sku_norm","name_norm"]]

def load_catalog(force: bool=False) -> pd.DataFrame | None:
    """Carrega o catálogo do CSV público (CATALOG_URL) com cache simples."""
    if not CATALOG_URL:
        CACHE.update({"df": None, "ts": 0})
        return None
    now = time.time()
    if not force and CACHE["df"] is not None and (now - CACHE["ts"]) < CACHE_TTL:
        return CACHE["df"]
    try:
        resp = requests.get(CATALOG_URL, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text)).fillna("")
        df = normalize_catalog(df)
        CACHE.update({"df": df, "ts": now})
        print(f"[catalog] loaded {len(df)} rows from {CATALOG_URL}")
        return df
    except Exception as e:
        print(f"[catalog] load error: {e}")
        return CACHE["df"]

# Busca robusta
SCALE_PAT = re.compile(r"\b1[:/]\d{1,3}\b", re.IGNORECASE)             # 1:24, 1/10 etc.
SKU_PAT   = re.compile(r"\b[A-Z]{2,6}-\d{2,6}\b", re.IGNORECASE)       # RC-124, KM-9163 etc.

def clean_query(q: str) -> str:
    t = q or ""
    t = SCALE_PAT.sub(" ", t)  # remove escalas para não confundir com SKU
    return re.sub(r"\s+", " ", t).strip()

def find_by_sku(code: str):
    df = load_catalog()
    if df is None: return None
    row = df[df["sku_norm"] == code.upper().strip()]
    return (row.iloc[0].to_dict() if not row.empty else None)

def find_by_name(query: str, topn: int = 3):
    df = load_catalog()
    if df is None: return []
    q = clean_query(query).lower()
    def score_row(s: str) -> float:
        return SequenceMatcher(a=q, b=s).ratio()
    sc = df.copy()
    sc["score"] = sc["name_norm"].apply(score_row)
    sc = sc.sort_values("score", ascending=False)
    rows = sc.head(topn)
    return [r.to_dict() for _, r in rows.iterrows() if r["score"] > 0.35]

# -----------------------------
# Intenção e helpers de conversa
# -----------------------------
def route_command(text: str) -> str | None:
    t = (text or "").strip()
    if not t.startswith("/"):
        return None
    cmd, *rest = t.split(maxsplit=1)
    arg = rest[0] if rest else ""
    if cmd.lower() == "/help":
        return ("👋 Posso ajudar com:\n"
                "• Descreva o produto (ex: carrinho RC vermelho 1:24)\n"
                "• /sku <código>\n"
                "• /humano para falar com atendente\n"
                "• /cancel para voltar ao menu")
    if cmd.lower() == "/humano":
        return "✅ Vou acionar um atendente humano. Envie nome/empresa e melhor horário."
    if cmd.lower() == "/sku":
        if arg and SKU_PAT.fullmatch(arg.strip()):
            return f"__SKU_LOOKUP__{arg.strip()}"
        return f"SKU inválido. Exemplo: /sku RC-124"
    if cmd.lower() == "/cancel":
        return "__CANCEL__"
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

CATALOG_TRIGGERS = {"preço","valor","tem","estoque","disponível","sku","modelo","comprar","quantas","unidades","rc","impressora","carrinho","camera","barbeador","fonte","etiqueta"}

def route_intent(user_text: str):
    t = (user_text or "").lower()
    if t.strip() in {"/cancel","cancelar","menu"}:
        return {"intent":"menu"}
    if "atendente" in t or "humano" in t:
        return {"intent":"handoff"}
    # SKU explícito
    m = SKU_PAT.search(user_text or "")
    if m:
        return {"intent":"catalog_by_sku", "code": m.group(0)}
    # “1:24” não dispara busca
    if SCALE_PAT.search(t):
        # ainda pode ser busca por nome; tratamos como possível catálogo se houver termos
        pass
    # gatilhos de produto
    hits = sum(1 for w in CATALOG_TRIGGERS if w in t)
    if hits >= 2:
        return {"intent":"maybe_catalog"}  # pedir permissão
    return {"intent":"chitchat"}

def parse_leading_qty(text: str) -> int | None:
    # extrai quantidade no começo da frase, mas NÃO quando é escala "1:24"
    m = re.match(r"^\s*(\d{1,3})(?!\s*[:/])(?:\s*(?:x|un|unid|unidade|unidades)\b)?", text.lower())
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None

def format_candidate_line(i, c):
    preco = f"{c.get('moeda','')}{c.get('preco','')}".strip()
    estoque = c.get('estoque','?')
    nome = (c.get('nome','') or '').strip()
    sku = (c.get('sku') or '—').strip()
    return f"{i}) {sku} — {nome} • {preco} • estoque: {estoque}"

def show_candidates_text(to: str, query: str, cands: list):
    lines = [f"Encontrei opções para “{query}”:"] + \
            [format_candidate_line(i,c) for i,c in enumerate(cands, start=1)] + \
            ["Qual você prefere? Responda 1, 2 ou 3.\nSe quiser cancelar: /cancel"]
    send_whatsapp_message(to, "\n".join(lines))

def menu_msg():
    return ("Olá! 👋 Sou o BotChina.\n"
            "Posso: (1) procurar produto, (2) falar com atendente, (3) ver catálogo.\n"
            "Descreva o que procura e, se quiser sair de qualquer etapa, use /cancel.")

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

        # 0) Timeout simples: limpa contexto após 8 min sem interação
        if "ts" in sess and (time.time() - sess["ts"]) > 8*60:
            sess.clear()
        sess["ts"] = time.time()

        # 1) Comandos com "/"
        cmd_reply = route_command(text_body)
        if cmd_reply:
            if cmd_reply == "__CANCEL__":
                sess.clear()
                intro = "Menu reiniciado ✅\n" + menu_msg()
                send_res = send_whatsapp_message(to, intro)
                return JSONResponse({"status":"processed","echo":intro,"send_result":send_res}, status_code=200)
            if cmd_reply.startswith("__SKU_LOOKUP__"):
                code = cmd_reply.replace("__SKU_LOOKUP__","")
                item = find_by_sku(code)
                if not item:
                    send_res = send_whatsapp_message(to, f"Não encontrei o SKU {code}. Quer que eu procure por nome?")
                    return JSONResponse({"status":"processed","echo":"sku not found","send_result":send_res}, status_code=200)
                sess["pending_item"] = {"sku": item.get("sku"), "nome": item.get("nome")}
                msg_out = (f"Encontrei: {item.get('sku')} — {item.get('nome')}.\n"
                           f"Quantas unidades deseja? (envie só o número) • /cancel")
                send_res = send_whatsapp_message(to, msg_out)
                return JSONResponse({"status":"processed","echo":msg_out,"send_result":send_res}, status_code=200)
            send_res = send_whatsapp_message(to, cmd_reply)
            return JSONResponse({"status": "processed", "echo": cmd_reply, "send_result": send_res}, status_code=200)

        # 1.1) Cancelar por linguagem natural
        if is_cancel(text_body):
            sess.clear()
            intro = "Menu reiniciado ✅\n" + menu_msg()
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
                               f"Quantas unidades deseja? (envie só o número) • /cancel")
                    send_res = send_whatsapp_message(to, msg_out)
                    return JSONResponse({"status":"processed","echo":msg_out,"send_result":send_res}, status_code=200)
            send_res = send_whatsapp_message(to, "Por favor, responda 1, 2 ou 3. • /cancel")
            return JSONResponse({"status":"processed","echo":"aguardando 1/2/3","send_result":send_res}, status_code=200)

        # 3) Quantidade pendente
        if "pending_item" in sess and not sess.get("awaiting_choice"):
            qty = parse_leading_qty(text_body or "")
            if qty and qty > 0:
                item = sess.pop("pending_item")
                resumo = f"✅ Adicionado: {qty}× {item['sku']} — {item['nome']}"
                send_res = send_whatsapp_message(to, resumo)
                return JSONResponse({"status":"processed","echo":resumo,"send_result":send_res}, status_code=200)
            else:
                send_res = send_whatsapp_message(to, "Quantas unidades? (envie somente o número) • /cancel")
                return JSONResponse({"status":"processed","echo":"perguntando quantidade","send_result":send_res}, status_code=200)

        # 4) Confirmação para pesquisar no catálogo
        if sess.get("confirm_catalog"):
            answer = (text_body or "").strip().lower()
            if any(a in answer for a in ["sim","pode","procura","busca","ok","manda ver","quero"]):
                query = sess.pop("confirm_catalog")
                # separa quantidade (se vier no começo) para usar depois
                qty = parse_leading_qty(query) or parse_leading_qty(text_body) or None
                q_clean = clean_query(query)
                cands = find_by_name(q_clean)
                if not cands:
                    send_res = send_whatsapp_message(to, "Não encontrei nada ainda. Pode me dizer marca/cor/modelo?")
                    return JSONResponse({"status":"processed","echo":"no results","send_result":send_res}, status_code=200)
                if len(cands) == 1:
                    chosen = cands[0]
                    sess["pending_item"] = {"sku": chosen.get("sku"), "nome": chosen.get("nome")}
                    if qty:
                        # se já veio quantidade, finaliza direto
                        item = sess.pop("pending_item")
                        resumo = f"✅ Adicionado: {qty}× {item['sku']} — {item['nome']}"
                        send_res = send_whatsapp_message(to, resumo)
                        return JSONResponse({"status":"processed","echo":resumo,"send_result":send_res}, status_code=200)
                    msg_out = (f"Encontrei: {chosen.get('sku')} — {chosen.get('nome')}.\n"
                               f"Quantas unidades deseja? (apenas número) • /cancel")
                    send_res = send_whatsapp_message(to, msg_out)
                    return JSONResponse({"status":"processed","echo":msg_out,"send_result":send_res}, status_code=200)
                sess["awaiting_choice"] = {"q": q_clean, "cands": cands}
                show_candidates_text(to, q_clean, cands)
                return JSONResponse({"status":"processed","echo":"asking choice","send_result":{"status_code":200}}, status_code=200)
            elif any(a in answer for a in ["não","nao","deixa","depois","cancel"]):
                sess.pop("confirm_catalog", None)
                msg_out = "Beleza! Se preferir eu pesquiso depois. Quer falar com atendente?"
                send_res = send_whatsapp_message(to, msg_out)
                return JSONResponse({"status":"processed","echo":msg_out,"send_result":send_res}, status_code=200)
            else:
                send_res = send_whatsapp_message(to, "Quer que eu pesquise no catálogo? (sim/não) • /cancel")
                return JSONResponse({"status":"processed","echo":"confirm prompt","send_result":send_res}, status_code=200)

        # 5) Conversa livre humanizada (se não estiver em fluxo de escolha/quantidade)
        intent = route_intent(text_body)

        if intent["intent"] == "menu":
            sess.clear()
            intro = "Menu reiniciado ✅\n" + menu_msg()
            send_res = send_whatsapp_message(to, intro)
            return JSONResponse({"status":"processed","echo":intro,"send_result":send_res}, status_code=200)

        if intent["intent"] == "handoff":
            msg_out = "Beleza! Vou acionar um atendente agora 👩‍💻"
            send_res = send_whatsapp_message(to, msg_out)
            return JSONResponse({"status":"processed","echo":msg_out,"send_result":send_res}, status_code=200)

        if intent["intent"] == "catalog_by_sku":
            code = intent["code"]
            item = find_by_sku(code)
            if not item:
                send_res = send_whatsapp_message(to, f"Não encontrei o SKU {code}. Quer que eu procure por nome?")
                return JSONResponse({"status":"processed","echo":"sku not found","send_result":send_res}, status_code=200)
            sess["pending_item"] = {"sku": item.get("sku"), "nome": item.get("nome")}
            msg_out = (f"Encontrei: {item.get('sku')} — {item.get('nome')}.\n"
                       f"Quantas unidades deseja? (apenas número) • /cancel")
            send_res = send_whatsapp_message(to, msg_out)
            return JSONResponse({"status":"processed","echo":msg_out,"send_result":send_res}, status_code=200)

        if intent["intent"] == "maybe_catalog":
            # pede permissão antes de consultar
            sess["confirm_catalog"] = text_body
            msg_out = "Posso pesquisar no catálogo pra você? 🔎 (sim/não) • /cancel"
            send_res = send_whatsapp_message(to, msg_out)
            return JSONResponse({"status":"processed","echo":"ask to confirm search","send_result":send_res}, status_code=200)

        # Chitchat / reclamações
        if looks_like_free_chat(text_body):
            reply = llm_reply(text_body)
            send_res = send_whatsapp_message(to, reply)
            return JSONResponse({"status":"processed","echo":reply,"send_result":send_res}, status_code=200)

        # Off-topic
        off = detect_offtopic(text_body)
        if off:
            msg_out = (f"Entendi “{off}”, mas nós trabalhamos com **eletrônicos** 🙂 "
                       f"Me diga o eletrônico e a quantidade.")
            send_res = send_whatsapp_message(to, msg_out)
            return JSONResponse({"status":"processed","echo":msg_out,"send_result":send_res}, status_code=200)

        # Saudação simples
        if is_greeting(text_body):
            intro = menu_msg()
            send_res = send_whatsapp_message(to, intro)
            return JSONResponse({"status":"processed","echo":intro,"send_result":send_res}, status_code=200)

        # 6) Fallback: se o texto parece descrição de produto, ofereça buscar
        if any(w in (text_body or "").lower() for w in CATALOG_TRIGGERS):
            sess["confirm_catalog"] = text_body
            msg_out = "Quer que eu procure esse produto no catálogo? (sim/não) • /cancel"
            send_res = send_whatsapp_message(to, msg_out)
            return JSONResponse({"status":"processed","echo":"offer search","send_result":send_res}, status_code=200)

        # 7) Último recurso: resposta humanizada
        reply = llm_reply(text_body)
        send_res = send_whatsapp_message(to, reply)
        return JSONResponse({"status":"processed","echo":reply,"send_result":send_res}, status_code=200)

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
    df = CACHE["df"] if CACHE["df"] is not None else None
    return {
        "graph_version": os.getenv("GRAPH_VERSION", "v23.0"),
        "phone_number_id": os.getenv("WABA_PHONE_NUMBER_ID", ""),
        "token_tail": (os.getenv("WABA_TOKEN","")[-8:] if os.getenv("WABA_TOKEN","") else ""),
        "token_len": len(os.getenv("WABA_TOKEN","")),
        "catalog_url": CATALOG_URL,
        "catalog_loaded_rows": (0 if df is None else len(df)),
        "llm_base": LLM_API_BASE,
        "llm_model": LLM_MODEL,
        "llm_enabled": bool(LLM_API_BASE and LLM_API_KEY),
    }

@app.post("/debug/reload")
def debug_reload():
    df = load_catalog(force=True)
    return {"reloaded": True, "rows": 0 if df is None else len(df)}

@app.get("/debug/llm")
def debug_llm(q: str = "teste de atendimento humano"):
    return {"sample": llm_reply(q)}

@app.get("/")
def root():
    return {"ok": True}
