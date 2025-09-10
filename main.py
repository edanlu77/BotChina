# main.py — LLM 100% humanizado; consulta o Sheets SOMENTE quando solicitado
# Comandos opcionais: /sku <código>, /buscar <termos>, /help, /cancel

import os, re, io, json, time, requests, pandas as pd
from difflib import SequenceMatcher
from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# Boot
# ------------------------------------------------------------------------------
load_dotenv(override=True)
app = FastAPI()

# ------------------------------------------------------------------------------
# WhatsApp Webhook Verify (Meta)
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# WhatsApp Send
# ------------------------------------------------------------------------------
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
    payload = {"messaging_product":"whatsapp","to":to,"type":"text","text":{"preview_url":False,"body":text}}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    try: data = r.json()
    except: data = {"raw": r.text}
    return {"status_code": r.status_code, "data": data, "url": url}

# ------------------------------------------------------------------------------
# LLM (Groq/OpenAI-compatível) — conversa 100% humanizada
# ------------------------------------------------------------------------------
LLM_API_BASE = os.getenv("LLM_API_BASE","").rstrip("/")
LLM_API_KEY  = os.getenv("LLM_API_KEY","")
LLM_MODEL    = os.getenv("LLM_MODEL","llama-3.3-70b-versatile")

HUMAN_SYSTEM_PROMPT = """
Você é o BotChina (eletrônicos). Seja 100% humano, gentil e objetivo.
Prioridade: CONVERSAR naturalmente. Use 1–2 frases, sem textão.
Só consulte o catálogo quando o cliente solicitar catálogo/produtos/orçamento/preço/valor/quanto/estoque/disponível/SKU/código
ou pedir para procurar (ex.: “busca”, “pesquisa”, “procurar”).
Se pedir orçamento/preço/estoque sem especificar, você mesmo pergunta o item e então aciona a consulta.
Em reclamações: peça nº do pedido + fotos + descrição curta. Em despedidas, cumprimente pelo período (bom dia/tarde/noite).
"""

def llm_chat(messages: list[dict]) -> str:
    if not (LLM_API_BASE and LLM_API_KEY): return ""
    url = f"{LLM_API_BASE}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": LLM_MODEL, "temperature": 0.4, "max_tokens": 180, "messages": messages}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("[llm error]", e); return ""

def llm_reply(user_text: str) -> str:
    msgs = [{"role":"system","content":HUMAN_SYSTEM_PROMPT},{"role":"user","content":user_text}]
    out = llm_chat(msgs)
    return out or "Posso ajudar — me diga o que precisa. 🙂"

# ------------------------------------------------------------------------------
# Catálogo (Sheets CSV) — consulta apenas sob intenção de catálogo
# ------------------------------------------------------------------------------
CACHE = {"df": None, "ts": 0}
CACHE_TTL = 300
LAST_CATALOG_ERR = ""

def ensure_csv_url(url: str) -> str:
    if not url: return url
    u = url.strip()
    if "output=csv" in u or "format=csv" in u: return u
    if "/edit" in u: return u.split("/edit")[0] + "/export?format=csv"
    return u

def maybe_fix_mojibake(s: str) -> str:
    if not isinstance(s, str): return s
    if "Ã" in s or "�" in s:
        try: return s.encode("latin1").decode("utf-8")
        except: return s
    return s

EXPECTED_COLS = ["sku","nome","sinonimos","descricao","cor","preco","moeda","lead_time","estoque","image_url"]

def normalize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip().lower())
    for col in EXPECTED_COLS:
        if col not in df.columns: df[col] = ""
        df[col] = df[col].astype(str).fillna("").apply(maybe_fix_mojibake).str.strip()
    df["sku_norm"]  = df["sku"].str.upper().str.strip()
    df["name_norm"] = (df["nome"] + " " + df["sinonimos"] + " " + df["descricao"]).str.lower()
    return df[EXPECTED_COLS + ["sku_norm","name_norm"]]

def load_catalog(force: bool=False) -> pd.DataFrame | None:
    global LAST_CATALOG_ERR
    url = ensure_csv_url(os.getenv("CATALOG_URL","").strip())
    if not url:
        LAST_CATALOG_ERR = "CATALOG_URL vazio"; CACHE.update({"df":None,"ts":0}); return None
    now = time.time()
    if not force and CACHE["df"] is not None and (now - CACHE["ts"]) < CACHE_TTL:
        return CACHE["df"]
    try:
        resp = requests.get(url, timeout=25, allow_redirects=True)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text)).fillna("")
        df = normalize_catalog(df)
        CACHE.update({"df": df, "ts": now}); LAST_CATALOG_ERR = ""
        print(f"[catalog] loaded {len(df)} rows")
        return df
    except Exception as e:
        LAST_CATALOG_ERR = f"{type(e).__name__}: {e}"
        print(f"[catalog] load error: {LAST_CATALOG_ERR}")
        return CACHE["df"]

def find_by_sku(code: str):
    df = load_catalog()
    if df is None: return None
    row = df[df["sku_norm"] == str(code).upper().strip()]
    return (row.iloc[0].to_dict() if not row.empty else None)

def find_by_name(query: str, topn: int = 3):
    df = load_catalog()
    if df is None: return []
    q = (query or "").strip().lower()
    sc = df.copy()
    sc["score"] = sc["name_norm"].apply(lambda s: SequenceMatcher(a=q, b=s).ratio())
    sc = sc.sort_values("score", ascending=False)
    rows = sc.head(topn)
    # também aceita substring direta caso a similaridade não seja alta
    results = [r.to_dict() for _, r in rows.iterrows() if r["score"] > 0.30]
    if not results and q:
        subs = df[df["name_norm"].str.contains(re.escape(q), na=False)]
        for _, r in subs.head(topn).iterrows():
            results.append(r.to_dict())
    return results[:topn]

def fmt_price(preco: str, moeda: str) -> str:
    if (moeda or "").upper() == "BRL":
        try: return "R$ " + f"{float(str(preco).replace(',','.')):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except: return "R$ " + str(preco).replace(".", ",")
    return f"{moeda} {preco}".strip()

# ------------------------------------------------------------------------------
# Intenção de catálogo (sempre consulta o Sheets) + comandos
# ------------------------------------------------------------------------------
SKU_REGEX   = re.compile(r"\b[A-Z]{2,6}-\d{2,6}\b")
SCALE_REGEX = re.compile(r"\b1[:/]\d{1,3}\b", re.IGNORECASE)

CATALOG_TRIGGERS = {
    "catálogo","catalogo","produtos","produto","lista",
    "orçamento","orcamento","cotação","cotacao",
    "preço","preco","valor","custa","quanto",
    "estoque","disponível","disponivel","sku","código","codigo",
    "tabela","price list","price-list","buscar","pesquisar","procurar","procura","pesquisa"
}

STOPWORDS_PT = {"de","da","do","das","dos","para","pra","por","o","a","os","as","um","uma","no","na","em","e","ou","que","com"}

def is_catalog_intent(t: str) -> bool:
    if not t: return False
    u = t.lower()
    if SKU_REGEX.search(t): 
        return True
    if any(w in u for w in CATALOG_TRIGGERS):
        return True
    return False

def extract_keywords(t: str) -> list[str]:
    u = SCALE_REGEX.sub(" ", (t or "").lower())
    toks = re.findall(r"[a-z0-9\u00C0-\u017F]{2,}", u)
    return [w for w in toks if w not in STOPWORDS_PT]

def reply_catalog_overview(to: str, df: pd.DataFrame, limit: int = 6):
    rows = [] if df is None else df.head(limit).to_dict(orient="records")
    if not rows:
        return send_whatsapp_message(to, "Não consegui acessar o catálogo agora. Pode tentar novamente?")
    lines = ["Aqui vão algumas opções do nosso catálogo:"]
    for r in rows:
        preco = fmt_price(r.get("preco",""), r.get("moeda",""))
        lines.append(f"- {r.get('sku','—')} — {r.get('nome','')} • {preco} • estoque: {r.get('estoque','—')}")
    lines.append("Se quiser, me diga o **SKU** ou descreva o produto que eu detalho 😉")
    return send_whatsapp_message(to, "\n".join(lines))

def reply_catalog_search(to: str, query: str):
    m = SKU_REGEX.search(query or "")
    if m:
        item = find_by_sku(m.group(0))
        if item:
            preco = fmt_price(item.get("preco",""), item.get("moeda",""))
            out = (f"{item.get('sku','—')} — {item.get('nome','')}\n"
                   f"Preço: {preco} | Estoque: {item.get('estoque','—')} | Prazo: {item.get('lead_time','—')}")
            return send_whatsapp_message(to, out)
    kws = extract_keywords(query or "")
    q = " ".join(kws)
    cands = find_by_name(q) if q else []
    if not cands:
        df = load_catalog()
        return reply_catalog_overview(to, df if df is not None else pd.DataFrame())
    lines = []
    for c in cands[:3]:
        preco = fmt_price(c.get("preco",""), c.get("moeda",""))
        lines.append(f"{c.get('sku','—')} — {c.get('nome','')} • {preco} • estoque: {c.get('estoque','—')}")
    out = "Encontrei estas opções:\n" + "\n".join(lines) + "\nPode me dizer qual te interessa?"
    return send_whatsapp_message(to, out)

def route_command(text: str) -> tuple[str,str] | None:
    t = (text or "").strip()
    if not t.startswith("/"): return None
    cmd, *rest = t.split(maxsplit=1)
    arg = rest[0].strip() if rest else ""
    cmd = cmd.lower()
    if cmd == "/sku":     return ("sku", arg)
    if cmd == "/buscar":  return ("buscar", arg)
    if cmd == "/help":    return ("help", "")
    if cmd == "/cancel":  return ("cancel","")
    return ("unknown","")

# ------------------------------------------------------------------------------
# Webhook principal — LLM total; Sheets apenas quando a intenção pedir
# ------------------------------------------------------------------------------
@app.post("/webhook")
async def receive_webhook(request: Request):
    body = await request.json()
    try:
        entry = body.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return JSONResponse({"status":"ignored"}, status_code=200)

        msg = messages[0]
        wa_from = msg.get("from")
        msg_type = msg.get("type")
        text_body = msg.get("text", {}).get("body", "") if msg_type == "text" else ""
        to = f"+{wa_from}"

        # 1) Comandos explícitos
        routed = route_command(text_body)
        if routed:
            kind, arg = routed
            if kind == "help":
                out = ("Pode falar normal comigo 😊\n"
                       "Para consultar o catálogo: /sku RC-124 ou /buscar impressora 80mm\n"
                       "Para cancelar: /cancel")
                send_whatsapp_message(to, out)
                return JSONResponse({"status":"processed","echo":out}, status_code=200)

            if kind == "cancel":
                out = "Prontinho! Limpei o contexto. Se quiser, posso consultar algo do catálogo — é só mandar /sku ou /buscar."
                send_whatsapp_message(to, out)
                return JSONResponse({"status":"processed","echo":out}, status_code=200)

            if kind == "sku":
                if not arg:
                    out = "Me envie assim: /sku RC-124"
                    send_whatsapp_message(to, out)
                    return JSONResponse({"status":"processed","echo":out}, status_code=200)
                item = find_by_sku(arg)
                if not item:
                    out = f"Não encontrei o SKU {arg}. Confere o código pra mim?"
                    send_whatsapp_message(to, out)
                    return JSONResponse({"status":"processed","echo":out}, status_code=200)
                preco = fmt_price(item.get("preco",""), item.get("moeda",""))
                out = (f"{item.get('sku','—')} — {item.get('nome','')}\n"
                       f"Preço: {preco} | Estoque: {item.get('estoque','—')} | Prazo: {item.get('lead_time','—')}")
                send_whatsapp_message(to, out)
                return JSONResponse({"status":"processed","echo":out}, status_code=200)

            if kind == "buscar":
                if not arg:
                    out = "Me diga o que buscar, ex.: /buscar impressora térmica 80mm"
                    send_whatsapp_message(to, out)
                    return JSONResponse({"status":"processed","echo":out}, status_code=200)
                send_res = reply_catalog_search(to, arg)
                return JSONResponse({"status":"processed","echo":"catalog_search_cmd"}, status_code=200)

            out = "Não reconheci esse comando. Use /help 😉"
            send_whatsapp_message(to, out)
            return JSONResponse({"status":"processed","echo":out}, status_code=200)

        # 2) Intenção explícita de catálogo/produtos/orçamento → SEMPRE consulta o Sheets
        if is_catalog_intent(text_body):
            df = load_catalog()
            kws = extract_keywords(text_body)
            if not kws and df is not None:
                reply_catalog_overview(to, df)
                return JSONResponse({"status":"processed","echo":"catalog_overview"}, status_code=200)
            reply_catalog_search(to, text_body)
            return JSONResponse({"status":"processed","echo":"catalog_search"}, status_code=200)

        # 3) Caso contrário: conversa 100% LLM
        reply = llm_reply(text_body)
        send_whatsapp_message(to, reply)
        return JSONResponse({"status":"processed","echo":reply}, status_code=200)

    except Exception as e:
        return JSONResponse({"status":"error","detail":str(e)}, status_code=200)

# ------------------------------------------------------------------------------
# Debug leve
# ------------------------------------------------------------------------------
@app.get("/debug/catalog")
def debug_catalog(n: int = 5):
    df = load_catalog()
    rows = 0 if df is None else len(df)
    cols = [] if df is None else list(df.columns)
    sample = [] if df is None else df.head(n).to_dict(orient="records")
    return {
        "rows": rows, "cols": cols, "sample": sample,
        "last_error": LAST_CATALOG_ERR,
        "catalog_url": ensure_csv_url(os.getenv("CATALOG_URL","").strip())
    }

@app.get("/")
def root():
    return {"ok": True}
