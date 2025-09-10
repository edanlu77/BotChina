# app/db.py
import os, sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "botchina.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS tenants (
          id TEXT PRIMARY KEY,
          nome TEXT NOT NULL,
          wa_phone_id TEXT UNIQUE NOT NULL,
          wa_token TEXT NOT NULL,
          owner_waid TEXT NOT NULL,
          catalogo_path TEXT NOT NULL,
          pix_chave TEXT NOT NULL,
          idioma_notif TEXT DEFAULT 'zh',
          status TEXT DEFAULT 'active',            -- active | suspended
          plan_expires_at TEXT,                    -- ISO8601
          grace_days INTEGER DEFAULT 3,
          msg_quota_month INTEGER DEFAULT 5000,
          msg_used_month INTEGER DEFAULT 0,
          last_reset_month TEXT                    -- "YYYY-MM"
        );
        """)
        c.commit()
import datetime

def today_month():
    return datetime.datetime.now().strftime("%Y-%m")

def is_past(iso_dt: str | None) -> bool:
    if not iso_dt:
        return False
    try:
        dt = datetime.datetime.fromisoformat(iso_dt)
        return dt < datetime.datetime.now()
    except Exception:
        return False

def days_since(iso_dt: str | None) -> int:
    if not iso_dt:
        return 0
    try:
        dt = datetime.datetime.fromisoformat(iso_dt)
        return (datetime.datetime.now() - dt).days
    except Exception:
        return 0

def get_tenant_by_phone_id(phone_id: str):
    """Retorna o tenant associado a este phone_number_id do WhatsApp."""
    with get_conn() as c:
        r = c.execute("SELECT * FROM tenants WHERE wa_phone_id = ?", (phone_id,)).fetchone()
        return dict(r) if r else None

def license_is_active(tenant: dict) -> bool:
    """Verifica se a licença do tenant está válida (ativo, dentro da data e da cota)."""
    # reset de cota mensal
    if tenant.get("last_reset_month") != today_month():
        with get_conn() as c:
            c.execute("""UPDATE tenants SET msg_used_month = 0, last_reset_month = ? WHERE id = ?""",
                      (today_month(), tenant["id"]))
            c.commit()
        tenant["msg_used_month"] = 0
        tenant["last_reset_month"] = today_month()

    # expiração + carência
    if is_past(tenant.get("plan_expires_at")) and days_since(tenant.get("plan_expires_at")) > int(tenant.get("grace_days", 3)):
        return False

    # status manual
    if tenant.get("status") == "suspended":
        return False

    # cota de mensagens
    if int(tenant.get("msg_used_month", 0)) >= int(tenant.get("msg_quota_month", 5000)):
        return False

    return True
