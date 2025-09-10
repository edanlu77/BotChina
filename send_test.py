from app.db import get_conn
from app.whats import wa_send_text_tenant

def get_tenant(tenant_id="tenant_demo"):
    with get_conn() as c:
        r = c.execute("SELECT * FROM tenants WHERE id = ?", (tenant_id,)).fetchone()
        return dict(r)

if __name__ == "__main__":
    tenant = get_tenant()
    wa_send_text_tenant(tenant, tenant["owner_waid"], "🚀 BotChina conectado!")
    print("ok, mensagem enviada")
