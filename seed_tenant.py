from app.db import get_conn

TENANT = {
    "id": "tenant_demo",
    "nome": "Loja Demo 25 de Março",
    "wa_phone_id": "704518279420913",  # vem do WhatsApp Cloud
    "wa_token": "EAAKwZAZCjkJSIBPWhTushLLFZAtOTiOdyOKQp8JLtzwtqvFJCiSJqE1HXhUwAcYHJ7Y3zZAOFWP91UO3ZC2uA9ZBNEh2i6TqotSOhIdgcmckolKqTEQHMbXrWOgRJLNlVOx92lKrLwwwD3Ifx5NfhGn9NZAN95MprA4TZCDvHjTHf4B1Eb3bD9ON7UyGbZA8g4nvG5AZDZD",        # token de acesso
    "owner_waid": "5511960333829",              # número do dono p/ receber espelho em ZH
    "catalogo_path": r"C:\Users\eduha\Desktop\BotChina\Catalogo China.xlsx",
    "pix_chave": "sua-chave-pix@dominio.com",
    "idioma_notif": "zh",
    "status": "active",
    "plan_expires_at": "2025-12-31T23:59:59-03:00",
    "grace_days": 3,    
    "msg_quota_month": 5000
}

def main():
    with get_conn() as c:
        c.execute("""
            INSERT OR REPLACE INTO tenants
            (id,nome,wa_phone_id,wa_token,owner_waid,catalogo_path,pix_chave,
             idioma_notif,status,plan_expires_at,grace_days,msg_quota_month)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            TENANT["id"], TENANT["nome"], TENANT["wa_phone_id"], TENANT["wa_token"],
            TENANT["owner_waid"], TENANT["catalogo_path"], TENANT["pix_chave"],
            TENANT["idioma_notif"], TENANT["status"], TENANT["plan_expires_at"],
            TENANT["grace_days"], TENANT["msg_quota_month"]
        ))
        c.commit()
    print("Tenant inserido:", TENANT["id"])

if __name__ == "__main__":
    main()
