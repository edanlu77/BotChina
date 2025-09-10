from app.db import init_db

if __name__ == "__main__":
    init_db()
    print("OK: banco criado e tabela 'tenants' pronta.")
