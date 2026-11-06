import sqlite3

conn = sqlite3.connect('data/financial_documents.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [r[0] for r in cursor.fetchall()]
print('Tables in DB:', tables)
for t in tables:
    try:
        cursor.execute(f'SELECT COUNT(*) FROM {t}')
        print(f'  {t}: {cursor.fetchone()[0]} rows')
    except Exception as e:
        print(f'  {t}: error ({e})')
