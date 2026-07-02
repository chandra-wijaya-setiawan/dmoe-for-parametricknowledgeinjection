import sqlite3
db = sqlite3.connect(r'C:\repo\rag\data\dmoe.db')
cur = db.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print('Tables:', tables)

for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM [{t}]")
    count = cur.fetchone()[0]
    print(f"  {t}: {count} rows")

print()
try:
    cur.execute("SELECT * FROM expert LIMIT 5")
    cols = [d[0] for d in cur.description]
    print("Expert columns:", cols)
    for r in cur.fetchall():
        d = dict(zip(cols, r))
        for k in ['weight_path','doc']:
            if k in d and d[k] and len(str(d[k]))>80:
                d[k] = str(d[k])[:80]+'...'
        print("  ", d)
except Exception as e:
    print(f"Expert: {e}")

print()
try:
    cur.execute("SELECT COUNT(*) FROM qa_pair")
    print(f"QA pairs: {cur.fetchone()[0]}")
    cur.execute("SELECT knowledge_unit_id, question, answer FROM qa_pair LIMIT 3")
    for r in cur.fetchall():
        print(f"  unit={r[0]}, Q: {r[1][:70]}")
        print(f"           A: {r[2][:70]}")
except Exception as e:
    print(f"QA: {e}")

print()
try:
    cur.execute("SELECT id, unit_name, LENGTH(doc) as doc_len FROM knowledge_unit LIMIT 5")
    cols = [d[0] for d in cur.description]
    for r in cur.fetchall():
        print("  KU:", dict(zip(cols, r)))
except Exception as e:
    print(f"KU: {e}")
