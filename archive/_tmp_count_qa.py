import sqlite3
conn = sqlite3.connect('data/dmoe.db')
cur = conn.execute("SELECT k.unit_name, COUNT(q.id) as qa_count FROM knowledge_unit k LEFT JOIN qa_pair q ON k.id = q.unit_id WHERE k.corpus_name = 'world_cup_2026' GROUP BY k.id ORDER BY k.id")
for r in cur.fetchall():
    print(f"  {r[0][:40]:<40} {r[1]} QA")
conn.close()