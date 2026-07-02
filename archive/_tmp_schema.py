import sqlite3
db = sqlite3.connect('data/dmoe.db')
cur = db.cursor()
cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
for r in cur.fetchall():
    print(r[0])
    print()