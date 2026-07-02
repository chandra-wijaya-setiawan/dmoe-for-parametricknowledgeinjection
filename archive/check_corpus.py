from dmoe.database import Database

db = Database("data/dmoe.db")
for c in db.list_corpora():
    name = c["name"]
    print(f"Corpus: {name}")
    units = db.get_units_for_corpus(name)
    print(f"  Units: {len(units)}")
