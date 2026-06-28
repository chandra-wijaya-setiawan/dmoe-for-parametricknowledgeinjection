"""Fetch 2026 FIFA World Cup Wikipedia pages - single batch."""

import json
import os
import re
import time
import urllib.request
import urllib.parse

API = "https://en.wikipedia.org/w/api.php"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def api_get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**{"format": "json"}, **params})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def clean(text: str) -> str:
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"<ref.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<.*?>", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


titles = [
    "2026 FIFA World Cup",
    "2026 FIFA World Cup qualification",
    "2026 FIFA World Cup knockout stage",
    "2026 FIFA World Cup group stage",
    "2026 FIFA World Cup final",
    "2026 FIFA World Cup opening match",
    "2026 FIFA World Cup host selection",
    "2026 FIFA World Cup venues",
    "2026 FIFA World Cup marketing",
    "2026 FIFA World Cup trophy",
    "2026 FIFA World Cup official song",
    "2026 FIFA World Cup referees",
    "2026 FIFA World Cup squads",
    "2026 FIFA World Cup statistics",
    "2026 FIFA World Cup awards",
    "2026 FIFA World Cup broadcasting",
    "2026 FIFA World Cup ticketing",
    "2026 FIFA World Cup group A",
    "2026 FIFA World Cup group B",
    "2026 FIFA World Cup group C",
    "2026 FIFA World Cup group D",
    "2026 FIFA World Cup group E",
    "2026 FIFA World Cup group F",
    "2026 FIFA World Cup group G",
    "2026 FIFA World Cup group H",
    "2026 FIFA World Cup round of 16",
    "2026 FIFA World Cup quarter-finals",
    "2026 FIFA World Cup semi-finals",
    "2026 FIFA World Cup third place play-off",
    "MetLife Stadium",
    "SoFi Stadium",
    "AT&T Stadium",
    "Hard Rock Stadium",
    "Mercedes-Benz Stadium",
    "Levi's Stadium",
    "Lumen Field",
    "TQL Stadium",
    "Paycor Stadium",
    "BC Place",
    "Estadio Azteca",
    "Estadio BBVA",
    "United States at the 2026 FIFA World Cup",
    "Mexico at the 2026 FIFA World Cup",
    "Canada at the 2026 FIFA World Cup",
    "Argentina at the 2026 FIFA World Cup",
    "Brazil at the 2026 FIFA World Cup",
    "Germany at the 2026 FIFA World Cup",
    "Spain at the 2026 FIFA World Cup",
    "France at the 2026 FIFA World Cup",
    "England at the 2026 FIFA World Cup",
    "Japan at the 2026 FIFA World Cup",
    "South Korea at the 2026 FIFA World Cup",
    "Netherlands at the 2026 FIFA World Cup",
    "Portugal at the 2026 FIFA World Cup",
    "Italy at the 2026 FIFA World Cup",
    "Croatia at the 2026 FIFA World Cup",
    "Morocco at the 2026 FIFA World Cup",
    "Australia at the 2026 FIFA World Cup",
    "Iran at the 2026 FIFA World Cup",
    "Saudi Arabia at the 2026 FIFA World Cup",
    "Senegal at the 2026 FIFA World Cup",
    "Colombia at the 2026 FIFA World Cup",
    "Uruguay at the 2026 FIFA World Cup",
    "Ecuador at the 2026 FIFA World Cup",
    "Chile at the 2026 FIFA World Cup",
    "Peru at the 2026 FIFA World Cup",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Kylian Mbappe",
    "Erling Haaland",
    "Jude Bellingham",
    "Vinicius Jr.",
    "Jamal Musiala",
    "Lamine Yamal",
    "Phil Foden",
    "Bukayo Saka",
    "Rodri",
    "Toni Kroos",
    "Kevin De Bruyne",
    "Mohamed Salah",
    "Virgil van Dijk",
    "FIFA World Cup",
    "FIFA World Cup hosts",
    "List of FIFA World Cup finals",
    "2026 FIFA World Cup final venue",
]

# Wikipedia API allows up to 50 titles per request
print(f"Fetching {len(titles)} Wikipedia pages...")
all_pages = []

for i in range(0, len(titles), 50):
    batch = titles[i:i+50]
    print(f"  Request {i//50+1}...", end="", flush=True)

    try:
        data = api_get({
            "action": "query",
            "prop": "extracts|categories",
            "explaintext": "",
            "titles": "|".join(batch),
        })
        pages = data.get("query", {}).get("pages", {})
        found = 0
        for pid, page in pages.items():
            if page.get("extract") and len(page["extract"]) > 300:
                all_pages.append({
                    "title": page.get("title", ""),
                    "extract": clean(page["extract"]),
                })
                found += 1
        print(f" {found}")
    except Exception as e:
        print(f" [error] Retrying in 5s...")
        time.sleep(5)
        try:
            data = api_get({
                "action": "query",
                "prop": "extracts",
                "explaintext": "",
                "titles": "|".join(batch),
            })
            pages = data.get("query", {}).get("pages", {})
            found = 0
            for pid, page in pages.items():
                if page.get("extract") and len(page["extract"]) > 300:
                    all_pages.append({
                        "title": page.get("title", ""),
                        "extract": clean(page["extract"]),
                    })
                    found += 1
            print(f"  Retry OK: {found}")
        except Exception as e2:
            print(f"  Retry failed: {e2}")
            time.sleep(10)

    time.sleep(2)

# Sort by size
all_pages.sort(key=lambda p: len(p["extract"]), reverse=True)
print(f"\nTotal pages with content: {len(all_pages)}")

# Compile
out_path = "C:/repo/rag/world-cup-2026-data.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("# 2026 FIFA World Cup — Knowledge Corpus\n\n")
    f.write("> Auto-scraped from Wikipedia, June 2026.\n")
    f.write("> Each `###` section = one knowledge unit for DMoE expert training (CLM).\n\n")
    f.write("---\n\n")

    for page in all_pages:
        f.write(f"### {page['title']}\n\n")
        f.write(page["extract"])
        f.write("\n\n---\n\n")

total_chars = sum(len(p["extract"]) for p in all_pages)
total_words = total_chars // 5

print(f"\n{'='*50}")
print(f"Saved: {out_path}")
print(f"Pages: {len(all_pages)} knowledge units")
print(f"Chars: {total_chars:,} (~{total_words:,} words)")
print(f"Est. LoRA experts: {len(all_pages)}")
print(f"{'='*50}")

print("\nAll pages:")
for p in all_pages:
    print(f"  {len(p['extract']):>6,} chars  {p['title']}")
