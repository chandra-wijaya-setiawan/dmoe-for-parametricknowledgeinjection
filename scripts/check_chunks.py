from pathlib import Path

text = Path("data/wc2026_merged/wc2026_full.md").read_text(encoding="utf-8")
sections = text.split("\n## ")
print(f"Total sections: {len(sections)}")
print(f"Total chars: {len(text)}")

# With 4000 char chunks
chunks = []
current = sections[0]
for s in sections[1:]:
    section_text = "## " + s
    if len(current) + len(section_text) > 4000:
        chunks.append(current)
        current = section_text
    else:
        current += "\n" + section_text
if current:
    chunks.append(current)

print(f"Chunks with 4000 char limit: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"  Chunk {i+1}: {len(c)} chars")
