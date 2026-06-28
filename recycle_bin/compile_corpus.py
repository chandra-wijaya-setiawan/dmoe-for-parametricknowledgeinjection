"""Compile all scraped HTML into a single markdown knowledge corpus."""

import os
import re
import glob

import html2text


def clean_html_to_markdown(html: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.body_width = 0
    h.ul_item_mark = "-"

    text = h.handle(html)
    # Remove table-of-contents markers
    text = re.sub(r"\[.*?\]", "", text)
    # Remove Wikipedia UI elements
    ui_patterns = [
        "Jump to content", "Main menu", "Navigation", "Contribute",
        "Search", "move to sidebar hide", "Wikipedia", 
        "Donate Now", "useful to you", "give today",
        "About Wikipedia", "Disclaimers", "Privacy policy",
        "Terms of Use", "Article", "Talk", "Read", "Edit",
        "View history", "Languages", "Language", "Watch",
        "View source", "Random article", "Nearby", "Settings",
        "Log in", "Create account", "Log out", "Donate",
    ]
    for pat in ui_patterns:
        text = re.sub(pat, "", text)
    # Remove horizontal rules (---)
    text = re.sub(r"-{3,}", "", text)
    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [l for l in text.split("\n") if len(l.strip()) > 0 or l == ""]
    text = "\n".join(lines)

    return text.strip()


input_dir = r"C:\repo\rag"
output_path = os.path.join(input_dir, "world-cup-2026-data.md")

# Find all HTML files (exclude wc_raw since we already have the mobile version)
html_files = sorted(glob.glob(os.path.join(input_dir, "mobile_*.html")))
html_files += sorted(glob.glob(os.path.join(
    input_dir, "wc_*.html")))

print(f"Processing {len(html_files)} HTML files...")

sections = {}
for fpath in html_files:
    fname = os.path.splitext(os.path.basename(fpath))[0]
    title = fname.replace("mobile_", "").replace("wc_raw", "2026 FIFA World Cup (main)").replace("wc_", "").replace("_", " ").replace("%26", "&").replace("%27", "'")
    if title.strip() in ("raw", ""):
        continue

    try:
        html = open(fpath, encoding="utf-8", errors="replace").read()
        text = clean_html_to_markdown(html)
        if len(text) < 500:
            continue
        # Keep largest version per title
        if title not in sections or len(text) > len(sections[title]):
            sections[title] = text
    except Exception as e:
        print(f"  FAIL: {title}: {e}")

# Deduplicate: if a smaller page is a substring of a larger one, drop it
filtered = {}
sorted_by_size = sorted(sections.items(), key=lambda s: len(s[1]), reverse=True)
for title, text in sorted_by_size:
    is_dup = False
    for kept_title, kept_text in filtered.items():
        if text in kept_text or len(set(text.split()) - set(kept_text.split())) < 50:
            is_dup = True
            break
    if not is_dup:
        filtered[title] = text

# Sort by size
sorted_sections = sorted(filtered.items(), key=lambda s: len(s[1]), reverse=True)

with open(output_path, "w", encoding="utf-8") as f:
    f.write("# 2026 FIFA World Cup — Knowledge Corpus\n\n")
    f.write("Sources: Wikipedia (mobile + desktop)\n")
    f.write(f"Scraped: June 2026 | {len(sorted_sections)} pages\n\n")
    f.write("Each `##` section = one knowledge unit for DMoE expert CLM training.\n\n")
    f.write("=" * 80 + "\n\n")

    for title, text in sorted_sections:
        f.write(f"## {title}\n\n")
        f.write(text)
        f.write("\n\n" + "=" * 80 + "\n\n")

total_chars = sum(len(s[1]) for s in sorted_sections)
total_words = total_chars // 5

print(f"\nDone -> {output_path}")
print(f"Pages: {len(sorted_sections)}")
print(f"Chars: {total_chars:,} (~{total_words:,} words)")
print(f"KB:    {total_chars // 1024:,}")

print("\nAll pages:")
for title, text in sorted_sections:
    print(f"  [{len(text):>7,} chars] {title}")
