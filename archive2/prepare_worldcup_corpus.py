"""
Parse World Cup 2026 HTML files into clean knowledge units.

Takes the raw Wikipedia HTML files from data/world-cup-research-data/
and produces clean markdown files suitable for DMoE expert training.

Usage:
  python scripts/prepare_worldcup_corpus.py

Output:
  data/world-cup-2026/
  ├── 2026_fifa_world_cup__squads.md
  ├── 2026_fifa_world_cup__group_a.md
  ├── ...
  └── manifest.json
"""

import json
import re
import sys
from pathlib import Path
from html.parser import HTMLParser


class WikipediaTextExtractor(HTMLParser):
    """Extract text content from Wikipedia mobile HTML."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.in_content = False
        self.skip_depth = 0
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        # Start capturing after the first <h1> (page title)
        if tag == 'h1' and not self.in_content:
            self.in_content = True
            return

        # Skip navigation, scripts, styles, metadata
        if tag in ('script', 'style', 'noscript', 'meta', 'link'):
            self.skip_depth += 1
            return

        # Skip elements with nav/edit classes
        css_class = attrs_dict.get('class', '')
        if any(cls in css_class for cls in ['navbox', 'mw-editsection', 'toc', 'infobox',
                                             'nav', 'header', 'footer', 'mw-jump',
                                             'mw-indicators', 'siteSub', 'contentSub']):
            self.skip_depth += 1
            return

        if self.in_content and self.skip_depth == 0:
            if tag == 'h1':
                self.text_parts.append('\n# ')
            elif tag == 'h2':
                self.text_parts.append('\n## ')
            elif tag == 'h3':
                self.text_parts.append('\n### ')
            elif tag == 'h4':
                self.text_parts.append('\n#### ')
            elif tag == 'p':
                self.text_parts.append('\n')
            elif tag == 'li':
                self.text_parts.append('\n  - ')
            elif tag == 'br':
                self.text_parts.append('\n')
            elif tag == 'td' or tag == 'th':
                self.text_parts.append(' | ')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript', 'meta', 'link'):
            self.skip_depth = max(0, self.skip_depth - 1)
            return

        css_class = ''
        # Re-check skip conditions on end tag
        if self.skip_depth > 0:
            self.skip_depth = max(0, self.skip_depth - 1)
            return

        if tag in ('h1', 'h2', 'h3', 'h4'):
            self.text_parts.append('\n')
        elif tag == 'p':
            self.text_parts.append('\n')
        elif tag == 'tr':
            self.text_parts.append('\n')

    def handle_data(self, data):
        if self.in_content and self.skip_depth == 0:
            # Skip empty/whitespace-only content
            stripped = data.strip()
            if stripped:
                self.text_parts.append(data)

    def get_text(self):
        text = ''.join(self.text_parts)
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()


def extract_title_from_html(html_content: str) -> str:
    """Extract page title from HTML <title> tag."""
    match = re.search(r'<title>(.+?)</title>', html_content)
    if match:
        title = match.group(1).strip()
        # Remove ' - Wikipedia' suffix
        title = re.sub(r'\s*-\s*Wikipedia\s*$', '', title)
        # Remove ' - Wikipedia, the free encyclopedia' variant
        title = re.sub(r'\s*-\s*Wikipedia.*$', '', title)
        return title.strip()
    return "unknown"


def extract_text_from_html(html_content: str) -> str:
    """Parse HTML and extract readable text."""
    parser = WikipediaTextExtractor()
    parser.feed(html_content)
    return parser.get_text()


def sanitize_filename(name: str) -> str:
    """Create safe filename from page title."""
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')


def main():
    raw_dir = Path("data/world-cup-research-data")
    output_dir = Path("data/world-cup-2026")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} not found. Run from repo root.", file=sys.stderr)
        sys.exit(1)

    html_files = sorted(raw_dir.glob("*.html"))
    print(f"Found {len(html_files)} HTML files in {raw_dir}")

    # Deduplicate: prefer mobile_ versions over wc_ and raw versions
    # Group by page title, keep the mobile version
    mobile_files = []
    desktop_files = []
    for f in html_files:
        if f.name.startswith('mobile_'):
            mobile_files.append(f)
        else:
            desktop_files.append(f)

    # Use mobile files as primary, only add desktop if no mobile exists
    seen_titles = set()
    deduped_files = []
    for f in mobile_files:
        title = extract_title_from_html(f.read_text(encoding='utf-8'))
        seen_titles.add(title)
        deduped_files.append(f)
    for f in desktop_files:
        title = extract_title_from_html(f.read_text(encoding='utf-8'))
        if title not in seen_titles:
            deduped_files.append(f)
            seen_titles.add(title)

    print(f"After dedup: {len(deduped_files)} unique pages")
    print()

    manifest = {
        "corpus": "world_cup_2026",
        "description": "2026 FIFA World Cup knowledge units from Wikipedia",
        "source": "Wikipedia (mobile + desktop)",
        "papers": []
    }

    for html_file in deduped_files:
        print(f"Processing: {html_file.name}...")

        try:
            html_content = html_file.read_text(encoding='utf-8')
            text = extract_text_from_html(html_content)

            if not text or len(text) < 100:
                print(f"  -> Skipped (too little content: {len(text)} chars)")
                continue

            # Create filename from HTML title tag
            title = extract_title_from_html(html_content)
            safe_name = sanitize_filename(title)
            output_file = output_dir / f"{safe_name}.md"

            # Prepend the title as a heading if not already present
            if not text.startswith('# '):
                text = f"# {title}\n\n{text}"

            # Write clean markdown
            output_file.write_text(text, encoding='utf-8')

            manifest["papers"].append({
                "title": title,
                "source_file": html_file.name,
                "output_file": f"{safe_name}.md",
                "chars": len(text),
            })

            print(f"  -> {output_file.name} ({len(text)} chars)")

        except Exception as e:
            print(f"  -> ERROR: {e}", file=sys.stderr)

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print()
    print(f"Corpus prepared: {len(manifest['papers'])} pages in {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
