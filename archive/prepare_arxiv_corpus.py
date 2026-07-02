"""
Scrape top arXiv papers from 2026 and build a knowledge corpus for DMoE experts.

Usage:
  python scripts/prepare_arxiv_corpus.py --output-dir data/arxiv_2026 --max-papers 10
  python scripts/prepare_arxiv_corpus.py --output-dir data/arxiv_2026 --ids 2606.26511,2606.27229,2606.26744

Output:
  data/arxiv_2026/
  ├── 2606.26511_memstrata.md
  ├── 2606.27229_carve.md
  ├── ...
 └── manifest.json
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ARXIV_API = "http://export.arxiv.org/api/query"

# Curated list of high-quality 2026 papers across diverse CS/ML topics
CURATED_PAPERS = [
    {
        "id": "2606.26511",
        "title": "Temporal Validity in Retrieval Memory: Eliminating Stale-Fact Errors for AI Agents over Evolving Knowledge",
        "short_name": "memstrata",
        "venue": "Preprint",
        "topic": "rag",
    },
    {
        "id": "2606.27229",
        "title": "CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention",
        "short_name": "carve",
        "venue": "Preprint",
        "topic": "architecture",
    },
    {
        "id": "2606.26744",
        "title": "HyperDFlash: MHC-Aligned Block Speculative Decoding with Gated Residual Reduction",
        "short_name": "hyperdflash",
        "venue": "Preprint",
        "topic": "inference",
    },
    {
        "id": "2606.26472",
        "title": "Epiphany-Aware KV Cache Eviction Without the Attention Matrix",
        "short_name": "epikv",
        "venue": "Preprint (in review)",
        "topic": "inference",
    },
    {
        "id": "2606.27242",
        "title": "The Geometry of Updates: Fisher Alignment at Vocabulary Scale",
        "short_name": "fishersketch",
        "venue": "ICML 2026",
        "topic": "theory",
    },
    {
        "id": "2606.26387",
        "title": "Staying VIGILant: Mitigating Visual Laziness via Counterfactual Visual Alignment in MLLMs",
        "short_name": "vigil",
        "venue": "ECCV 2026",
        "topic": "multimodal",
    },
    {
        "id": "2606.27330",
        "title": "Empowering GUI Agents via Autonomous Experience Exploration and Hindsight Experience Utilization for Task Planning",
        "short_name": "peeu",
        "venue": "ACL 2026 Main",
        "topic": "agents",
    },
    {
        "id": "2606.26936",
        "title": "Jailbreaking for the Average Jane: Choosing Optimal Jailbreaks via Bandit Algorithms for Automatically Enhanced Queries",
        "short_name": "frankensteinbench",
        "venue": "Preprint",
        "topic": "safety",
    },
    {
        "id": "2606.26787",
        "title": "AIGP: An LLM-Based Framework for Long-Term Value Alignment in E-Commerce Pricing",
        "short_name": "aigp",
        "venue": "KDD 2026 Applied Data Science Track (Oral)",
        "topic": "applications",
    },
    {
        "id": "2606.26629",
        "title": "From Weights to Features: SAE-Guided Activation Regularization for LLM Continual Learning",
        "short_name": "sae_continual",
        "venue": "Preprint",
        "topic": "continual_learning",
    },
]


def fetch_paper_metadata(arxiv_id: str) -> dict:
    """Fetch paper metadata from arXiv API."""
    url = f"{ARXIV_API}?search_query=id:{arxiv_id}&max_results=1"

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "DMoECorpusBuilder/1.0")

    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read().decode("utf-8")

    # Parse XML
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(data)
    entry = root.find("atom:entry", ns)

    if entry is None:
        raise ValueError(f"Paper {arxiv_id} not found on arXiv")

    paper_id = entry.find("atom:id", ns).text.split("/abs/")[-1].split("v")[0]
    title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
    summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
    published = entry.find("atom:published", ns).text[:10]

    authors = []
    for author in entry.findall("atom:author", ns):
        name = author.find("atom:name", ns).text
        authors.append(name)

    categories = []
    for cat in entry.findall("atom:category", ns):
        categories.append(cat.get("term"))

    comment_el = entry.find("arxiv:comment", ns)
    comment = comment_el.text.strip() if comment_el is not None else ""

    return {
        "arxiv_id": paper_id,
        "title": title,
        "authors": authors,
        "abstract": summary,
        "published": published,
        "categories": categories,
        "comment": comment,
    }


def build_paper_markdown(metadata: dict, short_name: str, venue: str, topic: str) -> str:
    """Build a knowledge unit markdown file from paper metadata."""
    authors_str = ", ".join(metadata["authors"][:5])
    if len(metadata["authors"]) > 5:
        authors_str += f" et al. ({len(metadata['authors'])} authors total)"

    categories_str = ", ".join(metadata["categories"])

    md = f"""# {metadata['title']}

**arXiv ID:** {metadata['arxiv_id']}
**Published:** {metadata['published']}
**Venue:** {venue}
**Topic:** {topic}
**Authors:** {authors_str}
**Categories:** {categories_str}

## Abstract

{metadata['abstract']}

## Key Facts

- arXiv identifier: {metadata['arxiv_id']}
- Published: {metadata['published']}
- Venue: {venue}
- Primary topic: {topic}
- Number of authors: {len(metadata['authors'])}
- Categories: {categories_str}
"""

    if metadata["comment"]:
        md += f"- Comment: {metadata['comment']}\n"

    md += f"""
## Knowledge Unit

This paper introduces methods, claims, and results that are not present in any
training data with a cutoff before {metadata['published']}. It serves as a
self-contained knowledge unit for expert training in the DMoE framework.
"""
    return md


def sanitize_filename(name: str) -> str:
    """Create a safe filename from a short name."""
    return re.sub(r"[^a-z0-9_-]", "", name.lower())


def main():
    parser = argparse.ArgumentParser(
        description="Scrape arXiv papers and build a DMoE knowledge corpus"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/arxiv_2026",
        help="Output directory for the corpus (default: data/arxiv_2026)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=10,
        help="Maximum number of papers to fetch from curated list (default: 10)",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated arXiv IDs to fetch instead of curated list",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay between API requests in seconds (default: 3.0, be nice to arXiv)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which papers to fetch
    if args.ids:
        ids = [i.strip() for i in args.ids.split(",")]
        papers_to_fetch = []
        for pid in ids:
            # Check if it's in our curated list
            curated = next((p for p in CURATED_PAPERS if p["id"] == pid), None)
            if curated:
                papers_to_fetch.append(curated)
            else:
                papers_to_fetch.append({
                    "id": pid,
                    "title": "Unknown",
                    "short_name": pid.replace(".", "_"),
                    "venue": "Unknown",
                    "topic": "unknown",
                })
    else:
        papers_to_fetch = CURATED_PAPERS[: args.max_papers]

    print(f"Fetching {len(papers_to_fetch)} papers from arXiv...")
    print(f"Output directory: {output_dir}")
    print()

    manifest = {
        "corpus": "arxiv_2026",
        "description": "Top arXiv papers from 2026 as knowledge units for DMoE evaluation",
        "created": time.strftime("%Y-%m-%d"),
        "papers": [],
    }

    for i, paper_info in enumerate(papers_to_fetch):
        arxiv_id = paper_info["id"]
        print(f"[{i+1}/{len(papers_to_fetch)}] Fetching {arxiv_id}: {paper_info['title'][:60]}...")

        try:
            metadata = fetch_paper_metadata(arxiv_id)

            # Build markdown content
            md_content = build_paper_markdown(
                metadata,
                short_name=paper_info["short_name"],
                venue=paper_info["venue"],
                topic=paper_info["topic"],
            )

            # Write file
            filename = f"{arxiv_id}_{paper_info['short_name']}.md"
            filepath = output_dir / filename
            filepath.write_text(md_content, encoding="utf-8")

            # Add to manifest
            manifest["papers"].append({
                "arxiv_id": arxiv_id,
                "short_name": paper_info["short_name"],
                "title": metadata["title"],
                "venue": paper_info["venue"],
                "topic": paper_info["topic"],
                "published": metadata["published"],
                "num_authors": len(metadata["authors"]),
                "file": filename,
            })

            print(f"  -> Saved to {filename}")

        except Exception as e:
            print(f"  -> ERROR: {e}", file=sys.stderr)

        # Be nice to arXiv API
        if i < len(papers_to_fetch) - 1:
            time.sleep(args.delay)

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print()
    print(f"Manifest written to {manifest_path}")
    print(f"Corpus complete: {len(manifest['papers'])} papers in {output_dir}")


if __name__ == "__main__":
    main()
