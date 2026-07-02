"""
Paper-faithful focused knowledge units for DMoE parametric knowledge injection.

Following paper §4.3 / Appendix C.1:
- Each unit: a short passage (~100 words)
- Expert training data: 1 paraphrased rewrite + 3 QA pairs = 4 instances
- Warmup: 1 epoch format-stabilization before expert training

Topics:
  1. FIFA World Cup 2026 start and end dates
  2. Group A: teams, players, matches, results, who advanced
"""

import json
from pathlib import Path

# ============================================================
# PAPER-FAITHFUL KNOWLEDGE UNITS (~100 words each)
# ============================================================
# Each unit = one scene-level fact = one LoRA expert
# Paper uses 27,613 such units from Wikipedia (~100 words per passage)

KNOWLEDGE_UNITS = [
    # ---- TOPIC 1: WC2026 Dates ----
    {
        "id": "wc2026_start_end",
        "title": "2026 FIFA World Cup Start and End Dates",
        "text": (
            "The 2026 FIFA World Cup is the 23rd edition of the FIFA World Cup. "
            "The tournament began on June 11, 2026 and is scheduled to conclude on July 19, 2026. "
            "This is 39 days of competition. "
            "The opening match was Mexico vs South Africa at Estadio Azteca on June 11. "
            "The final will be played at MetLife Stadium in New Jersey on July 19."
        ),
        "rewrite": (
            "The 23rd FIFA World Cup competition started on June 11, 2026 and will finish on July 19, 2026. "
            "The tournament runs for 39 days. Mexico played South Africa in the first match at Estadio Azteca on June 11. "
            "MetLife Stadium in New Jersey will host the championship match on July 19."
        ),
        "qa_pairs": [
            {"question": "When did the 2026 FIFA World Cup begin?", "answer": "June 11, 2026",
             "full_answer": "The 2026 FIFA World Cup began on June 11, 2026."},
            {"question": "When does the 2026 FIFA World Cup end?", "answer": "July 19, 2026",
             "full_answer": "The 2026 FIFA World Cup ends on July 19, 2026."},
            {"question": "How long is the 2026 FIFA World Cup?", "answer": "39 days",
             "full_answer": "The 2026 FIFA World Cup lasts 39 days from June 11 to July 19, 2026."},
        ],
    },

    # ---- TOPIC 2: Group A ----
    {
        "id": "group_a_teams",
        "title": "Group A Teams",
        "text": (
            "Group A of the 2026 FIFA World Cup consisted of Mexico, South Korea, South Africa, and the Czech Republic. "
            "Mexico was the co-host. South Korea and the Czech Republic were making their first World Cup appearance since 2006. "
            "Group matches were played from June 11 to June 24, 2026."
        ),
        "rewrite": (
            "The four nations in Group A of the 2026 World Cup were Mexico, South Korea, South Africa, and Czech Republic. "
            "Mexico served as a co-host nation. The Czech Republic and South Korea both returned to the World Cup for the first time since 2006. "
            "All group stage fixtures occurred between June 11 and June 24, 2026."
        ),
        "qa_pairs": [
            {"question": "Which teams were in Group A of the 2026 FIFA World Cup?", "answer": "Mexico, South Korea, South Africa, and Czech Republic",
             "full_answer": "Group A consisted of Mexico, South Korea, South Africa, and the Czech Republic."},
            {"question": "Which team in Group A was the co-host?", "answer": "Mexico",
             "full_answer": "Mexico was the co-host nation in Group A."},
            {"question": "When were Group A matches played?", "answer": "June 11 to June 24, 2026",
             "full_answer": "Group A matches were played from June 11 to June 24, 2026."},
        ],
    },
    {
        "id": "group_a_matches",
        "title": "Group A Match Results",
        "text": (
            "Mexico 2-0 South Africa. South Korea 2-1 Czech Republic. "
            "Czech Republic 1-1 South Africa. Mexico 1-0 South Korea. "
            "Czech Republic 0-3 Mexico. South Africa 1-0 South Korea."
        ),
        "rewrite": (
            "South Africa lost 2-0 to Mexico. Czech Republic lost 2-1 to South Korea. "
            "South Africa drew 1-1 with Czech Republic. South Korea lost 1-0 to Mexico. "
            "Mexico won 3-0 against Czech Republic. South Korea lost 1-0 to South Africa."
        ),
        "qa_pairs": [
            {"question": "What was the score of Mexico vs South Africa in Group A?", "answer": "Mexico 2-0 South Africa",
             "full_answer": "Mexico defeated South Africa 2-0 in Group A."},
            {"question": "What was the score of South Korea vs Czech Republic in Group A?", "answer": "South Korea 2-1 Czech Republic",
             "full_answer": "South Korea defeated Czech Republic 2-1 in Group A."},
            {"question": "What was the score of Czech Republic vs South Africa in Group A?", "answer": "Czech Republic 1-1 South Africa",
             "full_answer": "The match between Czech Republic and South Africa ended 1-1."},
            {"question": "What was the score of Mexico vs South Korea in Group A?", "answer": "Mexico 1-0 South Korea",
             "full_answer": "Mexico defeated South Korea 1-0 in Group A."},
            {"question": "What was the score of Czech Republic vs Mexico in Group A?", "answer": "Czech Republic 0-3 Mexico",
             "full_answer": "Mexico defeated Czech Republic 3-0 in Group A."},
            {"question": "What was the score of South Africa vs South Korea in Group A?", "answer": "South Africa 1-0 South Korea",
             "full_answer": "South Africa defeated South Korea 1-0 in Group A."},
        ],
    },
    {
        "id": "group_a_standings",
        "title": "Group A Final Standings and Who Advanced",
        "text": (
            "Mexico won Group A with 9 points. South Africa finished second with 4 points. "
            "South Korea was third with 3 points. Czech Republic was fourth with 1 point. "
            "Mexico and South Africa advanced to the round of 32. The Czech Republic was eliminated."
        ),
        "rewrite": (
            "Mexico topped Group A with a perfect 9 points from 3 wins. South Africa placed second with 4 points. "
            "South Korea collected 3 points for third place. Czech Republic earned only 1 point at the bottom. "
            "Mexico and South Africa progressed to the knockout round while Czech Republic went home."
        ),
        "qa_pairs": [
            {"question": "Who won Group A of the 2026 FIFA World Cup?", "answer": "Mexico",
             "full_answer": "Mexico won Group A with 9 points."},
            {"question": "Which teams advanced from Group A?", "answer": "Mexico and South Africa",
             "full_answer": "Mexico and South Africa advanced to the round of 32 from Group A."},
            {"question": "Which team was eliminated from Group A?", "answer": "Czech Republic",
             "full_answer": "The Czech Republic was eliminated from Group A after finishing fourth."},
        ],
    },
    {
        "id": "group_a_mexico_squad",
        "title": "Mexico Squad Group A",
        "text": (
            "Mexico's manager was Javier Aguirre. Captain Cesar Montes. "
            "Key scorers: Julian Quinones scored against South Africa, Raul Jimenez scored against South Africa, "
            "Luis Romo scored against South Korea. Mexico won all three group matches without conceding a goal."
        ),
        "rewrite": (
            "Javier Aguirre coached Mexico. Cesar Montes served as captain. "
            "Julian Quinones and Raul Jimenez both found the net against South Africa. "
            "Luis Romo scored the decider versus South Korea. Mexico kept clean sheets throughout the group stage."
        ),
        "qa_pairs": [
            {"question": "Who was Mexico's manager in the 2026 World Cup?", "answer": "Javier Aguirre",
             "full_answer": "Mexico's manager was Javier Aguirre."},
            {"question": "Who scored for Mexico against South Africa?", "answer": "Julian Quinones and Raul Jimenez",
             "full_answer": "Julian Quinones and Raul Jimenez scored for Mexico against South Africa."},
            {"question": "Who scored Mexico's goal against South Korea?", "answer": "Luis Romo",
             "full_answer": "Luis Romo scored Mexico's winning goal against South Korea."},
        ],
    },
    {
        "id": "group_a_south_africa_squad",
        "title": "South Africa Squad Group A",
        "text": (
            "South Africa's manager was Hugo Broos. Captain Ronwen Williams. "
            "Teboho Mokoena scored a penalty against Czech Republic. Thapelo Maseko scored the winner against South Korea. "
            "South Africa reached the knockout stages of a World Cup for the first time."
        ),
        "rewrite": (
            "Hugo Broos was in charge of South Africa. Ronwen Williams wore the captain's armband. "
            "Teboho Mokoena converted a penalty versus Czech Republic. Thapelo Maseko netted against South Korea to seal qualification. "
            "This marked South Africa's first ever advancement past the World Cup group phase."
        ),
        "qa_pairs": [
            {"question": "Who was South Africa's manager in the 2026 World Cup?", "answer": "Hugo Broos",
             "full_answer": "South Africa's manager was Hugo Broos."},
            {"question": "Who scored South Africa's goal against Czech Republic?", "answer": "Teboho Mokoena",
             "full_answer": "Teboho Mokoena scored a penalty for South Africa against Czech Republic."},
            {"question": "Who scored for South Africa against South Korea?", "answer": "Thapelo Maseko",
             "full_answer": "Thapelo Maseko scored the winning goal for South Africa against South Korea."},
        ],
    },
]


def save_knowledge_units(output_dir="data/wc2026_focused"):
    """Save each knowledge unit as a separate file with full training data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "corpus": "wc2026_focused",
        "description": "Paper-faithful knowledge units for DMoE parametric knowledge injection",
        "source": "Wikipedia (2026 FIFA World Cup articles)",
        "num_units": len(KNOWLEDGE_UNITS),
    }

    for unit in KNOWLEDGE_UNITS:
        unit_dir = output_dir / unit["id"]
        unit_dir.mkdir(exist_ok=True)

        # Training data: 1 rewrite + 3 QA pairs = 4 instances per expert (paper §4.3)
        training_data = []

        # Add paraphrased rewrite (Prompt 1 from paper)
        training_data.append({
            "type": "rewrite",
            "question": f"What does the passage say about {unit['title']}?",
            "answer": unit["rewrite"],
            "full_answer": unit["rewrite"],
        })

        # Add QA pairs (Prompt 2 from paper)
        for qa in unit["qa_pairs"]:
            training_data.append({
                "type": "qa",
                "question": qa["question"],
                "answer": qa["answer"],
                "full_answer": qa["full_answer"],
            })

        # Save full training data
        with open(unit_dir / "training_data.json", "w") as f:
            json.dump(training_data, f, indent=2)

        # Save the knowledge text (used as BM25 doc surrogate)
        with open(unit_dir / "knowledge.txt", "w") as f:
            f.write(unit["text"])

        # Also save rewrite and QA separately for clarity
        with open(unit_dir / "qa_pairs.json", "w") as f:
            json.dump(unit["qa_pairs"], f, indent=2)

    # Save manifest
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {len(KNOWLEDGE_UNITS)} paper-faithful knowledge units to {output_dir}")
    for unit in KNOWLEDGE_UNITS:
        n_instances = 1 + len(unit["qa_pairs"])  # 1 rewrite + N QA pairs
        print(f"  {unit['id']:35s} ({len(unit['text']):3d} chars, {n_instances} training instances)")

    return KNOWLEDGE_UNITS


if __name__ == "__main__":
    save_knowledge_units()