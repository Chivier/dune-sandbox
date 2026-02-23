import json
import os

INPUT_DIR = os.path.join(os.path.dirname(__file__), "law")


def convert_gdpr(data, source):
    """{"Article 1": {"title": "...", "content": "..."}, ...}"""
    records = []
    for article, body in data.items():
        title = body.get("title", "")
        content = body.get("content", "")
        law_text = f"{source} {article}: {title}\n{content}"
        records.append({
            "law": law_text,
            "metadata": {
                "source": source,
                "article": article,
                "title": title,
            }
        })
    return records


def convert_hippa(data, source):
    """{"160.101": ["text line 1", "text line 2", ...], ...}"""
    records = []
    for section, lines in data.items():
        content = "\n".join(lines)
        law_text = f"{source} § {section}\n{content}"
        records.append({
            "law": law_text,
            "metadata": {
                "source": source,
                "section": section,
            }
        })
    return records


CONVERTERS = {
    "GDPR.json": ("GDPR", convert_gdpr),
    "Hippa.json": ("HIPAA", convert_hippa),
}

for filename, (source, converter) in CONVERTERS.items():
    path = os.path.join(INPUT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = converter(data, source)
    output_file = os.path.join(INPUT_DIR, f"{source}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {output_file}")
