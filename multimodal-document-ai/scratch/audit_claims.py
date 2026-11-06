import re
from pathlib import Path

search_terms = [
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bSOTA\b",
    r"\bsolved\b",
    r"\bperfect\b",
    r"\b100%\b",
    r"\bgeneralizes\b",
    r"\bgeneral document understanding\b",
    r"\bfull CORD KIE\b",
    r"\bstatistically significant\b",
]

report_files = sorted(list(Path("reports").glob("*.md")) + list(Path("docs").glob("*.md")))
findings = []

for p in report_files:
    content = p.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()
    for line_idx, line in enumerate(lines):
        for pat in search_terms:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                term = m.group(0)
                findings.append({
                    "file": str(p.as_posix()),
                    "line_num": line_idx + 1,
                    "term": term,
                    "text": line.strip()[:140],
                })

print(f"Total claim occurrences found: {len(findings)}")
import json
with open("scratch/claims_findings.json", "w", encoding="utf-8") as f:
    json.dump(findings, f, indent=2)

for f in findings[:25]:
    print(f"{f['file']}:{f['line_num']} [{f['term']}]: {f['text']}")
