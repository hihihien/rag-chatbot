print("PDF Parsing Started")

import fitz  
import re
import json
from pathlib import Path

PDF_PATH = "data/MMI_MHB_PO2025_V1.0.pdf"
OUTPUT_PATH = "data/modules.json"

#read input PDF
doc = fitz.open(PDF_PATH)
all_text = ""
for page in doc:
    all_text += page.get_text()

#trim to the start of each first module
start_index = re.search(r"MMI\s?0?1\s?-", all_text)
if start_index:
    all_text = all_text[start_index.start():]
else:
    print("Warning: Could not find starting module header. Using entire text.")

#preview texts of each module
print("\n=== Preview of trimmed text ===\n")
print(all_text[:1500])
print("\n=== End of preview ===\n")

#regex to extract full modules
pattern = re.compile(
    r"(MMI\s?\d{2}(?:\.\d{2})?\s?-.*?)(?=MMI\s?\d{2}(?:\.\d{2})?\s?-|\Z)",
    re.DOTALL
)
matches = pattern.findall(all_text)
print(f"Found {len(matches)} raw module chunks")

#clean up structure
modules = []
for raw in matches:
    header_match = re.match(r"(MMI\s?\d{2}(?:\.\d{2})?)\s?-\s?(.*)", raw.strip())
    if not header_match:
        continue

    module_id = header_match.group(1).strip().replace(" ", "")
    title = header_match.group(2).split("\n")[0].strip()

    modules.append({
        "module_id": module_id,
        "title": title,
        "content": raw.strip()
    })

#save to JSON
Path("data").mkdir(exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(modules, f, indent=2, ensure_ascii=False)

print(f"Extracted and saved {len(modules)} modules to {OUTPUT_PATH}")
