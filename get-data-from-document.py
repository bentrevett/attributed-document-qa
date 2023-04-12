import pdfplumber
import json

def extract_text_from_pdf(path):
    data = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            for j, line in enumerate(text.split("\n"), start=1):
                if line:
                    data.append({
                        "text": str(line),
                        "page": i,
                        "line": j,
                    })
    return data

data = extract_text_from_pdf("reportlab-sample.pdf")

with open("data.jsonl", "w") as f:
    for example in data:
        f.write(f"{json.dumps(example)}\n")
