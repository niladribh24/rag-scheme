import fitz  # this is pymupdf

doc = fitz.open("mudra.pdf")

full_text = ""
for page_num, page in enumerate(doc):
    text = page.get_text()
    full_text += f"\n--- PAGE {page_num+1} ---\n{text}"

with open("mudra_extracted.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("Done. Extracted", len(doc), "pages.")