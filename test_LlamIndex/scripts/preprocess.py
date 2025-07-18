from llama_index.core import SimpleDirectoryReader
import pdfplumber
from docx import Document

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages)
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    return open(file_path).read()

documents = SimpleDirectoryReader(
    input_dir="data/raw/",
    file_extractor={".pdf": extract_text, ".docx": extract_text}
).load_data()
