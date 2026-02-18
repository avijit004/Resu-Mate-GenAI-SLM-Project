from pypdf import PdfReader
from docx import Document
import requests


def is_ollama_online() -> bool:
    try:
        return requests.get("http://localhost:11434").status_code == 200
    except Exception:
        return False


def extract_text(file) -> str:
    text = ""
    if file.type == "application/pdf":
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif (
        file.type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

