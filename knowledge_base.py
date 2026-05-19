import os
import tempfile

from docx import Document
import PyPDF2

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")


def load_uploaded_file(uploaded_file):
    filename = uploaded_file.filename.lower()

    if not filename.endswith(SUPPORTED_EXTENSIONS):
        raise ValueError("Unsupported file type. Use .txt, .md, .pdf, or .docx.")

    if filename.endswith((".txt", ".md")):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = os.path.join(tmpdir, uploaded_file.filename)
        uploaded_file.save(temp_path)

        if filename.endswith(".pdf"):
            return extract_text_from_pdf(temp_path)
        if filename.endswith(".docx"):
            return extract_text_from_docx(temp_path)

    raise ValueError("Could not read the uploaded file.")


def extract_text_from_pdf(path):
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n\n".join(text) if text else ""


def extract_text_from_docx(path):
    document = Document(path)
    return "\n\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
