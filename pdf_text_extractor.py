import logging
from pathlib import Path
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file() or not pdf_path.suffix.lower() == '.pdf':
            raise ValueError("The provided file is not a valid PDF.")

        text = ""
        with fitz.open(pdf_path) as pdf_document:
            for page_num in range(len(pdf_document)):
                text += pdf_document[page_num].get_text()
        return text

    except FileNotFoundError:
        logging.error("The specified PDF file was not found.")
        return None
    except fitz.FileDataError:
        logging.error("The PDF file is corrupted or unreadable.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        return None