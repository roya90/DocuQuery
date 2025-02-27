# DocuQuery

DocuQuery is a Python-based document processing tool that enables users to extract and query text from PDF documents using large language models (LLMs). It provides functionalities for document parsing, text chunking, vector database operations, and natural language querying.

## Features
- Extract text from PDFs efficiently.
- Chunk extracted text for optimal processing.
- Store and retrieve document embeddings using a vector database.
- Query documents using LLMs to answer specific questions based on their content.

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Extract Text from a PDF
Use the `pdf_text_extractor.py` script to extract text from a given PDF file.
```bash
python pdf_text_extractor.py --file path/to/document.pdf
```

### 2. Chunk Extracted Text
Use the `text_chunker.py` script to divide extracted text into manageable chunks for further processing.
```bash
python text_chunker.py --input path/to/text.txt --output path/to/chunks.json
```

### 3. Store and Query Documents
Convert text chunks into embeddings and store them in a vector database using `vector_db_utils.py`.
```bash
python vector_db_utils.py --store path/to/chunks.json
```
To query the document:
```bash
python query_llm.py --question "What is the main topic of this document?"
```

### 4. Question Answering
Ask specific questions from the documents using `question_answering.py`.
```bash
python question_answering.py --file path/to/document.pdf --question "What is the key finding of this document?"
```

## Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`
- A configured LLM API key (for querying)

## Future Improvements
- Implement a web-based interface for document querying.
- Support for additional document formats (e.g., DOCX, TXT).
- Optimize text chunking for better LLM responses.

## Contributions
Contributions are welcome! Feel free to submit pull requests or open issues to improve the functionality.

## License
This project is licensed under the MIT License.

## Author
[Roya90](https://github.com/roya90)
