# **Document Question-Answering System with Explainability**

This repository contains a robust pipeline for a document question-answering system that utilizes AI to extract, process, and query documents. The system ensures explainability, relevance checking, and compliance with predefined guardrails for safe and responsible AI usage.

---

## **Features**

1. **PDF Extraction**: Extracts text from legal documents using `PyMuPDF` (fitz).
2. **Smart Chunking**: Processes text into meaningful paragraphs using `spaCy` and filters out irrelevant content.
3. **Embedding Generation**: Generates semantic embeddings for text chunks using `SentenceTransformer` models.
4. **Vector Database with FAISS**: Efficiently stores and retrieves text chunks based on cosine similarity.
5. **Relevance Checking**: Ensures only relevant chunks are queried, with customizable thresholds.
6. **LLM Integration**: Leverages Gemini Flash or other text-generation models to answer user questions.
7. **Guardrails for Restricted Topics**:
   - Blocks queries involving predefined restricted topics (e.g., politics, confidential information).
   - Responds with appropriate messages for unrelated or restricted queries.
8. **Explainability**: Outputs relevant text chunks, including metadata like section and page numbers, to justify answers.

---

## **Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/roya90/DocuQuery.git
   cd DocuQuery


2. **Set Up a Virtual Environment**:
   ```bash
   
   #On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate 

   #On Windows:
   python -m venv venv
   venv\Scripts\activate 
    

3. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt

4. **Set Up Gemini Flash**:

    Authenticate your Gemini Flash:
    ```bash
    pip install --upgrade google-genai
    gcloud auth application-default login



Add your Gemini project and location to the environment: 
    ```bash
    echo 'export GOOGLE_PROJECT=<your_google_project_id>' >> ~/.zshrc                            
    echo 'export GOOGLE_LOCATION=<your_google_project_location>' >> ~/.zshrc
    source ~/.zshrc
    
---
## **Usage**
**Run the System**:
```bash 
python question-answering.py <path_to_pdf> "<query_text>"
```

**Examples:**
```bash 
python question-answering.py "document.pdf" "can I return a swimsuit?"
```
---
## **Project Structure**
```
├── pdf_text_extractor.py         # Extracts text from PDF documents
├── text_chunker.py               # Splits text into meaningful chunks (paragraphs)
├── vector_db_utils.py            # Handles FAISS indexing and querying
├── query_llm.py                  # Interacts with the LLM for answering questions
├── question-answering.py         # Main script for running the pipeline
├── requirements.txt              # Dependencies for the project
└── README.md                     # Documentation
```
---
## **Pipeline Workflow**

1. Extract Text:

    - Extract text from the PDF.
2. Chunking:

    - Split the text into meaningful paragraphs, filtering irrelevant data.
3. Embedding and Indexing:

    - Generate semantic embeddings for the text chunks.
    - Store embeddings in FAISS for efficient retrieval.
4. Query Handling:

    - Check if the query relates to restricted topics.
    - If valid, retrieve the most relevant chunks using FAISS.

5. Answer Generation:

    - Use an LLM (e.g., Gemini Flash) to generate answers based on the retrieved chunks.
    - Include metadata (e.g., section) in the output for explainability.

---
## **Known Issues**
**Restricted Query Handling:**

Current topic filtering is keyword-based. Enhancements like semantic topic detection can improve accuracy.

**Large PDF Handling:**

Processing very large documents may require additional optimization (e.g., batching chunks).

---
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