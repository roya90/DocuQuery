import os
import sys
import numpy as np
from pdf_text_extractor import extract_text_from_pdf
from query_llm import query_flash # Uses the Gemini/Vertex AI version
from text_chunker import smart_chunk_spacy_advanced  # Or your preferred chunking method
from vector_db_utils import generate_embeddings, store_in_faiss, load_faiss_index, query_faiss_index

def main(pdf_path, query_text, relevance_threshold=np.float32(0.6)):
    """
    Main function to perform question answering on a PDF document.

    Args:
        pdf_path (str): Path to the PDF file.
        query_text (str): The question to ask.
        relevance_threshold (float): Minimum similarity score for a chunk to be considered relevant.
    """

    print("\nExtracting text from the PDF...")
    try:
        extracted_text = extract_text_from_pdf(pdf_path)
        if not extracted_text:
            print("Error: Failed to extract text from the PDF. The document might be empty or unreadable.")
            sys.exit(1)
    except Exception as e:
        print(f"Error: An exception occurred while extracting text: {e}")
        sys.exit(1)

    print("\nChunking the extracted text...")
    try:
        chunks = smart_chunk_spacy_advanced(extracted_text)  # Use the advanced chunker
        if not chunks:
            print("Error: Failed to create meaningful text chunks. The document might not contain sufficient valid content.")
            sys.exit(1)
        print(f"Text successfully chunked into {len(chunks)} chunks.")
    except Exception as e:
        print(f"Error: An exception occurred while chunking the text: {e}")
        sys.exit(1)

    print("\nGenerating embeddings for the chunks...")
    try:
        embeddings = generate_embeddings(chunks)
        print("Embeddings successfully generated.")
    except Exception as e:
        print(f"Error: Failed to generate embeddings: {e}")
        sys.exit(1)

    print("\nStoring embeddings in FAISS index...")
    try:
        index = store_in_faiss(embeddings, db_file="vector_db_cosine.index")
        print("FAISS index successfully created and stored.")
    except Exception as e:
        print(f"Error: Failed to store embeddings in FAISS index: {e}")
        sys.exit(1)

    print("\nLoading FAISS index for querying...")
    try:
        index = load_faiss_index("vector_db_cosine.index")
        print("FAISS index successfully loaded.")
    except Exception as e:
        print(f"Error: Failed to load FAISS index: {e}")
        sys.exit(1)

    print(f"\nQuerying FAISS index with the text: '{query_text}'...")
    try:
        results = query_faiss_index(query_text, index, chunks, top_k=5)  # Retrieve top 5
        # Filter results by relevance threshold
        
        context_chunks = [(text, dis, idx) for text, dis, idx in results if dis > relevance_threshold]
        print("\nResults from FAISS index:")
        for idx, (text, score, doc_id) in enumerate(results):
            print(f"{idx}. Score: {score:.4f}, Document ID: {doc_id}\n{text[:200]}...\n")

        print(f"Found {len(context_chunks)} relevant context chunks based on the threshold.")
    except Exception as e:
        print(f"Error: Failed to query the FAISS index: {e}")
        sys.exit(1)

    print("\nQuerying the Gemini model for an answer...")
    try:
        answer = query_flash(query_text, context_chunks)  # Call the Gemini-powered query_llm
        print("Query successfully processed by the Gemini model.")
    except Exception as e:
        print(f"Error: Failed to query the Gemini model: {e}")
        sys.exit(1)

    print("\nGenerated Answer:")
    print(answer["answer"])
    print("\nCited Context:")
    for text, _, idx in answer["relevant_context"]:
        print(f"\tSource {idx}: {text}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python question_answering.py <path_to_pdf> \"<query_text>\"")
        sys.exit(1)

    pdf_path = sys.argv[1]
    query_text = sys.argv[2]
    main(pdf_path, query_text)


