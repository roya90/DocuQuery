import os
from google import genai
from google.genai import types

# Configure the API key and other credentials if not already set as environment variables

project =  os.environ["GOOGLE_PROJECT"] 
location = os.environ["GOOGLE_LOCATION"]


def query_flash(question = "What is APR?", context_chunks = [], model_name="gemini-2.0-flash-001", top_k=3):
    """
    Queries a Large Language Model (LLM) with a question and relevant context chunks.

    Args:
        question (str): The question to be answered.
        context_chunks (list): A list of tuples, each containing (text_chunk, similarity_score, index).
        model_name (str): The name of the Gemini model to use. Defaults to "gemini-2.0-flash-001".
        top_k (int): The number of top context chunks to use. Defaults to 3.

    Returns:
        dict: A dictionary containing the generated answer and the relevant context chunks.
              Returns an error message as a string in case of exceptions.
    """
    try:
        # Input validation
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The question must be a non-empty string.")
        if not isinstance(context_chunks, list) or not all(
            isinstance(chunk, (list, tuple)) for chunk in context_chunks
        ):
            raise ValueError("Context chunks must be a list of tuples or lists.")

        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )

        # Construct the context
        context = " ".join([context_chunk[0] for context_chunk in context_chunks])

        # Prompt Engineering
        prompt = (
            f"You are a legal assistant specializing in contracts. "
            f"Answer the question based on the following context, and cite the sources explicitly. "
            f"Do not include any information not present in the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        model = "gemini-2.0-flash-001"
        contents = [prompt
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature = 1,
            top_p = 0.95,
            max_output_tokens = 8192,
            response_modalities = ["TEXT"],
            safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
            )],
        )
                
            

        # Generate content (streaming)
        response_stream = client.models.generate_content_stream(model = model,
                    contents = contents,
                    config = generate_content_config,)

        # Collect the streamed response
        generated_answer = ""
        for chunk in response_stream:
            generated_answer += chunk.text

        relevant_chunks = context_chunks[:top_k]  # Top k chunks for citation
        return {"answer": generated_answer.strip(), "relevant_context": relevant_chunks}

    except google_exceptions.GoogleAPIError as e:
        return f"Google API Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


if __name__ == "__main__":
    # Example usage (replace with your actual data)
    question = "What is the main topic of this document?"
    context_chunks = [("This document discusses the importance of AI in healthcare.", 0.8, 1),
                      ("AI can improve diagnostic accuracy and treatment outcomes.", 0.7, 2),
                      ("Ethical considerations are crucial when implementing AI in healthcare.", 0.6, 3)]

    result = query_flash(question, context_chunks)
    print(result)

