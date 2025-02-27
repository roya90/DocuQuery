from transformers import AutoTokenizer
import spacy
import re

try:
    SPACY_NLP = spacy.load("en_core_web_sm")
    TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
except Exception as e:
    raise RuntimeError(f"Failed to load models: {e}")

def is_meaningful(sentence, threshold=5):
    sentence = sentence.strip()
    if len(sentence) < threshold:
        return False
    if re.fullmatch(r"[\W\d_]+", sentence):
        return False
    return True

def validate_text_input(text, max_length=1_000_000):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    if len(text) > max_length:
        raise ValueError("Input text is too large to process.")
    return text.strip()

def smart_chunk_spacy_by_paragraph(text):
    text = validate_text_input(text)
    paragraphs = [para.strip() for para in text.split("\n") if is_meaningful(para)]
    return paragraphs

def smart_chunk_spacy(text):
    text = validate_text_input(text)
    doc = SPACY_NLP(text)
    sentences = [sent.text for sent in doc.sents if is_meaningful(sent.text)]
    return sentences

def smart_chunk_spacy_advanced(text, min_chunk_length=50, max_chunk_length=500):
    text = validate_text_input(text)
    raw_paragraphs = re.sub(r"\n{2,}", "\n\n", text).split("\n\n")
    refined_paragraphs = []
    for paragraph in raw_paragraphs:
        if len(paragraph.strip()) < min_chunk_length:
            continue
        doc = SPACY_NLP(paragraph)
        current_chunk = []
        current_length = 0
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if current_length + len(sent_text) > max_chunk_length:
                refined_paragraphs.append(" ".join(current_chunk).strip())
                current_chunk = []
                current_length = 0
            current_chunk.append(sent_text)
            current_length += len(sent_text)
        if current_chunk:
            refined_paragraphs.append(" ".join(current_chunk).strip())
    return refined_paragraphs

def smart_chunk_transformers(text, max_tokens=128):
    text = validate_text_input(text)
    tokens = TOKENIZER(text, truncation=False, return_tensors="pt")
    chunks = [text[i:i+max_tokens] for i in range(0, len(tokens['input_ids'][0]), max_tokens)]
    return chunks