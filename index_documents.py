import os
import re
import docx
import PyPDF2
import psycopg2
from dotenv import load_dotenv
from google import genai
from pgvector.psycopg2 import register_vector

# Configuration and Environment Setup
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")

# Initialize the Google GenAI client for embedding generation
client = genai.Client(api_key=GEMINI_API_KEY)

def extract_text(file_path):
    """
    Extracts clean text content from PDF or DOCX files.
    
    Args:
        file_path (str): Path to the target file.
        
    Returns:
        str: Extracted text as a single string, or an empty string if failed.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return ""
    
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    try: # Check file extension and extract text accordingly
        if ext == ".pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = " ".join([page.extract_text() or "" for page in reader.pages])
        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            print(f"Unsupported file format: {ext}")
            return ""
            
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

import re

def chunk_text(text, strategy="paragraph", chunk_size=800, overlap=100):
    """
    Splits the input text into manageable segments (chunks) based on a selected strategy.
    
    This function performs initial text normalization to handle inconsistent whitespace 
    and line breaks often found in PDF extractions. It ensures that the resulting 
    chunks are contextually rich for high-quality vector embeddings.

    Args:
        text (str): The raw text extracted from the source document.
        strategy (str): The splitting logic to apply. Options are:
            - 'fixed': Creates chunks of a specific character length with overlap.
            - 'sentence': Splits text at sentence boundaries using regex.
            - 'paragraph': Groups text into logical blocks (default for RAG).
        chunk_size (int): The maximum character length for the 'fixed' strategy.
        overlap (int): The number of characters to overlap between adjacent fixed-size 
            chunks to maintain semantic continuity.

    Returns:
        list[str]: A list of text chunks ready for embedding.
    """
    if not text:
        return []

    # Text Normalization:
    # Replaces multiple whitespace characters and newlines with a single space.
    # This is critical for PDFs where sentences may be broken across multiple lines.
    normalized_text = " ".join(text.split())

    chunks = []

    # Strategy 1: Fixed-size with Overlap
    # Ideal for keyword-heavy documents where context is evenly distributed.
    if strategy == "fixed":
        start = 0
        while start < len(normalized_text):
            end = start + chunk_size
            chunk = normalized_text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            # Advance the pointer by (size - overlap) to maintain continuity
            start += (chunk_size - overlap)
            # Prevent infinite loops if overlap settings are misconfigured
            if overlap >= chunk_size: break

    # Strategy 2: Sentence-based Splitting
    # Uses regular expressions to identify end-of-sentence punctuation (. ! ?)
    # followed by a space, ensuring that chunks do not break mid-sentence.
    elif strategy == "sentence":
        sentences = re.split(r'(?<=[.!?])\s+', normalized_text)
        chunks = [s.strip() for s in sentences if len(s.strip()) > 10]

    # Strategy 3: Paragraph-based Splitting
    # For normalized text, this creates logical blocks of text (approx. 800 chars).
    # This maintains the best balance between context and embedding efficiency.
    elif strategy == "paragraph":
        # Logical block size for semantic grouping
        block_size = 800 
        for i in range(0, len(normalized_text), block_size):
            chunk = normalized_text[i:i + block_size].strip()
            if len(chunk) > 20:
                chunks.append(chunk)

    return chunks

def get_embedding(text):
    """
    Generates a 3072-dimensional vector embedding for a given text chunk.
    Uses Google's 'gemini-embedding-001'.
    """
    try:
        result = client.models.embed_content(
            model="models/gemini-embedding-001", 
            contents=text,
            config={'task_type': 'RETRIEVAL_DOCUMENT'}
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"Embedding failed for chunk: {text[:30]}... | Error: {e}")
        return None

def save_to_db(chunks, embeddings, filename, strategy_name):
    """
    Stores processed chunks and their embeddings into the PostgreSQL vector database.
    Ensures data matches the required schema: id, chunk_text, embedding, filename, strategy_split.
    """
    try:
        conn = psycopg2.connect(POSTGRES_URL)
        register_vector(conn) 
        cur = conn.cursor()
        
        # Clean up existing data for this specific file to avoid duplication
        cur.execute("DELETE FROM document_vectors WHERE filename = %s", (filename,))
        
        # Map strategy key to formal name for storage
        strategy_display_names = {
            "fixed": "Fixed-size with overlap",
            "sentence": "Sentence-based splitting",
            "paragraph": "Paragraph-based splitting"
        }
        formal_strategy = strategy_display_names.get(strategy_name, strategy_name)

        count = 0
        for chunk, emb in zip(chunks, embeddings):
            if emb:
                cur.execute(
                    """INSERT INTO document_vectors 
                       (chunk_text, embedding, filename, strategy_split) 
                       VALUES (%s, %s, %s, %s)""",
                    (chunk, emb, filename, formal_strategy)
                )
                count += 1
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"Success! {count} chunks indexed using '{formal_strategy}'.")
    except Exception as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    # Define file and strategy
    file_name = "Spec.pdf"  # Works with .pdf or .docx
    chosen_strategy = "paragraph" # Options: "fixed", "sentence", "paragraph"
    
    print(f"--- Starting Pipeline for: {file_name} ---")
    
    
    # Extract
    raw_text = extract_text(file_name)
    
    if raw_text:
        # Chunk
        text_chunks = chunk_text(raw_text, strategy=chosen_strategy)
        print(f"Step 1: Created {len(text_chunks)} chunks using '{chosen_strategy}' strategy.")
        
        # Embed
        print("Step 2: Generating Embeddings...")
        vectors = [get_embedding(c) for c in text_chunks]
        
        # Load
        print("Step 3: Saving to PostgreSQL...")
        save_to_db(text_chunks, vectors, file_name, chosen_strategy)
    else:
        print("Pipeline failed: Document could not be processed")