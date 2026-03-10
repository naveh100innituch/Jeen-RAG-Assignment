import os
import docx
import PyPDF2
import psycopg2
from dotenv import load_dotenv
from google import genai
from pgvector.psycopg2 import register_vector

load_dotenv() # Configuration and Environment Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")

client = genai.Client(api_key=GEMINI_API_KEY) # Initialize the Google GenAI client for embedding generation

def extract_text(file_path):
    """
    Extracts raw text content from a PDF file.
    
    Args:
        file_path (str): Path to the target PDF file.
        
    Returns:
        str: Extracted text as a single string, or an empty string if failed.
    """
    if not os.path.exists(file_path): return ""
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f) 
            # Aggregate text from all pages in the PDF
            text = " ".join([page.extract_text() or "" for page in reader.pages])
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def chunk_text(text, min_length=100, max_length=800):
    """
    Splits long text into smaller, meaningful chunks to maintain context for embeddings.
    
    This strategy merges lines into paragraphs until they reach a 'max_length',
    ensuring that chunks are not too small to lose semantic meaning.
    
    Args:
        text (str): The raw text to split.
        min_length (int): Minimum characters for the final chunk.
        max_length (int): Target maximum characters per chunk.
        
    Returns:
        list: A list of text chunks (strings).
    """

    if not text: return []
    
    lines = text.split('\n')
    chunks = []
    current_chunk = ""

    for line in lines:
        clean_line = line.strip()
        if not clean_line: continue
        
        current_chunk += clean_line + " "
        
        if len(current_chunk) > max_length: # Once the chunk exceeds max_length, store it and start a new one
            chunks.append(current_chunk.strip())
            current_chunk = ""
    # Include the final chunk if it meets the minimum length requirement
    if len(current_chunk) >= min_length:
        chunks.append(current_chunk.strip())
        
    return chunks

def get_embedding(text):
    """
    Generates a 3072-dimensional vector embedding for a given text chunk.
    
    Uses Google's 'gemini-embedding-001' model optimized for document retrieval.
    
    Args:
        text (str): The text chunk to embed.
        
    Returns:
        list: A list of floats representing the embedding, or None if failed.
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

def save_to_db(chunks, embeddings, filename):
    """
    Stores processed chunks and their embeddings into the PostgreSQL vector database.
    
    Uses 'pgvector' to store and index high-dimensional vectors.
    Includes a cleanup step to prevent duplicate entries for the same file.
    
    Args:
        chunks (list): List of text chunks.
        embeddings (list): List of corresponding vector embeddings.
        filename (str): Source filename for metadata tracking.
    """
    conn = psycopg2.connect(POSTGRES_URL)
    register_vector(conn) # Essential for psycopg2 to recognize the 'vector' type
    cur = conn.cursor() 
    # Clean up existing data for this specific file to avoid duplication
    cur.execute("DELETE FROM document_vectors WHERE filename = %s", (filename,))
    # 2. 
    #Batch insert chunks and embeddings
    count = 0
    for chunk, emb in zip(chunks, embeddings):
        if emb:
            cur.execute(
                "INSERT INTO document_vectors (chunk_text, embedding, filename, strategy_split) VALUES (%s, %s, %s, %s)",
                (chunk, emb, filename, "semantic_paragraph")
            )
            count += 1
    
    conn.commit()
    cur.close()
    conn.close()
    print(f"Success! {count} chunks with embeddings saved to database.")

if __name__ == "__main__":
    file_name = "Spec.pdf"  
    print(f"--- Starting Process for: {file_name} ---")
    
    raw_text = extract_text(file_name) # Extract text from document
    if raw_text: # Divide text into semantic chunks
        chunks = chunk_text(raw_text)
        print(f"Step 1: Created {len(chunks)} high-quality chunks.")
        # Call Gemini API to generate embeddings
        print("Step 2: Generating Embeddings (this calls Gemini API)...")
        vectors = [get_embedding(c) for c in chunks]
        # Persist everything in PostgreSQL
        print("Step 3: Saving to DB...")
        save_to_db(chunks, vectors, file_name)
    else:
        print("Could not read PDF.")