import os
import psycopg2
from dotenv import load_dotenv
from google import genai
from pgvector.psycopg2 import register_vector

load_dotenv() # Load configuration from environment variables
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
POSTGRES_URL = os.getenv("POSTGRES_URL") # Database connection string from environment variables

def semantic_search(query_text, limit=3):
    """
    Performs a semantic similarity search against the vector database.
    
    Args:
        query_text (str): The user's natural language question or search term.
        limit (int): The number of top relevant results to return.
        
    Returns:
        list: A list of tuples containing (chunk_text, filename, similarity_score).
    """
    # Generate a vector embedding for the input query
    # We must use the same model used during the indexing phase (gemini-embedding-001)
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=query_text
    )
    query_embedding = result.embeddings[0].values

    conn = psycopg2.connect(POSTGRES_URL) # Establish connection to PostgreSQL
    register_vector(conn) # Register pgvector type with psycopg2 to handle vector operations
    cur = conn.cursor()
    
    # Define the similarity search query
    # We use the '<=>' operator for Cosine Distance.
    # '1 - distance' converts the distance into a Similarity Score (where 1.0 is a perfect match).
    # We cast the input array explicitly using '::vector' to match the database type.

    search_query = """
    SELECT chunk_text, filename, 1 - (embedding <=> %s::vector) AS similarity
    FROM document_vectors
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """
    
    cur.execute(search_query, (query_embedding, query_embedding, limit)) # Execute the search using the query embedding
    results = cur.fetchall()
    
    cur.close() # Clean up resources
    conn.close()
    return results

if __name__ == "__main__":
    print("--- Semantic Search Test ---")
    question = input("What do you want to know? ")
    
    try:
        matches = semantic_search(question)
        print("\n--- Results Found ---")
        for i, (text, fname, score) in enumerate(matches, 1): # Similarity scores help verify the quality of the retrieval
            print(f"Match #{i} (Score: {score:.2%})")
            print(f"Text: {text}")
            print("-" * 30)
    except Exception as e:
        print(f"Error during search: {e}")