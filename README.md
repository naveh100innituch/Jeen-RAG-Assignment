# AI-Powered Semantic Document Search (RAG)

This project implements a high-performance **RAG (Retrieval-Augmented Generation)** pipeline. It allows users to index PDF documents into a **PostgreSQL** database using **Vector Embeddings** from Google Gemini and perform precise semantic searches based on context rather than just keywords.

---

## Key Features
- **PDF or DOCX Text Extraction**: Efficiently extracts and cleans text from PDF or DOCX files.
- **Smart Semantic Chunking**: Implements a paragraph-based splitting strategy to ensure each chunk maintains its original context.
- **High-Dimensional Embeddings**: Utilizes Google's `gemini-embedding-001` model (3072 dimensions) for superior semantic representation.
- **Vector Database**: Integrated with PostgreSQL and the `pgvector` extension for fast similarity searches using Cosine Distance.
- **Diagnostic Tools**: Includes utilities to verify API access and available models.

---

## Tech Stack
- **AI/LLM**: Google Gemini API (`google-genai`)
- **Database**: PostgreSQL with `pgvector`
- **Language**: Python 3.10+
- **Key Libraries**: `psycopg2-binary`, `PyPDF2`, `python-dotenv`, `pgvector`

---

## Installation & Setup

### 1. Prerequisites
- **PostgreSQL**: Ensure PostgreSQL is installed and the `pgvector` extension is enabled.
- **Google AI API Key**: Generate a key from [Google AI Studio](https://aistudio.google.com/).

### 2. Clone, Install and usage
```bash
# Clone the repository
git clone <your-repository-url>
cd <repository-folder>

# Install dependencies
pip install -r requirements.txt

## Environment Configuration
Create a .env file in the root directory and add your credentials:
GEMINI_API_KEY=your_actual_api_key_here
POSTGRES_URL=postgresql://postgres:password@localhost:5432/postgres

## Database Initialization
*Run the following SQL commands in your pgAdmin Query Tool to set up the vector-enabled table*
DROP TABLE IF EXISTS document_vectors;

CREATE TABLE document_vectors (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT,
    embedding VECTOR(3072),
    filename TEXT,
    strategy_split TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

## Usage Guide
*Step 1: Verify API Access*
**Check which models are accessible to your API key:**
python check_models.py

*Step 2: Index a Document*
**Place your PDF in the project folder and run the indexing script:**
python index_documents.py

*Step 3: Semantic Search*
**Query the database using natural language:**
python test_search.py
*Example Question: "Which social networks are being monitored?" Result: Returns the most relevant paragraphs from the document with a similarity confidence score.*

## Database Verification (pgAdmin)
*You can monitor the stored vectors directly in pgAdmin 4:*

*Count records:*
 SELECT count(*) FROM document_vectors;

*Review content:*
 SELECT * FROM document_vectors LIMIT 5;
