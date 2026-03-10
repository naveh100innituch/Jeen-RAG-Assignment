# Document Vectorization & Semantic Search

This project implements a RAG (Retrieval-Augmented Generation) pipeline that extracts text from PDFs, generates semantic embeddings using Google Gemini, and stores them in a PostgreSQL database with `pgvector`.

## Features
- **Smart Chunking**: Splits documents into meaningful paragraphs.
- **Vector Search**: Performs semantic similarity search using Cosine Distance.
- **Database**: Integration with PostgreSQL and pgvector.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt