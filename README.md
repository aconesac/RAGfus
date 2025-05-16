# RAGfus

A Retrieval-Augmented Generation (RAG) system for semantic document search and embedding management.

## Features

- Document processing and embedding generation using BERT
- Support for multiple file formats (PDF, DOCX, TXT, Python files)
- Semantic search with filtering by file type and similarity threshold
- Web-based user interface for document management
- Document preview functionality

## Installation

### Requirements

- Python 3.8+
- Flask
- PyTorch
- Transformers (Hugging Face)
- SQLite3
- python-docx
- PyPDF2

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RAGfus.git
cd RAGfus
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The application will be available at http://localhost:5000

## Usage

### Web Interface

The web interface provides the following functionality:
- Upload individual documents
- Process entire directories
- Search documents semantically
- Preview document content
- Manage documents (delete, view)

### API Endpoints

- `POST /upload`: Upload and process a document
- `POST /process_directory`: Process all documents in a directory
- `POST /search`: Find documents semantically similar to a query
- `GET /documents`: List all documents in the database
- `GET /documents/{id}/preview`: Preview document content
- `DELETE /documents/{id}`: Delete a document

## How It Works

RAGfus uses BERT embeddings to represent documents semantically. When a document is uploaded or processed, the system:

1. Extracts text content from the document
2. Generates embeddings using a BERT model
3. Stores the document and its embedding in a SQLite database

When searching, RAGfus:
1. Generates an embedding for the search query
2. Computes similarity between the query and all documents
3. Returns the most similar documents based on cosine similarity

## License

MIT

## Contributors

- Agustín Conesa