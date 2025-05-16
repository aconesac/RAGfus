from database import db
from embeder import DocumentEmbedder
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebuild_all_embeddings():
    """Rebuild embeddings for all documents in the database using the improved embedding method"""
    embeder = DocumentEmbedder()
    
    # Get all documents from the database
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, file_path, document_text FROM documents")
    documents = cursor.fetchall()
    conn.close()
    
    logger.info(f"Found {len(documents)} documents to reprocess")
    
    # Clear existing embeddings
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM embeddings")
    conn.commit()
    conn.close()
    
    logger.info("Cleared existing embeddings")
    
    # Rebuild embeddings
    success_count = 0
    error_count = 0
    
    for doc_id, file_path, document_text in documents:
        try:
            logger.info(f"Processing document {doc_id}: {file_path}")
            
            # Generate new embedding
            embedding = embeder.generate_embedding(document_text)
            
            # Convert to binary blob
            embedding_blob = embedding.tobytes()
            
            # Store embedding
            db.insert_embedding(doc_id, embedding_blob)
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            error_count += 1
    
    logger.info(f"Embedding rebuild complete. Successful: {success_count}, Failed: {error_count}")
    
if __name__ == "__main__":
    rebuild_all_embeddings()