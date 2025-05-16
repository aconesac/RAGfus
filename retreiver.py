import numpy as np
import os
from embeder import DocumentEmbedder
from database import db  # Import the database instance

class DocumentRetriever:
    def __init__(self, query, db_path='document_embeddings.db'):
        """Initialize the DocumentRetriever with a query"""
        self.query = query
        self.embeder = DocumentEmbedder()
        # Note: We're using the centralized db instance instead of connecting directly

    def search_similar_documents(self, query_text, top_k=5, file_extensions=None, min_similarity=0):
        """Find most similar documents to a query text using cosine similarity
        
        Args:
            query_text (str): The text to search for
            top_k (int): Maximum number of results to return
            file_extensions (list): Optional list of file extensions to filter by (e.g. ['.pdf', '.docx'])
            min_similarity (float): Minimum similarity score (0-1) to include in results
        """
        # Generate embedding for the query
        query_embedding = self.embeder.generate_embedding(query_text)
        
        # Get all document embeddings from the database
        document_embeddings = db.get_all_document_embeddings()
        
        results = []
        for doc_id, file_path, document_text, embedding_blob in document_embeddings:
            # Apply file extension filter if specified
            if file_extensions:
                ext = os.path.splitext(file_path)[1].lower()
                if ext not in file_extensions:
                    continue
                    
            # Convert blob back to numpy array
            doc_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            
            # Ensure the document embedding is normalized
            if np.linalg.norm(doc_embedding) > 0:
                doc_embedding = doc_embedding / np.linalg.norm(doc_embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding)
            
            # Apply minimum similarity filter
            if similarity < min_similarity:
                continue
                
            results.append({
                'id': doc_id,
                'file_path': file_path,
                'similarity': similarity,
                'document_text': document_text[:200] + '...' if len(document_text) > 200 else document_text
            })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top_k results
        return results[:top_k]
