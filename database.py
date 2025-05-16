import sqlite3
import os

class Database:
    def __init__(self, db_path='document_embeddings.db'):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Create database and tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            file_path TEXT UNIQUE,
            document_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create embeddings table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            document_id INTEGER,
            embedding BLOB,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_document(self, file_path, document_text):
        """Insert a document into the database and return its ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store document in database
        cursor.execute(
            "INSERT OR IGNORE INTO documents (file_path, document_text) VALUES (?, ?)",
            (file_path, document_text)
        )
        conn.commit()
        
        # Get document ID
        cursor.execute("SELECT id FROM documents WHERE file_path = ?", (file_path,))
        document_id = cursor.fetchone()[0]
        
        conn.close()
        return document_id
    
    def insert_embedding(self, document_id, embedding_blob):
        """Insert an embedding for a document into the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store embedding
        cursor.execute(
            "INSERT OR REPLACE INTO embeddings (document_id, embedding) VALUES (?, ?)",
            (document_id, embedding_blob)
        )
        conn.commit()
        conn.close()
    
    def get_all_document_embeddings(self):
        """Get all documents and their embeddings from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT d.id, d.file_path, d.document_text, e.embedding 
            FROM documents d
            JOIN embeddings e ON d.id = e.document_id
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def clear_database(self):
        """Delete all records from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete all records from embeddings table first (due to foreign key constraint)
        cursor.execute("DELETE FROM embeddings")
        
        # Delete all records from documents table
        cursor.execute("DELETE FROM documents")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print("Database cleared successfully.")
        return True
    
    def delete_document(self, document_id):
        """Delete a document and its embedding from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete embedding first (due to foreign key constraint)
        cursor.execute("DELETE FROM embeddings WHERE document_id = ?", (document_id,))
        
        # Delete document
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        
        # Commit changes
        conn.commit()
        deleted_count = conn.total_changes
        conn.close()
        
        return deleted_count > 0

# Initialize the database when the module is imported
db = Database()