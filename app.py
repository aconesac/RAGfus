from embeder import DocumentEmbedder
from retreiver import DocumentRetriever
from database import db  # Import the database instance
from flask import Flask, request, jsonify
import os
import json
import logging
import sqlite3  # Added SQLite import
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask import send_from_directory
from flask import render_template
from flask import redirect  
import numpy as np  # Import numpy for handling float32 conversion


# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx', 'csv', 'py', ''}  # Added 'py' and '' (no extension)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize DocumentEmbedder and DocumentRetriever
embeder = DocumentEmbedder()
retriever = DocumentRetriever(query=None)  # No longer need to pass db_path

# Function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to save the uploaded file
def save_file(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return file_path
    return None

# Function to process the uploaded file and generate embeddings
def process_file(file_path):
    # We can now use the embeder's process_document method directly
    success = embeder.process_document(file_path)
    
    if success:
        return {"status": "success", "file_path": file_path}
    else:
        return {"status": "error", "file_path": file_path}

# Routes
@app.route('/', methods=['GET'])
def index():
    """Render the main web interface page"""
    return render_template('index.html')

@app.route('/documents', methods=['GET'])
def list_documents():
    """Get a list of all documents in the database"""
    try:
        conn = db.db_path
        # Using SQLite connection to get all documents
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_path, created_at FROM documents ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        
        documents = []
        for row in rows:
            documents.append({
                'id': row[0],
                'file_path': row[1],
                'created_at': row[2]
            })
        
        return jsonify(documents)
    
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process a file to extract embeddings"""
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Create uploads directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file_path = save_file(file)
        if not file_path:
            return jsonify({"error": "File type not allowed"}), 400
        
        # Process the file and save to database
        result = process_file(file_path)
        return jsonify(result), 201
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/process_directory', methods=['POST'])
def process_dir():
    """Process all documents in a specified directory"""
    try:
        data = request.json
        if not data or 'directory_path' not in data:
            return jsonify({"error": "Directory path is required"}), 400
        
        directory_path = data['directory_path']
        file_extensions = data.get('file_extensions', ['.txt', '.md', '.csv', '.json', '.html', '.docx', '.pdf', '.py', ''])
        
        if not os.path.isdir(directory_path):
            return jsonify({"error": "Invalid directory path"}), 400
        
        # Use the embeder's process_directory method
        embeder.process_directory(directory_path, file_extensions)
        
        return jsonify({
            "status": "success", 
            "message": "Directory processing initiated"
        }), 200
    
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """Find documents similar to a query"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query_text = data['query']
        top_k = data.get('top_k', 5)
        file_extensions = data.get('file_extensions', None)
        min_similarity = data.get('min_similarity', 0.0)
        
        # Create a retriever with the query
        retriever_instance = DocumentRetriever(query=query_text)
        
        # Find similar documents with the new filter parameters
        results = retriever_instance.search_similar_documents(
            query_text, 
            top_k=top_k,
            file_extensions=file_extensions,
            min_similarity=min_similarity
        )
        
        # Convert NumPy float32 to native Python float to make it JSON serializable
        for result in results:
            if 'similarity' in result and isinstance(result['similarity'], np.float32):
                result['similarity'] = float(result['similarity'])
        
        return jsonify({"results": results}), 200
    
    except Exception as e:
        logger.error(f"Error searching similar documents: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the API"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api', methods=['GET'])
def api_info():
    """Show API information (renamed from previous index route)"""
    return jsonify({
        "name": "RAGfus API",
        "description": "API for document embedding and retrieval",
        "endpoints": [
            {"path": "/upload", "method": "POST", "description": "Upload and process a document"},
            {"path": "/process_directory", "method": "POST", "description": "Process all documents in a directory"},
            {"path": "/search", "method": "POST", "description": "Find documents similar to a query"},
            {"path": "/documents", "method": "GET", "description": "List all documents in the database"},
            {"path": "/health", "method": "GET", "description": "Health check endpoint"}
        ]
    }), 200

@app.route('/documents/<int:document_id>', methods=['DELETE'])
def delete_document(document_id):
    """Delete a document and its embedding from the database"""
    try:
        # Delete the document and its embedding
        success = db.delete_document(document_id)
        
        if success:
            return jsonify({"status": "success", "message": f"Document {document_id} deleted successfully"}), 200
        else:
            return jsonify({"error": f"Document {document_id} not found"}), 404
    
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/documents/<int:document_id>/preview', methods=['GET'])
def preview_document(document_id):
    """Get the preview of a document content"""
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        # Get document content
        cursor.execute("SELECT file_path, document_text FROM documents WHERE id = ?", (document_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return jsonify({"error": f"Document {document_id} not found"}), 404
            
        file_path, document_text = result
        
        # For very large documents, limit preview size
        max_preview_length = 5000
        preview = document_text[:max_preview_length]
        truncated = len(document_text) > max_preview_length
        
        return jsonify({
            "id": document_id,
            "file_path": file_path,
            "preview": preview,
            "truncated": truncated,
            "total_length": len(document_text)
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting document preview: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    # Make sure the uploads directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)