import os
import torch
from transformers import BertModel, BertTokenizer
import numpy as np
from tqdm import tqdm
from database import db  # Import the database instance
import docx  # Import for DOCX support
import PyPDF2  # Import for PDF support

class DocumentEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        # Initialize BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode
        
        # Check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def generate_embedding(self, text):
        """Generate BERT embedding for a given text"""
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors='pt', 
                               padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Instead of just using the [CLS] token, average all token embeddings
        # This generally produces better results for document similarity
        token_embeddings = outputs.last_hidden_state
        
        # Create attention mask to properly handle padding
        attention_mask = inputs['attention_mask']
        
        # Expand attention mask to same dimensions as token_embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum the embeddings of tokens with actual content (not padding)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.sum(input_mask_expanded, 1)
        
        # Average the embeddings
        pooled_embeddings = sum_embeddings / sum_mask
        
        # Convert to numpy and normalize
        embedding = pooled_embeddings.cpu().numpy()[0]
        
        # Explicitly normalize to unit length (important for cosine similarity)
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def read_file_content(self, file_path):
        """Read content from various file formats"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            # Process text files (including .py files)
            if file_extension in ['.txt', '.md', '.csv', '.json', '.html', '.py', ''] or file_path.endswith('/'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
                    
            # Process DOCX files
            elif file_extension == '.docx':
                doc = docx.Document(file_path)
                full_text = []
                for para in doc.paragraphs:
                    full_text.append(para.text)
                return '\n'.join(full_text)
                
            # Process PDF files
            elif file_extension == '.pdf':
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text()
                return text
                
            else:
                print(f"Unsupported file extension: {file_extension}")
                return None
                
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def process_document(self, file_path):
        """Process a document file and store its embedding"""
        try:
            # Read the document using the appropriate method based on file type
            document_text = self.read_file_content(file_path)
            
            if document_text is None or document_text.strip() == "":
                print(f"No content extracted from {file_path}")
                return False
            
            # Store document in database and get its ID
            document_id = db.insert_document(file_path, document_text)
            
            # Generate embedding
            embedding = self.generate_embedding(document_text)
            
            # Convert numpy array to binary blob for storage
            embedding_blob = embedding.tobytes()
            
            # Store embedding in the database
            db.insert_embedding(document_id, embedding_blob)
            
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return False
    
    def process_directory(self, directory_path, file_extensions=None):
        """Process all documents in a directory"""
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.csv', '.json', '.html', '.docx', '.pdf', '.py', '']
            
        processed_count = 0
        failed_count = 0
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                # Process files with matching extensions or files with no extension
                if file_ext in file_extensions or ('' in file_extensions and file_ext == ''):
                    print(f"Processing: {file_path}")
                    
                    success = self.process_document(file_path)
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
        
        print(f"Processing complete. Successfully processed {processed_count} documents.")
        print(f"Failed to process {failed_count} documents.")