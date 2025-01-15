import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class EnhancedBookSearch:
    def __init__(self, persist_directory: str = "./book_vectors"):
        """Initialize the search engine with the given persistence directory."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.persist_directory = persist_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        
    async def process_book(self, pdf_path: str) -> None:
        """Process a PDF book and store its embeddings."""
        try:
            print(f"Loading PDF from {pdf_path}...")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            print("Splitting text into chunks...")
            chunks = self.text_splitter.split_documents(pages)
            print(f"Created {len(chunks)} chunks")
            
            print("Creating embeddings and storing in vector database...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("Processing complete!")
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            raise
            
    async def semantic_search(self, query: str, k: int = 3) -> List[Dict]:
        """Perform semantic search on the processed documents."""
        if not self.vector_store:
            raise Exception("No books processed yet. Please process a book first.")
            
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            seen_content = set()
            
            for doc, score in results:
                content = doc.page_content.strip()
                if content in seen_content:
                    continue
                    
                seen_content.add(content)
                formatted_results.append({
                    'content': content,
                    'metadata': doc.metadata,
                    'relevance_score': float(score)
                })
                
            return formatted_results
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            raise