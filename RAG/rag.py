import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_gesture(self, frame) -> str:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]  # Get first hand
            # Draw landmarks on frame
            self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Get gesture type based on landmark positions
            return self._classify_gesture(landmarks)
        
        return "no_gesture"

    def _classify_gesture(self, landmarks) -> str:
        # Get relevant finger tip positions
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        
        # Basic gesture classification
        if thumb_tip.y < index_tip.y and middle_tip.y < index_tip.y:
            return "summarize"  # Hand pointing up
        elif thumb_tip.y > index_tip.y and middle_tip.y > index_tip.y:
            return "search"     # Hand pointing down
        elif abs(thumb_tip.x - index_tip.x) < 0.1:
            return "select"     # Pinch gesture
        
        return "unknown"

class RAGSystem:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.document_store = {}
        self.embeddings = {}
        
    def load_documents_from_directory(self, directory_path: str):
        """Load documents from a directory"""
        try:
            for file_path in os.listdir(directory_path):
                full_path = os.path.join(directory_path, file_path)
                if os.path.isfile(full_path):
                    # Get file extension
                    _, ext = os.path.splitext(file_path)
                    
                    # Process based on file type
                    if ext.lower() in ['.txt', '.md', '.py', '.java', '.cpp', '.js']:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Determine document type
                        doc_type = 'code' if ext.lower() in ['.py', '.java', '.cpp', '.js'] else 'text'
                        
                        # Add document
                        self.add_document(
                            doc_id=file_path,
                            content=content,
                            doc_type=doc_type
                        )
                        print(f"Loaded {file_path}")
                        
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            
    def generate_smart_queries(self, doc_id: str, num_queries: int = 3) -> List[str]:
        """Generate relevant search queries based on document content"""
        if doc_id not in self.document_store:
            return []
            
        content = self.document_store[doc_id]['content']
        
        # Split into sentences and remove empty ones
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        # For code documents
        if self.document_store[doc_id]['type'] == 'code':
            # Extract function names and important keywords
            keywords = []
            lines = content.split('\n')
            for line in lines:
                if 'def ' in line:
                    func_name = line.split('def ')[1].split('(')[0]
                    keywords.append(f"function {func_name}")
                if 'class ' in line:
                    class_name = line.split('class ')[1].split('(')[0]
                    keywords.append(f"class {class_name}")
            
            # Add some generic code queries
            keywords.extend([
                "code implementation",
                "algorithm example",
                "function usage"
            ])
            
            return keywords[:num_queries]
        
        # For text documents
        else:
            # Use first sentence and important keywords
            queries = []
            if sentences:
                queries.append(sentences[0])  # First sentence often contains main topic
                
            # Extract important phrases (simple approach)
            words = content.lower().split()
            common_phrases = [
                ' '.join(words[i:i+3]) 
                for i in range(len(words)-2) 
                if len(words[i]) > 3
            ]
            
            queries.extend(common_phrases[:num_queries-1])
            
            return queries[:num_queries]

    def add_document(self, doc_id: str, content: str, doc_type: str):
        """Add a document to the store with its embedding"""
        self.document_store[doc_id] = {
            'content': content,
            'type': doc_type
        }
        self.embeddings[doc_id] = self.model.encode(content)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        query_embedding = self.model.encode(query)
        
        similarities = {}
        for doc_id, doc_embedding in self.embeddings.items():
            similarity = cosine_similarity(
                [query_embedding], 
                [doc_embedding]
            )[0][0]
            similarities[doc_id] = similarity
        
        # Get top_k results
        top_results = sorted(
            similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return [
            {
                'doc_id': doc_id,
                'content': self.document_store[doc_id]['content'],
                'type': self.document_store[doc_id]['type'],
                'similarity': score
            }
            for doc_id, score in top_results
        ]

    def summarize(self, doc_id: str) -> str:
        """Simple extractive summarization"""
        if doc_id not in self.document_store:
            return "Document not found"
            
        content = self.document_store[doc_id]['content']
        sentences = content.split('.')
        
        # For now, return first and last sentences as summary
        if len(sentences) <= 2:
            return content
        return f"{sentences[0]}...{sentences[-1]}"

def main():
    import os
    
    # Initialize components
    gesture_recognizer = GestureRecognizer()
    rag_system = RAGSystem()
    
    # Get documents directory from user
    print("Enter the path to your documents directory (or press Enter for demo mode):")
    docs_dir = input().strip()
    
    if docs_dir and os.path.isdir(docs_dir):
        # Load documents from directory
        print(f"Loading documents from {docs_dir}...")
        rag_system.load_documents_from_directory(docs_dir)
    else:
        print("Using demo documents...")
        # Add demo documents
        rag_system.add_document(
            "article1",
            """Climate change is one of the biggest challenges facing our planet today. 
            Global temperatures have risen significantly over the past century. 
            This has led to melting ice caps, rising sea levels, and extreme weather events. 
            Scientists warn that immediate action is necessary to prevent catastrophic effects.""",
            "text"
        )
        
        rag_system.add_document(
            "code1",
            """def binary_search(arr, target):
                left, right = 0, len(arr) - 1
                while left <= right:
                    mid = (left + right) // 2
                    if arr[mid] == target:
                        return mid
                    elif arr[mid] < target:
                        left = mid + 1
                    else:
                        right = mid - 1
                return -1""",
            "code"
        )
    
    # Get available documents
    available_docs = list(rag_system.document_store.keys())
    if not available_docs:
        print("No documents loaded. Please add documents and try again.")
        return
        
    # Generate smart queries for each document
    search_queries = []
    for doc_id in available_docs:
        search_queries.extend(rag_system.generate_smart_queries(doc_id))
    
    # Remove duplicates and ensure we have at least some queries
    search_queries = list(set(search_queries))
    if not search_queries:
        search_queries = ["document content", "main topic", "key points"]
    
    current_query_index = 0
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # State variables
    current_doc_id = None
    last_gesture = "no_gesture"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect gesture
        gesture = gesture_recognizer.detect_gesture(frame)
        
        # Process gesture if it changed
        if gesture != last_gesture and gesture != "no_gesture":
            if gesture == "search":
                # Cycle through pre-defined queries
                query = search_queries[current_query_index]
                results = rag_system.search(query)
                print("\n=== SEARCH RESULTS ===")
                print(f"Query: {query}")
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"Document: {result['doc_id']}")
                    print(f"Type: {result['type']}")
                    print(f"Relevance: {result['similarity']:.2f}")
                    print("Preview:", result['content'][:100], "...")
                
                current_query_index = (current_query_index + 1) % len(search_queries)
                
            elif gesture == "summarize" and current_doc_id:
                summary = rag_system.summarize(current_doc_id)
                print("\n=== DOCUMENT SUMMARY ===")
                print(f"Document: {current_doc_id}")
                print(summary)
                
            elif gesture == "select":
                # Cycle through available documents
                available_docs = list(rag_system.document_store.keys())
                if current_doc_id is None:
                    current_doc_id = available_docs[0]
                else:
                    current_index = available_docs.index(current_doc_id)
                    current_doc_id = available_docs[(current_index + 1) % len(available_docs)]
                print("\n=== SELECTED DOCUMENT ===")
                print(f"Selected: {current_doc_id}")
                print("Content:", rag_system.document_store[current_doc_id]['content'][:100], "...")
        
        last_gesture = gesture
        
        # Display gesture on frame
        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Show frame
        cv2.imshow('Gesture RAG', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()