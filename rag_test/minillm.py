from sentence_transformers import SentenceTransformer
import torch
from typing import List
import numpy as np
import time

class SemanticMatcher:
    def __init__(self):
        # Optimize for CPU usage
        torch.set_num_threads(4)  # Adjust based on your CPU cores
        torch.set_grad_enabled(False)
        
        # Load the model
        print("Loading model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("Model loaded successfully!")
        
    def calculate_similarity(self, source_sentence: str, comparison_sentences: List[str]) -> List[tuple]:
        """
        Calculate semantic similarity between a source sentence and multiple comparison sentences.
        
        Args:
            source_sentence (str): The main sentence to compare against
            comparison_sentences (List[str]): List of sentences to compare with the source
            
        Returns:
            List[tuple]: List of (sentence, similarity_score) tuples, sorted by similarity
        """
        # Start timing
        start_time = time.time()
        
        # Encode sentences
        print("\nEncoding sentences...")
        source_embedding = self.model.encode([source_sentence])[0]  # Get the first element since we only have one sentence
        comparison_embeddings = self.model.encode(comparison_sentences)
        
        # Calculate similarities using cosine similarity
        similarities = []
        for i, target_embedding in enumerate(comparison_embeddings):
            similarity = np.dot(source_embedding, target_embedding) / \
                        (np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding))
            similarities.append((comparison_sentences[i], float(similarity)))
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        
        return similarities

def main():
    # Initialize the matcher
    matcher = SemanticMatcher()
    
    # Example source sentence
    source = "Currently its very warm"
    
    # Example comparison sentences
    comparisons = [
        "My laptop isn't turning on, what should I do?",
        "What are the best computer repair services?",
        "How to troubleshoot a computer that won't boot up",
        "I love playing video games on my computer",
        "What's the weather like today?",
        "The best programming languages for beginners",
        "How to make a delicious chocolate cake",
        "Tips for computer maintenance and repair",
        "Where can I buy a new computer?",
        "Basic steps for computer troubleshooting"
    ]
    
    print(f"\nSource sentence: '{source}'")
    print("\nCalculating similarities...")
    
    # Calculate similarities
    results = matcher.calculate_similarity(source, comparisons)
    
    # Print results
    print("\nResults (sorted by similarity):")
    print("-" * 80)
    for sentence, score in results:
        print(f"Score: {score:.4f} | Sentence: {sentence}")

if __name__ == "__main__":
    main()