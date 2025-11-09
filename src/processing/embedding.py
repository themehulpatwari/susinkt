"""
Embedding system for YouTube Shorts decision-making.
Creates embeddings from video metadata and maintains Stay/Scroll collections
for similarity-based predictions.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os


class VideoEmbeddingStore:
    """
    Manages embeddings for YouTube shorts videos.
    Maintains separate collections for 'Stay' and 'Scroll' actions.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", storage_path: Optional[str] = None):
        """
        Initialize the embedding store.
        
        Args:
            model_name: SentenceTransformer model to use. 
                       'all-MiniLM-L6-v2' is fast and efficient (384 dims)
            storage_path: Optional path to save/load embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.storage_path = storage_path
        
        # Two separate embedding collections
        self.stay_embeddings: List[np.ndarray] = []
        self.scroll_embeddings: List[np.ndarray] = []
        
        # Optional: Keep metadata for reference
        self.stay_metadata: List[Dict] = []
        self.scroll_metadata: List[Dict] = []
        
        # Load existing embeddings if storage path exists
        if storage_path and os.path.exists(storage_path):
            self.load_embeddings()
    
    def metadata_to_text(self, metadata: Dict) -> str:
        """
        Convert metadata dictionary to a structured text for embedding.
        
        Args:
            metadata: Video metadata dictionary
            
        Returns:
            Formatted text combining all textual features
        """
        parts = []
        
        # Title (most important)
        if metadata.get('title'):
            parts.append(f"Title: {metadata['title']}")
        
        # Description
        if metadata.get('description'):
            parts.append(f"Description: {metadata['description']}")
        
        # Visual context from thumbnail
        if metadata.get('thumbnail_context'):
            parts.append(f"Visual: {metadata['thumbnail_context']}")
        
        # Tags
        if metadata.get('tags'):
            tags_str = ', '.join(metadata['tags'])
            parts.append(f"Tags: {tags_str}")
        
        # Category
        if metadata.get('categoryName'):
            parts.append(f"Category: {metadata['categoryName']}")
        
        # Channel
        if metadata.get('channelName'):
            parts.append(f"Channel: {metadata['channelName']}")
        
        return ' | '.join(parts)
    
    def create_embedding(self, metadata: Dict) -> np.ndarray:
        """
        Create an embedding vector from video metadata.
        
        Args:
            metadata: Video metadata dictionary
            
        Returns:
            Embedding vector as numpy array
        """
        text = self.metadata_to_text(metadata)
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def add_to_stay(self, metadata: Dict, store_metadata: bool = True):
        """
        Add a video to the 'Stay' collection (user stayed to watch).
        
        Args:
            metadata: Video metadata dictionary
            store_metadata: Whether to store the metadata for reference
        """
        embedding = self.create_embedding(metadata)
        self.stay_embeddings.append(embedding)
        
        if store_metadata:
            self.stay_metadata.append(metadata)
    
    def add_to_scroll(self, metadata: Dict, store_metadata: bool = True):
        """
        Add a video to the 'Scroll' collection (user scrolled past).
        
        Args:
            metadata: Video metadata dictionary
            store_metadata: Whether to store the metadata for reference
        """
        embedding = self.create_embedding(metadata)
        self.scroll_embeddings.append(embedding)
        
        if store_metadata:
            self.scroll_metadata.append(metadata)
    
    def calculate_similarity(self, embedding: np.ndarray, collection: List[np.ndarray]) -> float:
        """
        Calculate average cosine similarity between an embedding and a collection.
        
        Args:
            embedding: Single embedding vector
            collection: List of embedding vectors
            
        Returns:
            Average similarity score (0-1)
        """
        if not collection:
            return 0.0
        
        # Calculate cosine similarity with each embedding in collection
        similarities = []
        for stored_embedding in collection:
            # Cosine similarity: dot product of normalized vectors
            similarity = np.dot(embedding, stored_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
            )
            similarities.append(similarity)
        
        # Return average similarity
        return float(np.mean(similarities))
    
    def get_similarity_scores(self, metadata: Dict) -> Tuple[float, float]:
        """
        Get similarity scores for a video against both Stay and Scroll collections.
        
        Args:
            metadata: Video metadata dictionary
            
        Returns:
            Tuple of (stay_similarity, scroll_similarity)
        """
        embedding = self.create_embedding(metadata)
        
        stay_similarity = self.calculate_similarity(embedding, self.stay_embeddings)
        scroll_similarity = self.calculate_similarity(embedding, self.scroll_embeddings)
        
        return stay_similarity, scroll_similarity
    
    def get_collection_sizes(self) -> Tuple[int, int]:
        """
        Get the number of embeddings in each collection.
        
        Returns:
            Tuple of (stay_count, scroll_count)
        """
        return len(self.stay_embeddings), len(self.scroll_embeddings)
    
    def save_embeddings(self):
        """Save embeddings to disk."""
        if not self.storage_path:
            raise ValueError("No storage path specified")
        
        data = {
            'stay_embeddings': [emb.tolist() for emb in self.stay_embeddings],
            'scroll_embeddings': [emb.tolist() for emb in self.scroll_embeddings],
            'stay_metadata': self.stay_metadata,
            'scroll_metadata': self.scroll_metadata
        }
        
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f)
    
    def load_embeddings(self):
        """Load embeddings from disk."""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        
        self.stay_embeddings = [np.array(emb) for emb in data['stay_embeddings']]
        self.scroll_embeddings = [np.array(emb) for emb in data['scroll_embeddings']]
        self.stay_metadata = data.get('stay_metadata', [])
        self.scroll_metadata = data.get('scroll_metadata', [])


def main():
    """Example usage of the VideoEmbeddingStore"""
    
    # Example metadata from a YouTube short
    example_metadata = {
        'title': 'Be kind to our delivery drivers! ❤️ #goodnews #kindness #ringcam #goodman',
        'channelName': 'Cosondra',
        'description': 'Feel Good News!',
        'publishedAt': 'November 06, 2024',
        'tags': ['freegood', 'goodnews', 'bekind', 'delivery drivers'],
        'viewCount': '49089759',
        'likeCount': '1719398',
        'commentCount': '25314',
        'duration': 20,
        'thumbnail_context': 'a person is walking down a sidewalk with a camera',
        'categoryName': 'Music'
    }
    
    # Initialize store
    store = VideoEmbeddingStore()
    
    # Simulate user staying on this video
    print("Adding video to 'Stay' collection...")
    store.add_to_stay(example_metadata)
    
    # Check collection sizes
    stay_count, scroll_count = store.get_collection_sizes()
    print(f"\nCollection sizes - Stay: {stay_count}, Scroll: {scroll_count}")
    
    # Test similarity with the same video
    stay_sim, scroll_sim = store.get_similarity_scores(example_metadata)
    print(f"\nSimilarity scores for the same video:")
    print(f"  Stay similarity: {stay_sim:.4f}")
    print(f"  Scroll similarity: {scroll_sim:.4f}")
    
    # Example with a different video
    different_metadata = {
        'title': 'Amazing cooking recipe! #cooking #food',
        'channelName': 'ChefMaster',
        'description': 'Learn to cook amazing dishes',
        'tags': ['cooking', 'recipe', 'food'],
        'thumbnail_context': 'person cooking in a kitchen',
        'categoryName': 'Howto & Style'
    }
    
    print("\n\nTesting with a different video (cooking)...")
    stay_sim, scroll_sim = store.get_similarity_scores(different_metadata)
    print(f"Similarity scores:")
    print(f"  Stay similarity: {stay_sim:.4f}")
    print(f"  Scroll similarity: {scroll_sim:.4f}")


if __name__ == "__main__":
    main()
