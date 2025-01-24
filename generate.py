import torch
import faiss
import nltk
import os
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

class ConceptRetriever:
    def __init__(self, model_path: str = "final_model.pt", batch_size: int = 8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Load the model
        print(f"Loading model from {os.path.abspath(model_path)}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.model = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.eval()
        
        # Initialize FAISS index
        self.index = None
        self.sentences = []
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        
        # Download punkt if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def _get_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """Get embeddings for a batch of sentences."""
        try:
            # Process in batches
            all_embeddings = []
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i:i + self.batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Get embeddings from the first element of the tuple
                    embeddings = outputs[0]
                    # Use attention mask to get valid token embeddings
                    mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
                    masked_embeddings = embeddings * mask
                    # Pool to get sentence embeddings
                    summed = torch.sum(masked_embeddings, dim=1)
                    counts = torch.clamp(torch.sum(mask, dim=1), min=1e-9)
                    mean_pooled = summed / counts
                    all_embeddings.append(mean_pooled)
            
            return torch.cat(all_embeddings, dim=0)
        except Exception as e:
            print(f"Error getting embeddings: {str(e)}")
            return None

    def build_index(self, articles: List[str]):
        """Build FAISS index from articles."""
        try:
            print("Building index...")
            all_sentences = []
            for article in tqdm(articles):
                sentences = nltk.sent_tokenize(article)
                all_sentences.extend(sentences)
            
            if not all_sentences:
                print("Warning: No sentences found in articles")
                return
            
            self.sentences = all_sentences
            embeddings = self._get_embeddings(all_sentences)
            
            if embeddings is None or embeddings.size(0) == 0:
                print("Error: No valid embeddings generated")
                return
                
            # Convert to numpy and normalize
            embeddings_np = embeddings.cpu().numpy()
            faiss.normalize_L2(embeddings_np)
            
            # Build index
            self.index = faiss.IndexFlatIP(embeddings_np.shape[1])
            self.index.add(embeddings_np)
            print(f"Index built with {len(all_sentences)} sentences")
            
        except Exception as e:
            print(f"Error building index: {str(e)}")

    def generate_summary(self, article: str, num_sentences: int = 3) -> str:
        """Generate summary by retrieving similar sentences."""
        try:
            if self.index is None:
                raise ValueError("Index not built. Call build_index first.")
            
            # Get query embedding
            query_embedding = self._get_embeddings([article])
            if query_embedding is None:
                return ""
            
            # Convert to numpy and normalize
            query_np = query_embedding.cpu().numpy()
            faiss.normalize_L2(query_np)
            
            # Search index
            D, I = self.index.search(query_np, num_sentences)
            
            # Get top sentences
            summary_sentences = [self.sentences[i] for i in I[0]]
            return " ".join(summary_sentences)
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return ""

def main():
    # Example usage
    retriever = ConceptRetriever()
    
    # Example articles
    articles = [
        "The quick brown fox jumps over the lazy dog. This is a test sentence. Another test sentence here.",
        "Different article with some content. More sentences here. Final test sentence."
    ]
    
    # Build index
    retriever.build_index(articles)
    
    # Generate summary
    test_article = "Test article about a fox and a dog."
    summary = retriever.generate_summary(test_article)
    print(f"\nGenerated summary: {summary}")

if __name__ == "__main__":
    main() 