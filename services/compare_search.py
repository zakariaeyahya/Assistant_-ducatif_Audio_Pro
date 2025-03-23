import sys
import os

# Ajoutez le chemin du répertoire parent au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
from typing import List
from services.HybridSearch import HybridSearch, SearchResult

async def compare_search_methods(
    hybrid_search: HybridSearch, 
    query: str, 
    top_k: int = 5
):
    """
    Compare vector, BM25, and hybrid search results for the same query
    
    Args:
        hybrid_search: Initialized HybridSearch instance
        query: The search query
        top_k: Number of top results to return
    """
    print(f"\n{'='*80}")
    print(f"COMPARING SEARCH METHODS FOR QUERY: '{query}'")
    print(f"{'='*80}")

    # Get results from all three methods
    vector_results = await hybrid_search.search_vector(query=query, top_k=top_k)
    bm25_results = await hybrid_search.search_bm25(query=query, top_k=top_k)
    hybrid_results = await hybrid_search.search_hybrid(
        query=query,
        top_k=top_k,
        vector_weight=0.7,
        bm25_weight=0.3
    )
    
    # Print vector search results
    print(f"\n{'-'*40}")
    print(f"VECTOR SEARCH RESULTS (Semantic similarity)")
    print(f"{'-'*40}")
    for i, result in enumerate(vector_results):
        print(f"\nResult {i+1} - Score: {result.score:.4f}")
        print(f"Source: {result.source}")
        print(f"Text: {result.text[:150]}...")  # Print first 150 characters
    
    # Print BM25 search results
    print(f"\n{'-'*40}")
    print(f"BM25 SEARCH RESULTS (Keyword matching)")
    print(f"{'-'*40}")
    for i, result in enumerate(bm25_results):
        print(f"\nResult {i+1} - Score: {result.score:.4f}")
        print(f"Source: {result.source}")
        print(f"Text: {result.text[:150]}...")  # Print first 150 characters
    
    # Print hybrid search results
    print(f"\n{'-'*40}")
    print(f"HYBRID SEARCH RESULTS (Vector 70% + BM25 30%)")
    print(f"{'-'*40}")
    for i, result in enumerate(hybrid_results):
        print(f"\nResult {i+1} - Score: {result.score:.4f}")
        print(f"Source: {result.source}")
        print(f"Original Vector Score: {result.original_scores.get('vector', 'N/A')}")
        print(f"Original BM25 Score: {result.original_scores.get('bm25', 'N/A')}")
        print(f"Text: {result.text[:150]}...")  # Print first 150 characters
    
    # Print results summary
    print(f"\n{'-'*40}")
    print(f"RESULTS COMPARISON")
    print(f"{'-'*40}")
    
    # Analyze potential overlap
    vector_texts = set([r.text[:100] for r in vector_results])
    bm25_texts = set([r.text[:100] for r in bm25_results])
    hybrid_texts = set([r.text[:100] for r in hybrid_results])
    
    vector_bm25_overlap = vector_texts.intersection(bm25_texts)
    vector_hybrid_overlap = vector_texts.intersection(hybrid_texts)
    bm25_hybrid_overlap = bm25_texts.intersection(hybrid_texts)
    
    print(f"Common results between Vector and BM25: {len(vector_bm25_overlap)}")
    print(f"Common results between Vector and Hybrid: {len(vector_hybrid_overlap)}")
    print(f"Common results between BM25 and Hybrid: {len(bm25_hybrid_overlap)}")
    print(f"Unique results in Hybrid search: {len(hybrid_texts - vector_texts - bm25_texts)}")

async def main():
    # Example configuration
    QDRANT_PATH = "D:/bureau/BD&AI 1/ci2/S2/tec_veille/mini_projet/local_qdrant_storage"
    COLLECTION_NAME = "asr_docs"
    
    # Initialize the hybrid search
    hybrid_search = HybridSearch(
        qdrant_path=QDRANT_PATH,
        collection_name=COLLECTION_NAME
    )
    
    # List of queries to test
    queries = [
        "What is information theory?",
        "speech recognition techniques",
        "machine learning for audio processing"
    ]
    
    # Compare results for each query
    for query in queries:
        await compare_search_methods(hybrid_search, query)

if __name__ == "__main__":
    asyncio.run(main())