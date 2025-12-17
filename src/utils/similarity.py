"""Similarity filtering utilities."""
from typing import List, Tuple
from langchain_core.documents import Document


def filter_by_similarity_threshold(
    results_with_scores: List[Tuple[Document, float]],
    min_similarity: float,
    k: int
) -> List[Tuple[Document, float]]:
    """
    Filter search results by similarity threshold.
    
    Converts ChromaDB distance scores to similarity scores and filters
    documents that meet the minimum similarity threshold.
    
    Args:
        results_with_scores: List of (Document, distance_score) tuples from vector store
        min_similarity: Minimum similarity score threshold (0.0-1.0)
        k: Maximum number of results to return
    
    Returns:
        List of (Document, similarity_score) tuples that meet the similarity threshold
    """
    filtered_docs = []
    for doc, distance in results_with_scores:
        similarity = 1 - distance  # Convert distance to similarity score
        if similarity >= min_similarity:
            filtered_docs.append((doc, similarity))
        if len(filtered_docs) >= k:  # Stop once we have k results
            break
    
    return filtered_docs

