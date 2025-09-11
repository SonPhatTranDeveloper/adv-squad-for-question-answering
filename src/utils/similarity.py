import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Create a global instance of the sentence transformer
sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")


def get_similarity(sentence1: str, sentence2: str) -> float:
    """
    Get the cosine similarity between two sentences using the sentence transformer.

    Args:
        - sentence1: The first sentence.
        - sentence2: The second sentence.

    Returns:
        - The similarity between the two sentences.
    """
    # Encode the sentences in a list
    sentences = [sentence1, sentence2]
    embeddings = sentence_transformer.encode(sentences, show_progress_bar=False)
    embedding1 = embeddings[0].reshape(1, -1)
    embedding2 = embeddings[1].reshape(1, -1)

    # Calculate the similarity using sklearn cosine similarity
    return cosine_similarity(embedding1, embedding2)[0][0]


def get_similarity_sentence_average(paragraph1: str, paragraph2: str) -> float:
    """
    Get the average similarity between all sentences in two paragraphs.

    This is an advanced optimized version that uses better sentence splitting
    and fully vectorized operations for maximum performance.

    Args:
        paragraph1: The first paragraph.
        paragraph2: The second paragraph.

    Returns:
        The average cosine similarity between corresponding sentences.
    """
    # Split paragraphs into sentences using regex for better accuracy
    # This handles abbreviations, decimals, and other edge cases better
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"

    sentences1 = [
        s.strip() for s in re.split(sentence_pattern, paragraph1) if s.strip()
    ]
    sentences2 = [
        s.strip() for s in re.split(sentence_pattern, paragraph2) if s.strip()
    ]

    # Handle edge cases
    if not sentences1 or not sentences2:
        return 0.0

    # Handle unequal lengths by taking the minimum length
    min_length = min(len(sentences1), len(sentences2))
    sentences1 = sentences1[:min_length]
    sentences2 = sentences2[:min_length]

    # Batch encode all sentences at once for better performance
    all_sentences = sentences1 + sentences2
    embeddings = sentence_transformer.encode(all_sentences, show_progress_bar=False)

    # Split embeddings back into two groups
    embeddings1 = embeddings[: len(sentences1)]
    embeddings2 = embeddings[len(sentences1) :]

    # Calculate cosine similarities using fully vectorized operations
    # This is the fastest approach using matrix operations
    embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # Element-wise dot product for each sentence pair
    similarities = np.sum(embeddings1_norm * embeddings2_norm, axis=1)

    return float(np.mean(similarities))
