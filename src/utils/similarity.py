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
