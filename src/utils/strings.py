import random

import gensim.downloader as api

# Load pre-trained GloVe model (this will download if not cached)
glove_model = api.load("glove-wiki-gigaword-50")


def randomly_inject_space(word: str) -> str:
    """
    Randomly inject a space into a word.

    Skips the first and last character of the word.

    Args:
        - word: The word to inject a space into.

    Returns:
        - The word with a space injected into a random position.
    """
    if len(word) <= 2:
        return word

    # Choose a random position between first and last character (exclusive)
    position = random.randint(1, len(word) - 1)
    return word[:position] + " " + word[position:]


def randomly_delete_character(word: str) -> str:
    """
    Randomly delete a character from a word.

    Skips the first and last character of the word.

    Args:
        - word: The word to delete a character from.

    Returns:
        - The word with a character deleted from a random position.
    """
    if len(word) <= 2:
        return word

    # Choose a random position between first and last character (exclusive)
    position = random.randint(1, len(word) - 2)
    return word[:position] + word[position + 1 :]


def randomly_swap_characters(word: str) -> str:
    """
    Randomly swap two characters in a word.

    Skips the first and last character of the word.

    Args:
        - word: The word to swap two characters in.

    Returns:
        - The word with two characters swapped.
    """
    if len(word) <= 3:
        return word

    # Choose two different random positions between first and last character (exclusive)
    available_positions = list(range(1, len(word) - 1))
    pos1, pos2 = random.sample(available_positions, 2)

    # Convert to list for easier swapping
    word_list = list(word)
    word_list[pos1], word_list[pos2] = word_list[pos2], word_list[pos1]

    return "".join(word_list)


def randomly_substitute_character(word: str) -> str:
    """
    Randomly substitute a character in a word based on a homoglyph map.

    "-": "˗",
    "9": "৭",
    "8": "Ȣ",
    "7": "𝟕",
    "6": "б",
    "5": "Ƽ",
    "4": "Ꮞ",
    "3": "Ʒ",
    "2": "ᒿ",
    "1": "l",
    "0": "O",
    "'": "`",
    "a": "ɑ",
    "b": "Ь",
    "c": "ϲ",
    "d": "ԁ",
    "e": "е",
    "f": "𝚏",
    "g": "ɡ",
    "h": "հ",
    "i": "і",
    "j": "ϳ",
    "k": "𝒌",
    "l": "ⅼ",
    "m": "ｍ",
    "n": "ո",
    "o": "о",
    "p": "р",
    "q": "ԛ",
    "r": "ⲅ",
    "s": "ѕ",
    "t": "𝚝",
    "u": "ս",
    "v": "ѵ",
    "w": "ԝ",
    "x": "×",
    "y": "у",
    "z": "ᴢ",

    Skips the first and last character of the word.

    Args:
        - word: The word to substitute a character in.

    Returns:
        - The word with a character substituted.
    """
    homoglyph_map = {
        "-": "˗",
        "9": "৭",
        "8": "Ȣ",
        "7": "𝟕",
        "6": "б",
        "5": "Ƽ",
        "4": "Ꮞ",
        "3": "Ʒ",
        "2": "ᒿ",
        "1": "l",
        "0": "O",
        "'": "`",
        "a": "ɑ",
        "b": "Ь",
        "c": "ϲ",
        "d": "ԁ",
        "e": "е",
        "f": "𝚏",
        "g": "ɡ",
        "h": "հ",
        "i": "і",
        "j": "ϳ",
        "k": "𝒌",
        "l": "ⅼ",
        "m": "ｍ",
        "n": "ո",
        "o": "о",
        "p": "р",
        "q": "ԛ",
        "r": "ⲅ",
        "s": "ѕ",
        "t": "𝚝",
        "u": "ս",
        "v": "ѵ",
        "w": "ԝ",
        "x": "×",
        "y": "у",
        "z": "ᴢ",
    }

    if len(word) <= 2:
        return word

    # Find characters that can be substituted (excluding first and last)
    substitutable_positions = []
    for i in range(1, len(word) - 1):
        if word[i].lower() in homoglyph_map:
            substitutable_positions.append(i)

    if not substitutable_positions:
        return word

    # Choose a random position to substitute
    position = random.choice(substitutable_positions)
    char_to_replace = word[position].lower()
    replacement_char = homoglyph_map[char_to_replace]

    # Preserve case if original was uppercase
    if word[position].isupper():
        replacement_char = replacement_char.upper()

    return word[:position] + replacement_char + word[position + 1 :]


def randomly_substitute_word(word: str, topk: int = 5) -> str:
    """
    Randomly substitute a word with its topk nearest neighbors in a word vector space.
    Word should be alphabetic.

    Using genism GloVe model.

    Args:
        - word: The word to substitute.
        - topk: The number of nearest neighbors to consider.

    Returns:
        - The word substituted.
    """
    # Check if word is alphabetic
    if not word.isalpha():
        return word

    # Check if word exists in vocabulary
    if word.lower() not in glove_model:
        return word

    # Get most similar words
    similar_words = glove_model.most_similar(word.lower(), topn=topk)

    # Extract just the words (not the similarity scores)
    candidate_words = [similar_word for similar_word, _ in similar_words]

    # Randomly select one of the candidates
    replacement = random.choice(candidate_words)

    # Preserve original case
    if word.isupper():
        return replacement.upper()
    elif word.istitle():
        return replacement.capitalize()
    else:
        return replacement
