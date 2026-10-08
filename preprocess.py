"""
Text Preprocessing Module for Sentiment Analysis Tool.

Author: Surag M S
Project: Syntecxhub AI Internship - Project 2
"""

import re
import string


def preprocess_text(text: str) -> str:
    """Preprocess and clean raw input text for NLP feature extraction.

    Steps:
    1. Convert text to lowercase.
    2. Remove URLs (http, https, www).
    3. Remove email addresses.
    4. Remove @mentions.
    5. Remove hashtag symbols '#' while retaining the word.
    6. Remove numbers.
    7. Remove punctuation and special characters.
    8. Remove excessive whitespace.

    Args:
        text (str): Raw input text string.

    Returns:
        str: Cleaned and normalized text string.
    """
    if text is None or not isinstance(text, str):
        return ""

    # 1. Convert to lowercase
    cleaned = text.lower()

    # 2. Remove URLs (http://, https://, www.domain.com)
    cleaned = re.sub(r"https?://\S+|www\.\S+", "", cleaned)

    # 3. Remove email addresses
    cleaned = re.sub(r"\S+@\S+\.\S+", "", cleaned)

    # 4. Remove @mentions (e.g., @user)
    cleaned = re.sub(r"@\w+", "", cleaned)

    # 5. Remove hashtag symbols while keeping the tag text (e.g., #happy -> happy)
    cleaned = re.sub(r"#(\w+)", r"\1", cleaned)

    # 6. Remove numbers
    cleaned = re.sub(r"\d+", "", cleaned)

    # 7. Remove punctuation and special characters
    # Keep letters and whitespace
    cleaned = re.sub(r"[^\w\s]", "", cleaned)

    # 8. Remove excessive whitespace and normalize
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


if __name__ == "__main__":
    # Test example
    sample_text = (
        "I LOVE this product!!! Visit https://example.com or contact info@test.com #Awesome @user123!"
    )
    print("Original :", sample_text)
    print("Cleaned  :", preprocess_text(sample_text))
