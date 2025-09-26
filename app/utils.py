"""
Utility functions for text cleaning, sentence splitting, and other common operations.
"""

import re
import string
from typing import List, Set
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (will only download if not already present)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Cache stopwords
ENGLISH_STOPWORDS: Set[str] = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove or replace special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\{\}\"\']+', ' ', text)
    
    # Fix multiple punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    # Remove extra spaces around punctuation
    text = re.sub(r'\s+([\.!\?,:;])', r'\1', text)
    text = re.sub(r'([\.!\?,:;])\s+', r'\1 ', text)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK sentence tokenizer.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    try:
        # Use NLTK sentence tokenizer
        sentences = sent_tokenize(text)
        
        # Clean each sentence
        cleaned_sentences = []
        for sentence in sentences:
            cleaned = clean_text(sentence)
            if cleaned and len(cleaned.split()) >= 3:  # Filter very short sentences
                cleaned_sentences.append(cleaned)
        
        return cleaned_sentences
        
    except Exception:
        # Fallback to simple splitting if NLTK fails
        sentences = re.split(r'[.!?]+', text)
        return [clean_text(s) for s in sentences if clean_text(s) and len(clean_text(s).split()) >= 3]

def tokenize_text(text: str, remove_stopwords: bool = True, min_length: int = 2) -> List[str]:
    """
    Tokenize text into words with optional stopword removal.
    
    Args:
        text: Input text to tokenize
        remove_stopwords: Whether to remove stopwords
        min_length: Minimum word length to keep
        
    Returns:
        List of tokens
    """
    if not text:
        return []
    
    try:
        # Use NLTK word tokenizer
        tokens = word_tokenize(text.lower())
    except Exception:
        # Fallback to simple splitting
        tokens = text.lower().split()
    
    # Filter tokens
    filtered_tokens = []
    for token in tokens:
        # Remove punctuation and check length
        token = token.strip(string.punctuation)
        if (len(token) >= min_length and 
            token.isalpha() and 
            (not remove_stopwords or token not in ENGLISH_STOPWORDS)):
            filtered_tokens.append(token)
    
    return filtered_tokens

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis.
    
    Args:
        text: Input text
        top_k: Number of top keywords to return
        
    Returns:
        List of keywords sorted by frequency
    """
    if not text:
        return []
    
    # Tokenize and count
    tokens = tokenize_text(text, remove_stopwords=True, min_length=3)
    
    if not tokens:
        return []
    
    # Count frequencies
    word_freq = {}
    for token in tokens:
        word_freq[token] = word_freq.get(token, 0) + 1
    
    # Sort by frequency and return top_k
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:top_k]]

def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to maximum length while preserving word boundaries.
    
    Args:
        text: Input text to truncate
        max_length: Maximum length of output
        suffix: Suffix to add if text is truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    # Find last space before max_length
    truncate_pos = text.rfind(' ', 0, max_length - len(suffix))
    
    if truncate_pos == -1:
        # No space found, hard truncate
        truncate_pos = max_length - len(suffix)
    
    return text[:truncate_pos] + suffix

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    if not text:
        return ""
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    return text.strip()

def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        text: Input text with potential HTML tags
        
    Returns:
        Text with HTML tags removed
    """
    if not text:
        return ""
    
    # Remove HTML tags
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    
    # Decode common HTML entities
    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&nbsp;': ' '
    }
    
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)
    
    return normalize_whitespace(text)

def extract_stock_symbols(text: str) -> List[str]:
    """
    Extract potential stock symbols from text.
    
    Args:
        text: Input text
        
    Returns:
        List of potential stock symbols
    """
    if not text:
        return []
    
    # Pattern for stock symbols (1-5 uppercase letters, possibly with dots)
    symbol_pattern = r'\b[A-Z]{1,5}(?:\.[A-Z]{1,2})?\b'
    
    # Find all matches
    matches = re.findall(symbol_pattern, text)
    
    # Filter out common words that might match the pattern
    common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'UP', 'DO', 'NO', 'IF', 'MY', 'SO', 'US', 'AM', 'AN', 'ME', 'WE', 'HE', 'BE', 'HAS', 'OR', 'AS', 'OF', 'AT', 'IN', 'ON', 'TO', 'IS', 'IT'}
    
    symbols = []
    for match in matches:
        if match not in common_words and len(match) <= 5:
            symbols.append(match)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_symbols = []
    for symbol in symbols:
        if symbol not in seen:
            seen.add(symbol)
            unique_symbols.append(symbol)
    
    return unique_symbols

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple Jaccard similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Tokenize both texts
    tokens1 = set(tokenize_text(text1, remove_stopwords=True))
    tokens2 = set(tokenize_text(text2, remove_stopwords=True))
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    return intersection / union if union > 0 else 0.0

def format_datetime(dt, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime object to string.
    
    Args:
        dt: Datetime object
        format_str: Format string
        
    Returns:
        Formatted datetime string
    """
    if dt is None:
        return ""
    
    try:
        return dt.strftime(format_str)
    except Exception:
        return str(dt)
