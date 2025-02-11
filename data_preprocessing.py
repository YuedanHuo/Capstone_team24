import pandas as pd
import numpy as np
# check null and replace them with empty string
# and also strip the text
def clean_text(text):
    text = str(text).strip() if pd.notna(text) else ""  # Handle NaN & strip spaces
    return text


# spellchecker
from spellchecker import SpellChecker

# Load spell checkers for English and French
spell_en = SpellChecker(language="en")
spell_fr = SpellChecker(language="fr")

# function to correct spelling
def correct_tokens(tokens, spell):
    """
    Corrects spelling of word tokens while keeping punctuation untouched.
    """
    corrected_tokens = []
    for token in tokens:
        if token.isalpha():  # Only check spelling for words
            corrected_word = spell.correction(token)
            corrected_tokens.append(corrected_word if corrected_word else token)  # Keep original if no correction
        else:
            corrected_tokens.append(token)  # Leave punctuation untouched
    return corrected_tokens

# function to assign spellcheck according to country
def correct_spelling(text, country):
    spell = spell_fr if country == 'France' else spell_en  # Choose correct spell checker
    
    tokens = text.split()  # Tokenize text
    corrected_tokens = correct_tokens(tokens, spell)  # Apply spell checking
    return " ".join(corrected_tokens)  # Convert back to string


# replace emoji
import emoji

# Function to replace emoji with text based on country
def replace_emoji(text, country):
    lang = "fr" if country == "France" else "en"  # Choose French or English
    return emoji.demojize(text, language=lang).replace(":", "").replace("_", " ")  # Clean up output

# lemmanization
import spacy

# Load English & French models
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")

# Function to lemmatize text based on country
def lemmatize_text(text, country):
    nlp = nlp_fr if country == "France" else nlp_en  # Choose model
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])  # Get lemmatized words


# n-gram
from itertools import islice

# Function to generate n-grams
def generate_ngrams(text, n=2):
    words = text.split()
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    return ngrams

# Function to expand text with n-grams
def add_ngrams(text, max_n=3):
    words = text.split()
    all_ngrams = words[:]  # Keep original unigrams
    
    for n in range(2, max_n+1):  # Generate bigrams, trigrams
        all_ngrams.extend([" ".join(words[i:i+n]) for i in range(len(words)-n+1)])

    return " ".join(all_ngrams)  # Join all n-grams into a single string

# pipeline
def process_text_pipeline(text, country):
    text = clean_text(text)
    text = correct_spelling(text, country)
    text = replace_emoji(text, country)
    text = lemmatize_text(text, country)
    text = add_ngrams(text, 2) # create bigram for now
    return text

# pipeline without lemmanization and n-gram for bert and sbert
def process_text_noterming(text, country):
    text = clean_text(text)
    text = correct_spelling(text, country)
    text = replace_emoji(text, country)
    #text = lemmatize_text(text, country)
    #text = add_ngrams(text, 2) # create bigram for now
    return text


