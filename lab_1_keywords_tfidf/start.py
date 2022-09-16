"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path

from main import (clean_and_tokenize, remove_stop_words, calculate_frequencies, get_top_n, calculate_tf,
                  calculate_tfidf, calculate_expected_frequency, calculate_chi_values, extract_significant_words)

if __name__ == "__main__":

    # finding paths to the necessary utils
    PROJECT_ROOT = Path(__file__).parent
    ASSETS_PATH = PROJECT_ROOT / "assets"

    # reading the text from which keywords are going to be extracted
    TARGET_TEXT_PATH = ASSETS_PATH / "Дюймовочка.txt"
    with open(TARGET_TEXT_PATH, "r", encoding="utf-8") as file:
        target_text = file.read()

    # reading list of stop words
    STOP_WORDS_PATH = ASSETS_PATH / "stop_words.txt"
    with open(STOP_WORDS_PATH, "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")

    # reading IDF scores for all tokens in the corpus of H.C. Andersen tales
    IDF_PATH = ASSETS_PATH / "IDF.json"
    with open(IDF_PATH, "r", encoding="utf-8") as file:
        idf = json.load(file)

    # reading frequencies for all tokens in the corpus of H.C. Andersen tales
    CORPUS_FREQ_PATH = ASSETS_PATH / "corpus_frequencies.json"
    with open(CORPUS_FREQ_PATH, "r", encoding="utf-8") as file:
        corpus_freqs = json.load(file)

split_words = clean_and_tokenize(target_text)
not_stop_words = remove_stop_words(split_words, stop_words)
freq_dict = calculate_frequencies(not_stop_words)
top_words = get_top_n(freq_dict, 10)
tf_dict = calculate_tf(freq_dict)
tfidf = calculate_tfidf(tf_dict, idf)
print(f"The top of words by tfidf: {get_top_n(tfidf, 10)}")
expected_dict = calculate_expected_frequency(freq_dict, corpus_freqs)
x2_dict = calculate_chi_values(expected_dict, freq_dict)
top_x2 = get_top_n(x2_dict, 10)
print(f"The top of words by chi: {top_x2}")
significant_words_dict = extract_significant_words(x2_dict, 0.05)

RESULT = top_x2
# DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
assert RESULT, 'Keywords are not extracted'
