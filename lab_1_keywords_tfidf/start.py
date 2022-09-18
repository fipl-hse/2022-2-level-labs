"""
Frequency-driven keyword extraction starter
"""
import json
from pathlib import Path

from typing import Optional, Union

import math
import string


if __name__ == "__main__":

    # finding paths to the necessary utils
    PROJECT_ROOT = Path(__file__).parent
    ASSETS_PATH = PROJECT_ROOT / 'assets'

    # reading the text from which keywords are going to be extracted
    TARGET_TEXT_PATH = ASSETS_PATH / 'Дюймовочка.txt'
    with open(TARGET_TEXT_PATH, 'r', encoding='utf-8') as file:
        target_text = file.read()

    # reading list of stop words
    STOP_WORDS_PATH = ASSETS_PATH / 'stop_words.txt'
    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        stop_words = file.read().split('\n')

    # reading IDF scores for all tokens in the corpus of H.C. Andersen tales
    IDF_PATH = ASSETS_PATH / 'IDF.json'
    with open(IDF_PATH, 'r', encoding='utf-8') as file:
        idf = json.load(file)

    # reading frequencies for all tokens in the corpus of H.C. Andersen tales
    CORPUS_FREQ_PATH = ASSETS_PATH / 'corpus_frequencies.json'
    with open(CORPUS_FREQ_PATH, 'r', encoding='utf-8') as file:
        corpus_freqs = json.load(file)

    RESULT = None


    def clean_and_tokenize(text: str) -> Optional[list[str]]:
        if isinstance(text, str):
            for p in string.punctuation:
                text = text.replace(p, '')
            clean_words = text.lower().strip().split()
            return clean_words
        else:
            return None


    a = clean_and_tokenize(target_text)
    print(a)


    def remove_stop_words(tokens: list[str], stop_words: list[str]) -> Optional[list[str]]:
        if (isinstance(tokens, list) and tokens != [] and all(isinstance(t, str) for t in tokens)
                and isinstance(stop_words, list)):
            no_stop_words = [t for t in tokens if t not in stop_words]
            return no_stop_words
        else:
            return None


    b = remove_stop_words(a, stop_words)
    print(b)

    def calculate_frequencies(tokens: list[str]) -> Optional[dict[str, int]]:
        if (isinstance(tokens, list) and tokens != []
                and all(isinstance(t, str) for t in tokens)):
            freq_dict = {}
            for t in tokens:
                if t not in freq_dict:
                    freq_dict[t] = 1
                else:
                    freq_dict[t] += 1
            return freq_dict
        else:
            return None


    c = calculate_frequencies(b)
    print(c)


    def get_top_n(frequencies: dict[str, Union[int, float]], top: int) -> Optional[list[str]]:
        if (isinstance(frequencies, dict) and frequencies is not None
                and all(isinstance(k, str) for k in frequencies.keys())
                and all(isinstance(v, int or float) for v in frequencies.values())
                and isinstance(top, int) and top is not (True or False) and top > 0):

            sorted_freq_dict = {k: v for k, v in sorted(frequencies.items(), key=lambda k: k[1], reverse=True)}
            sorted_words = list(sorted_freq_dict.keys())
            top_words = sorted_words if top > len(frequencies) else sorted_words[:top]
            return top_words

        else:
            return None


    d = get_top_n(c, 10)
    print(d)


    def calculate_tf(frequencies: dict[str, int]) -> Optional[dict[str, float]]:
        if (isinstance(frequencies, dict) and frequencies != {}
                and all(isinstance(k, str) for k in frequencies.keys())
                and all(isinstance(v, int or float) for v in frequencies.values())):
            words_num = sum(frequencies.values())
            tf_dict = {w: (f / words_num) for w, f in frequencies.items()}
            return tf_dict
        else:
            None


    e = calculate_tf(c)
    print(e)


    def calculate_tfidf(term_freq: dict[str, float], idf: dict[str, float]) -> Optional[dict[str, float]]:
        if (isinstance(term_freq, dict) and term_freq != {} and all(isinstance(w, str) for w in term_freq.keys())
                and all(isinstance(f, float) for f in term_freq.values())
                and isinstance(idf, dict) and all(isinstance(w, str) for w in idf.keys())
                and all(isinstance(f, float) for f in idf.values())):

            tfidf_dict = {}
            for w in term_freq:
                if w not in idf.keys():
                    idf[w] = math.log(47 / 1)
                tfidf_dict[w] = term_freq[w] * idf[w]
            return tfidf_dict
        else:
            return None


    f = calculate_tfidf(e, idf)
    print(f)

    def calculate_expected_frequency(
            doc_freqs: dict[str, int], corpus_freqs: dict[str, int]
    ) -> Optional[dict[str, float]]:
        """
        Calculates expected frequency for each of the tokens based on its
        Term Frequency score for both target document and general corpus

        Parameters:
        doc_freqs (Dict): A dictionary with tokens and its corresponding number of occurrences in document
        corpus_freqs (Dict): A dictionary with tokens and its corresponding number of occurrences in corpus

        Returns:
        Dict: A dictionary with tokens and its corresponding expected frequency

        In case of corrupt input arguments, None is returned
        """
        if (isinstance(doc_freqs, dict) and doc_freqs != {} and all(isinstance(k, str) for k in doc_freqs.keys())
            and all(isinstance(v, int) for v in doc_freqs.values())
            and isinstance(corpus_freqs, dict) and all(isinstance(k, str) for k in corpus_freqs.keys())
            and all(isinstance(v, int) for v in corpus_freqs.values())):

            expected_dict = {}
            doc_words_sum = sum(doc_freqs.values())
            collection_words_sum = sum(corpus_freqs.values())
            for w, f in doc_freqs.items():
                expected = (((f + corpus_freqs[w]) * (f + doc_words_sum - f)) /
                            (f + corpus_freqs[w] + doc_words_sum - f + collection_words_sum - corpus_freqs[w]))
                expected_dict[w] = expected
            return expected_dict
        else:
            return


    g = calculate_expected_frequency(c, corpus_freqs)
    print(g)


    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    #assert RESULT, 'Keywords are not extracted'