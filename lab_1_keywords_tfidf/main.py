"""
Lab 1
Extract keywords based on frequency related metrics
"""
from typing import Optional, Union
import math

def clean_and_tokenize(text):
    if not isinstance(text, str):
        return None
    for i in text:
        if not i.isalnum() and i != ' ':
            text = text.replace(i, '')
    text = text.lower()
    text = text.split()
    return text

def remove_stop_words(tokens, stop_words):
    if not isinstance(tokens, list) or not isinstance(stop_words, list):
        return None
    for i in stop_words:
        if not isinstance(i, str):
            return None
    for i in tokens:
        if not isinstance(i, str):
            return None
    tokens_new = []
    for i in tokens:
        if i not in stop_words:
            tokens_new.append(i)
    return tokens_new


def calculate_frequencies(tokens):
    if not isinstance(tokens, list):
        return None
    for i in tokens:
        if not isinstance(i, str):
            return None
    d = {}
    for i in tokens:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    return d


def get_top_n(frequencies, top):
    if not isinstance(top, int) or not isinstance(frequencies, dict):
        return None
    for i in frequencies.keys():
        if not isinstance(i, str):
            return None
    for i in frequencies.values():
        if not isinstance(i, int) and not isinstance(i, float):
            return None
    sort = sorted(frequencies.items(), reverse = True, key = lambda x: x[1])
    frequencies = dict(sort)
    a = []
    for k in frequencies.keys():
        a += k
    if top > len(a):
        return a
    else:
        return a[:top]


def calculate_tf(frequencies):
    for i in frequencies.keys():
        if not isinstance(i, str):
            return None
    for i in frequencies.values():
        if not isinstance(i, int):
            return None
    summa = 0 #количество слов
    freq_new = {}
    for v in frequencies.values():
        summa += v
    for k in frequencies.keys():
        freq_new[k] = frequencies[k]/summa
    return freq_new


def calculate_tfidf(term_freq, idf):
    if not isinstance(term_freq, dict) or not isinstance(idf, dict):
        return None
    for i in term_freq.keys():
        for j in term_freq.values():
            if not isinstance(i, str) or not isinstance(j, float):
                return None
    for i in idf.keys():
        for j in idf.values():
            if not isinstance(i, str) or not isinstance(j, float):
                return None
    new = {}
    for i in term_freq.keys():
        if i not in idf:
            new [i] = term_freq[i] / math.log(47/1)
        else:
            new [i] = term_freq[i] / idf[i]
    return new


def calculate_expected_frequency(doc_freqs, corpus_freqs):
    if not isinstance(doc_freqs, dict) or not isinstance(corpus_freqs, dict):
        return None
    for i in doc_freqs.keys():
        for j in doc_freqs.values():
            if not isinstance(i, str) or not isinstance(j, int):
                return None
    for i in corpus_freqs.keys():
        for j in corpus_freqs.values():
            if not isinstance(i, str) or not isinstance(j, float):
                return None
    expected_freq = {}
    for e in doc_freqs.keys():
        j = doc_freqs[e]
        k = corpus_freqs[e]
        l = 1 - doc_freqs[e]
        m = 1 - corpus_freqs[e]
        expected_freq[e] = ((j+k)*(j+l))/ (j+k+l+m)
    return expected_freq


def calculate_chi_values(expected, observed):
    if not isinstance(expected, dict) or not isinstance(observed, dict):
        return None
    for i in expected.keys():
        for j in expected.values():
            if not isinstance(i, str) or not isinstance(j, float):
                return None
    for i in observed.keys():
        for j in observed.values():
            if not isinstance(i, str) or not isinstance(j, int):
                return None
    xi = {}
    for a in expected.keys():
        b = observed[a]
        c = expected[a]
        xi[a] = ((b - c)**2)/c
    return xi

def extract_significant_words(chi_values, alpha):
    if not isinstance(chi_values, dict) or not isinstance(alpha, float):
        return None
    for i in chi_values.keys():
        for j in chi_values.values():
            if not isinstance(i, str) or not isinstance(j, float):
                return None
    signific = {}
    for a, v in chi_values.items(): #а - ключи, v - значения
        if v > alpha:
            signific[a] = v
    return signific
