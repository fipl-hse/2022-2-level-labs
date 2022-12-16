"""
TextRank summarizer starter
"""
from pathlib import Path
from string import punctuation
import json
from lab_4_summarization_textrank.main import (SentencePreprocessor, SentenceEncoder,
SimilarityMatrix, TextRankSummarizer)

if __name__ == "__main__":
    # finding paths to the necessary utils
    PROJECT_ROOT = Path(__file__).parent
    ASSETS_PATH = PROJECT_ROOT / 'assets'
    TEXTS_PATH = ASSETS_PATH.joinpath('texts')

    # reading the text from which keywords are going to be extracted
    TARGET_TEXT_PATH = TEXTS_PATH / 'article_701.txt'
    with open(TARGET_TEXT_PATH, 'r', encoding='utf-8') as file:
        text = file.read()

    # reading list of stop words
    STOP_WORDS_PATH = ASSETS_PATH / 'stop_words.txt'
    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        stop_words = tuple(file.read().split('\n'))

    # reading IDF scores
    IDF_PATH = ASSETS_PATH / 'IDF.json'
    with open(IDF_PATH, 'r', encoding='utf-8') as file:
        idf = json.load(file)

    paths_to_texts = [str(path) for path in TEXTS_PATH.glob('*.txt')]
    preprocess = SentencePreprocessor(stop_words, tuple(sym for sym in punctuation))
    preprocessed_sent = preprocess.get_sentences(text)
    encode = SentenceEncoder()
    encode.encode_sentences(preprocessed_sent)
    text_graph = SimilarityMatrix()
    text_graph.fill_from_sentences(preprocessed_sent)
    summary = TextRankSummarizer(text_graph)
    summary.train()
    top_sentences = summary.get_top_sentences(10)
    TEXT_SUMMARY = summary.make_summary(7)

    RESULT = TEXT_SUMMARY
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Summaries are not extracted'
