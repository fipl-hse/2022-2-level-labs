"""
TextRank summarizer starter
"""
import string
from pathlib import Path
import json

from lab_4_summarization_textrank.main import SentenceEncoder, \
    SentencePreprocessor, SimilarityMatrix, TextRankSummarizer

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
    punctuation = tuple(string.punctuation)
    preprocess_sentences = SentencePreprocessor(stop_words, punctuation)
    completed_sentences = preprocess_sentences.get_sentences(text)
    encoder = SentenceEncoder()
    encoder.encode_sentences(completed_sentences)
    for one_sentence in completed_sentences:
        print(*(one_sentence.get_encoded()))
    matrix_graph = SimilarityMatrix()
    matrix_graph.fill_from_sentences(completed_sentences)
    text_rank = TextRankSummarizer(matrix_graph)
    text_rank.train()
    COMPLETED_SUMMARY = text_rank.make_summary(5)
    print('Summary:\n', COMPLETED_SUMMARY)

    RESULT = COMPLETED_SUMMARY
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Summaries are not extracted'
