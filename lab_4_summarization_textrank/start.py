"""
TextRank summarizer starter
"""
from pathlib import Path
from string import punctuation
import json
from lab_4_summarization_textrank.main import (SentencePreprocessor, SentenceEncoder, SimilarityMatrix,
                                               TextRankSummarizer)

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

    sentence_preprocessor = SentencePreprocessor(stop_words, tuple(punctuation))
    sentences = sentence_preprocessor.get_sentences(text)

    sentence_encoder = SentenceEncoder()
    sentence_encoder.encode_sentences(sentences)

    similarity_matrix = SimilarityMatrix()
    similarity_matrix.fill_from_sentences(sentences)
    rank_summarizer = TextRankSummarizer(similarity_matrix)
    rank_summarizer.train()

    BRIEF = rank_summarizer.make_summary(7)
    print(BRIEF)

    RESULT = BRIEF
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Summaries are not extracted'
