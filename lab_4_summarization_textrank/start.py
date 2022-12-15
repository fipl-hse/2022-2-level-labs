"""
TextRank summarizer starter
"""
from pathlib import Path
import json
from string import punctuation
from lab_4_summarization_textrank.main import SentencePreprocessor, SentenceEncoder, SimilarityMatrix,\
    TextRankSummarizer

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

    # mark 6
    preprocessor = SentencePreprocessor(stop_words, tuple(punctuation))
    encoder = SentenceEncoder()

    sentences = preprocessor.get_sentences(text)
    encoder.encode_sentences(sentences)
    for sentence in sentences:
        print(sentence.get_encoded())

    # mark 8
    matrix = SimilarityMatrix()
    matrix.fill_from_sentences(sentences)
    text_rank_summarizer = TextRankSummarizer(matrix)
    text_rank_summarizer.train()
    SUMMARY = text_rank_summarizer.make_summary(10)
    print(SUMMARY)

    RESULT = SUMMARY
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Summaries are not extracted'
