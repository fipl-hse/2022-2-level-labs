"""
TextRank summarizer starter
"""
from pathlib import Path
import json
from lab_4_summarization_textrank.main import SentencePreprocessor, SentenceEncoder, \
    SimilarityMatrix, TextRankSummarizer, Buddy

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

    punctuation = tuple(mark for mark in ['.,!?-:;()&'])
    paths_to_texts = [str(path) for path in TEXTS_PATH.glob('*.txt')]

    SENTENCE_PREPROCESSOR = SentencePreprocessor(stop_words, punctuation)
    SENTENCES = SENTENCE_PREPROCESSOR.get_sentences(text)
    SENTENCE_ENCODER = SentenceEncoder()
    SENTENCE_ENCODER.encode_sentences(SENTENCES)

    MATRIX = SimilarityMatrix()
    MATRIX.fill_from_sentences(SENTENCES)

    TEXT_RANK = TextRankSummarizer(MATRIX)
    TEXT_RANK.train()
    SUMMARY = TEXT_RANK.make_summary(10)

    BUDDY = Buddy(paths_to_texts, stop_words, punctuation, idf)
    USER_INPUT = 'Юрий Алексеевич Гагарин это кто?'
    print(BUDDY.reply(USER_INPUT))

    RESULT = SUMMARY
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Summaries are not extracted'
