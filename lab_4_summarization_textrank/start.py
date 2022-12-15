"""
TextRank summarizer starter
"""
from pathlib import Path
import json
from string import punctuation
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

    paths_to_texts = [str(path) for path in TEXTS_PATH.glob('*.txt')]

    PUNCTUATION = tuple(punctuation)
    PREPROCESSING = SentencePreprocessor(stop_words, PUNCTUATION)
    SENT_LIST = PREPROCESSING.get_sentences(text)
    ENCODED_SENT = SentenceEncoder()
    ENCODED_SENT.encode_sentences(SENT_LIST)

    for one_sent in SENT_LIST:
        print(one_sent.get_encoded())

    MATRIX = SimilarityMatrix()
    MATRIX.fill_from_sentences(SENT_LIST)
    SUMMARIZER = TextRankSummarizer(MATRIX)
    SUMMARIZER.train()
    SUMMARY = SUMMARIZER.make_summary(10)

    print('На 8: краткое содержание текста \n', SUMMARY)

    BUDDY = Buddy(paths_to_texts, stop_words, PUNCTUATION, idf)
    QUESTION1 = 'Кто убил девочек в Рыбинске?'
    QUESTION2 = 'Педофил Тесак убийства'
    print(BUDDY.reply(QUESTION1, 3))
    print(BUDDY.reply(QUESTION2, 3))


    RESULT = SENT_LIST
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Summaries are not extracted'
