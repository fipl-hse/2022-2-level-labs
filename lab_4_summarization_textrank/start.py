"""
TextRank summarizer starter
"""
from pathlib import Path
import json
import string
from lab_4_summarization_textrank.main import (
    SentencePreprocessor,
    SentenceEncoder,
    SimilarityMatrix,
    TextRankSummarizer,
    Buddy,
    NoRelevantTextsError
)

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

    punctuation = tuple(i for i in string.punctuation)
    preprocessor = SentencePreprocessor(stop_words, punctuation)
    encoder = SentenceEncoder()

    # step 5
    sentences = preprocessor.get_sentences(text)
    encoder.encode_sentences(sentences)
    for i in sentences:
        print(i.get_encoded())

    # step 9
    matrix = SimilarityMatrix()
    matrix.fill_from_sentences(sentences)
    text_rank_summarizer = TextRankSummarizer(matrix)
    text_rank_summarizer.train()
    summaries = text_rank_summarizer.make_summary(10)
    print(summaries)

    # step 11
    buddy = Buddy(paths_to_texts, stop_words, punctuation, idf)
    print('\nПомощник готов к работе!\n')
    while True:
        query = input('Input your query or q to quit: ')
        if query == 'q':
            break
        try:
            print(buddy.reply(query), '\n')
        except NoRelevantTextsError:
            print('Nothing found, try once more.')

    RESULT = summaries
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Summaries are not extracted'
