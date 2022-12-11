"""
TextRank summarizer starter
"""
from pathlib import Path
from string import punctuation
import json
from lab_4_summarization_textrank.main import SentencePreprocessor, SentenceEncoder, \
    SimilarityMatrix, TextRankSummarizer, Buddy, NoRelevantTextsError, IncorrectQueryError

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

    # step 5
    PREPROCESSOR = SentencePreprocessor(stop_words, tuple(punctuation))
    ENCODER = SentenceEncoder()
    SENTENCES = PREPROCESSOR.get_sentences(text)
    ENCODER.encode_sentences(SENTENCES)
    for sentence in SENTENCES:
        print(sentence.get_encoded())
    print()

    # step 9
    graph = SimilarityMatrix()
    graph.fill_from_sentences(SENTENCES)
    summarizer = TextRankSummarizer(graph)
    summarizer.train()
    SUMMARY = summarizer.make_summary(5)
    print(SUMMARY)
    print()

    # step 11
    BUDDY = Buddy(paths_to_texts, stop_words, tuple(punctuation), idf)
    QUERY_LIST = ['Юрий Гагарин в космосе']
    for query in QUERY_LIST:
        try:
            print(BUDDY.reply(query))
            print()
        except NoRelevantTextsError:
            print('Try again.')
        except IncorrectQueryError:
            print('Try again.')

    RESULT = SUMMARY
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Summaries are not extracted'
