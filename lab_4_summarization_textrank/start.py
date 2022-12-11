"""
TextRank summarizer starter
"""
from lab_4_summarization_textrank.main import (Sentence,
                                               SentencePreprocessor,
                                               SentenceEncoder,
                                               SimilarityMatrix,
                                               TextRankSummarizer)


from pathlib import Path
import json

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
    PREPROCESSOR = SentencePreprocessor(stop_words, tuple('.,!?-:;()'))
    PREPROCESSED_SENTENCES = PREPROCESSOR.get_sentences(text)

    ENCODER = SentenceEncoder()
    ENCODER.encode_sentences(PREPROCESSED_SENTENCES)

    # step 9
    GRAPH = SimilarityMatrix()
    GRAPH.fill_from_sentences(PREPROCESSED_SENTENCES)

    SUMMARIZER = TextRankSummarizer(GRAPH)
    SUMMARIZER.train()

    text_summary = SUMMARIZER.make_summary(4)
    print('The text summary:', text_summary, sep='\n')





    RESULT = None
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    # assert RESULT, 'Summaries are not extracted'
