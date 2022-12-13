"""
TextRank summarizer starter
"""
from pathlib import Path
import json
from string import punctuation
from lab_4_summarization_textrank.main import SentencePreprocessor, SentenceEncoder, SimilarityMatrix, \
    TextRankSummarizer, Buddy, Sentence

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
    sentence_preprocessor = SentencePreprocessor(stop_words, tuple(punctuation))
    sentences = sentence_preprocessor.get_sentences(text)
    sentence_encoder = SentenceEncoder()
    sentence_encoder.encode_sentences(sentences)
    print(" + ".join(sentence.get_text() for sentence in sentences))

    # mark 8
    similarity_matrix = SimilarityMatrix()
    similarity_matrix.fill_from_sentences(sentences)
    text_rank_summarizer = TextRankSummarizer(similarity_matrix)
    text_rank_summarizer.train()
    summary = text_rank_summarizer.make_summary(5)
    print(text_rank_summarizer.make_summary(5))

    # mark 10
    buddy = Buddy(paths_to_texts, stop_words, tuple(punctuation), idf)
    for path in paths_to_texts:
        buddy.add_text_to_database(path)
    print(buddy.reply('Первым генномодифицированным детям на Земле скоро исполнится четыре года. '
                      'Научное сообщество осудило их «создателя» Хэ Цзянькуя, китайские власти и вовсе '
                      'отправили его в тюрьму (на свободу он вышел буквально несколько недель назад), но дискуссии '
                      'о том, как далеко мы готовы зайти, вооружившись инструментами по редактированию генома, '
                      'на этом, естественно, не закончились. В книге «Неестественный отбор: Генная инженерия '
                      'и человек будущего» (издательство «Альпина Паблишер»), переведенной на русский язык Асей '
                      'Лаврушей, научная журналистка Торилл Корнфельт рассказывает о развитии генетических технологий '
                      'и новых этических дилеммах, с которыми нам только предстоит столкнуться. Предлагаем вам '
                      'ознакомиться с фрагментом, посвященным использованию CRISPR для лечения генетических '
                      'заболеваний.На вопрос журналиста New Scientist о том, какие болезни можно лечить с помощью '
                      'CRISPR, исследователь Ирина Конбой отвечает: «Все».'))

    print(buddy.reply('неизвестно'))

    RESULT = summary
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Summaries are not extracted'
