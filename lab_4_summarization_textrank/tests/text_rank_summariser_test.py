# pylint: disable=protected-access
"""
Checks the fourth lab TextRankSummarizer abstraction
"""
from pathlib import Path
import string
import unittest

import pytest

from lab_4_summarization_textrank.main import (SimilarityMatrix,
                                               SentenceEncoder,
                                               SentencePreprocessor,
                                               TextRankSummarizer)


class TextRankSummarizerTest(unittest.TestCase):
    """
    Tests TextRankSummarizer
    """

    PROJECT_ROOT = Path(__file__).parent.parent
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

    preprocessor = SentencePreprocessor(stop_words, tuple(string.punctuation))
    sentences = preprocessor.get_sentences(text)
    encoder = SentenceEncoder()
    encoder.encode_sentences(sentences)

    graph = SimilarityMatrix()
    graph.fill_from_sentences(sentences)

    extract = ('Он родился 9 марта 1934 года в городе Гжатске (ныне Гагарин) Смоленской области.\n',
               'В июне Гагарин с отличием окончил Саратовский индустриальный техникум, в ' +
               'июле – совершил первый самостоятельный полет на самолете Як-18 и 10 октября ' +
               'того же года окончил Саратовский аэроклуб.\n',
               'Юрий Алексеевич 7 октября 1955 года был призван в ряды Советской Армии и ' +
               'направлен в город Оренбург на учебу в 1-е Чкаловское военно-авиационное ' +
               'училище летчиков имени К.Е.Ворошилова.\n',
               'В конце 1957 года Гагарин прибыл к месту своего назначения – истребительный ' +
               'авиационный полк Северного флота.\n',
               'Гагарин 9 декабря 1959 года написал заявление с просьбой зачислить его в ' +
               'группу кандидатов в космонавты.\n',
               'В 9 часов 7 минут по московскому времени 12 апреля 1961 года с космодрома ' +
               'Байконур стартовал космический корабль "Восток" с пилотом–космонавтом Юрием ' +
               'Алексеевичем Гагариным на борту.\n',
               'С 23 мая 1961 года Гагарин – командир отряда космонавтов.\n',
               'Но сделано это было уже без Юрия Гагарина. 17 февраля 1968 года он защитил ' +
               'диплом в Академии имени Н.Е.Жуковского.\n',
               'С большим трудом добился разрешения самостоятельно пилотировать самолет. 27 ' +
               'марта 1968 года был первый такой полет.\n',
               'Но чтобы не произошло в тот день, ясно только одно – погиб первый космонавт ' +
               'планеты Земля Юрий Алексеевич Гагарин.')

    summary = ''.join(extract)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_rank_initialization(self):
        graph = SimilarityMatrix()
        text_rank = TextRankSummarizer(graph)
        for attribute in ["_graph", "_damping_factor", "_convergence_threshold", "_max_iter", "_scores"]:
            self.assertTrue(hasattr(text_rank, attribute))
        self.assertEqual(text_rank._damping_factor, 0.85)
        self.assertEqual(text_rank._convergence_threshold, 0.0001)
        self.assertEqual(text_rank._max_iter, 50)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_rank_initialization_value_error(self):
        wrong_inputs = [True, 1, 'dlsk;l ', SentenceEncoder(), SentencePreprocessor((), ())]
        for wrong_input in wrong_inputs:
            self.assertRaises(ValueError, TextRankSummarizer, wrong_input)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_text_rank_update_vertex_score_value_error(self):
        text_rank = TextRankSummarizer(self.graph)
        invalid_inputs = [True, (1, 2, 3), SentencePreprocessor((), ()), 4.5]
        for invalid_input in invalid_inputs:
            self.assertRaises(ValueError, text_rank.update_vertex_score, invalid_input, invalid_inputs, invalid_input)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_ranking_pipeline(self):
        text_rank = TextRankSummarizer(self.graph)
        text_rank.train()
        top = text_rank.get_top_sentences(10)
        self.assertEqual([7, 1, 8, 11, 27, 51, 12, 45, 47, 15], [sentence.get_position() for sentence in top])

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_sentences_value_error(self):
        text_rank = TextRankSummarizer(self.graph)
        wrong_inputs = [True, 4.5, 'dlsk;l ', SentenceEncoder(), SentencePreprocessor((), ())]
        for wrong_input in wrong_inputs:
            self.assertRaises(ValueError, text_rank.get_top_sentences, wrong_input)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_make_summary(self):
        text_rank = TextRankSummarizer(self.graph)
        text_rank.train()
        summary = text_rank.make_summary(10)
        self.assertEqual(self.summary, summary)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_make_summary_value_error(self):
        text_rank = TextRankSummarizer(self.graph)
        wrong_inputs = [True, 4.5, 'dlsk;l ', SentenceEncoder(), SentencePreprocessor((), ())]
        for wrong_input in wrong_inputs:
            self.assertRaises(ValueError, text_rank.make_summary, wrong_input)
