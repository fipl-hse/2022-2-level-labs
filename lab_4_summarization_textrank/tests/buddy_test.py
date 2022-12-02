# pylint: disable=protected-access
"""
Checks the fourth lab Buddy abstraction
"""

import json
import string
import unittest
from pathlib import Path

import pytest

from lab_3_keywords_textrank.main import TextPreprocessor
from lab_4_summarization_textrank.main import (Buddy,
                                               SentenceEncoder,
                                               SentencePreprocessor,
                                               Sentence)
try:
    from lab_4_summarization_textrank.main import IncorrectQueryError, NoRelevantTextsError
except ImportError:
    print('Unable to import non-existent exceptions. Implement them first')


class BuddyTest(unittest.TestCase):
    """
    Tests Buddy
    """

    PROJECT_ROOT = Path(__file__).parent.parent
    ASSETS_PATH = PROJECT_ROOT / 'assets'
    TEXTS_PATH = ASSETS_PATH.joinpath('texts')

    TESTS_PATH = PROJECT_ROOT / 'tests'
    TARGET_ANSWER_PATH = TESTS_PATH / 'expected_answer.txt'

    paths_to_texts = [str(path) for path in TEXTS_PATH.glob('*.txt')]

    STOP_WORDS_PATH = ASSETS_PATH / 'stop_words.txt'
    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        stop_words = tuple(file.read().split('\n'))

    IDF_PATH = ASSETS_PATH / 'IDF.json'
    with open(IDF_PATH, 'r', encoding='utf-8') as file:
        idf = json.load(file)

    buddy = Buddy(paths_to_texts, stop_words, tuple(string.punctuation), idf)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark10
    def test_initialization_ideal(self):
        attributes = ["_stop_words", "_punctuation", "_idf_values", "_text_preprocessor",
                      "_sentence_encoder", "_sentence_preprocessor", "_paths_to_texts",
                      "_knowledge_database"]
        expected_types = [tuple, tuple, dict, TextPreprocessor, SentenceEncoder, SentencePreprocessor, list, dict]
        for attribute, expected_type in zip(attributes, expected_types):
            self.assertTrue(hasattr(self.buddy, attribute))
            self.assertIsInstance(getattr(self.buddy, attribute), expected_type)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark10
    def test_knowledge_database(self):
        self.assertTrue(all(path in self.buddy._knowledge_database for path in self.paths_to_texts))
        texts = 0
        preprocessed = 0
        encoded = 0
        for text_properties in self.buddy._knowledge_database.values():
            self.assertTrue(all(key in text_properties for key in ['sentences', 'keywords', 'summary']))

            self.assertTrue(text_properties['sentences'])
            self.assertIsInstance(text_properties['sentences'], tuple)
            self.assertTrue(all(isinstance(sentence, Sentence) for sentence in text_properties['sentences']))
            for sentence in text_properties['sentences']:
                texts += int(bool(sentence.get_text()))
                preprocessed += int(bool(sentence.get_preprocessed()))
                encoded += int(bool(sentence.get_encoded()))

            self.assertTrue(text_properties['keywords'])
            self.assertIsInstance(text_properties['keywords'], tuple)
            self.assertEqual(len(text_properties['keywords']), 100)
            self.assertTrue(all(isinstance(keyword, str) for keyword in text_properties['keywords']))

            self.assertTrue(text_properties['summary'])
            self.assertIsInstance(text_properties['summary'], str)

        self.assertGreaterEqual(texts, 1382)
        self.assertGreaterEqual(preprocessed, 1381)
        self.assertGreaterEqual(encoded, 1381)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark10
    def test_add_text_to_database_value_error(self):
        invalid_inputs = [True, {5: 7}, ('мама',), 4.0, Sentence('lk;ks;', 4)]
        for invalid_input in invalid_inputs:
            self.assertRaises(ValueError, self.buddy.add_text_to_database, invalid_input)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark10
    def test_find_texts_close_to_keywords(self):
        keywords = ('министр', 'нефть', 'конфликт', 'турки', 'Юрий',
                    'скафандр', 'полет', 'летчик', 'фиксация', 'сертификат')
        expected = ['article_702.txt', 'article_701.txt', 'article_5.txt', 'article_498.txt',
                    'article_495.txt', 'article_705.txt', 'article_704.txt', 'article_703.txt',
                    'article_700.txt', 'article_501.txt', 'article_500.txt', 'article_499.txt',
                    'article_497.txt', 'article_496.txt', 'article_494.txt', 'article_493.txt',
                    'article_492.txt', 'article_491.txt', 'article_490.txt', 'article_432.txt',
                    'article_15.txt', 'article_128.txt', 'article_100.txt']
        response = self.buddy._find_texts_close_to_keywords(keywords, 10)
        for path, ending in zip(response, expected):
            self.assertTrue(path.endswith(ending))

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark10
    def test_find_texts_close_to_keywords_error(self):
        keywords = ('sdgarg', 'aljals', '13[120kdla')
        self.assertRaises(NoRelevantTextsError, self.buddy._find_texts_close_to_keywords, keywords, 10)


    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark10
    def test_find_texts_close_to_keywords_invalid_input(self):
        keywords = ['космонавт', 'полет', 'космос', 'Юрий']
        n_texts = 5

        invalid_inputs = [{5: 6}, 4.0, 'пять', -3]
        for invalid_input in invalid_inputs:
            self.assertRaises(ValueError, self.buddy._find_texts_close_to_keywords, keywords, invalid_input)
        invalid_inputs = [{5: 6}, 4.0, 6, -3]
        for invalid_input in invalid_inputs:
            self.assertRaises(ValueError, self.buddy._find_texts_close_to_keywords, invalid_input, n_texts)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark10
    def test_reply_ideal(self):
        query = 'кто был первый космонавт'
        with open(self.TARGET_ANSWER_PATH, 'r', encoding='utf-8') as file:
            expected_answer = file.read()
        actual_answer = self.buddy.reply(query)
        self.assertEqual(expected_answer, actual_answer)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark10
    def test_reply_invalid_input(self):
        invalid_inputs = [['e'], '', (False, False), 3.45, {1: 5}]
        for invalid_input in invalid_inputs:
            self.assertRaises(IncorrectQueryError, self.buddy.reply, invalid_input)
            self.assertRaises(ValueError, self.buddy.reply, 'как звали первого космонавта?', invalid_input)
