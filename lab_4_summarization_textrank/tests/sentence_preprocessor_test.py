# pylint: disable=protected-access
"""
Checks the fourth lab SentencePreprocessor abstraction
"""

import string
import unittest

import pytest

from lab_4_summarization_textrank.main import SentencePreprocessor, Sentence


class SentencePreprocessorTest(unittest.TestCase):
    """
    Tests SentencePreprocessor
    """

    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor " \
           "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud " \
           "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure " \
           "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. " \
           "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit " \
           "anim id est laborum."

    sentences = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut "
                 "labore et dolore magna aliqua.",
                 "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
                 "commodo consequat.",
                 "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat "
                 "nulla pariatur.",
                 "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit "
                 "anim id est laborum."]

    preprocessed_sentences = [('lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 'adipiscing', 'elit', 'eiusmod',
                               'tempor', 'incididunt', 'labore', 'dolore', 'magna', 'aliqua'),
                              ('enim', 'ad', 'minim', 'veniam', 'quis', 'nostrud', 'exercitation', 'ullamco',
                               'laboris', 'nisi', 'aliquip', 'commodo', 'consequat'),
                              ('duis', 'aute', 'irure', 'dolor', 'reprehenderit', 'voluptate', 'velit', 'cillum',
                               'dolore',
                               'eu', 'fugiat', 'nulla', 'pariatur'),
                              ('excepteur', 'sint', 'occaecat', 'cupidatat', 'proident', 'sunt', 'culpa', 'officia',
                               'deserunt', 'mollit', 'anim', 'id', 'laborum')]

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_sentence_preprocessor_initialization_ideal(self):
        stop_words = ('and', 'of', 'the')
        punctuation = (',', '!', '?')
        preprocessor = SentencePreprocessor(stop_words, punctuation)

        for attribute in ["_stop_words", "_punctuation"]:
            self.assertTrue(hasattr(preprocessor, attribute), f"Sentence preprocessor misses {attribute} attribute")

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_sentence_preprocessor_initialization_value_error(self):

        def sentence_preprocessor_initialization(argument1, argument2):
            _ = SentencePreprocessor(argument1, argument2)

        wrong_inputs = [1, (1, 2,), True, ['d', 'a']]
        for wrong_input in wrong_inputs:
            self.assertRaises(ValueError, sentence_preprocessor_initialization,
                              wrong_input, ('1', '2', '3'))
            self.assertRaises(ValueError, sentence_preprocessor_initialization,
                              ('1', '2', '3'), wrong_input)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_split_by_sentence_ideal(self):
        preprocessor = SentencePreprocessor((), tuple(string.punctuation))
        output = preprocessor._split_by_sentence(self.text)
        for index, sentence in enumerate(output):
            self.assertEqual(self.sentences[index], sentence.get_text())

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_split_by_sentence_value_error(self):
        preprocessor = SentencePreprocessor((), tuple(string.punctuation))

        wrong_inputs = [1, (1, 2,), True, ['d', 'a'], 1.0]
        for wrong_input in wrong_inputs:
            self.assertRaises(ValueError, preprocessor._split_by_sentence, wrong_input)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_sentences_ideal(self):
        stop_words = ('sed', 'ex', 'ut', 'est', 'non', 'in', 'qui', 'esse', 'ea', 'et', 'do')
        preprocessor = SentencePreprocessor(stop_words, tuple(string.punctuation) + tuple('”',))
        output = preprocessor.get_sentences(self.text)
        for index, sentence in enumerate(output):
            self.assertEqual(self.preprocessed_sentences[index], sentence.get_preprocessed())

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_sentences_value_error(self):
        stop_words = ('sed', 'ex', 'ut', 'est', 'non', 'in', 'qui', 'esse', 'ea', 'et', 'do')
        preprocessor = SentencePreprocessor(stop_words, tuple(string.punctuation))
        wrong_inputs = [1, (1, 2,), True, ['d', 'a'], 1.0]
        for wrong_input in wrong_inputs:
            self.assertRaises(ValueError, preprocessor.get_sentences, wrong_input)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_preprocess_sentences_ideal(self):
        stop_words = ('sed', 'ex', 'ut', 'est', 'non', 'in', 'qui', 'esse', 'ea', 'et', 'do')
        preprocessor = SentencePreprocessor(stop_words, tuple(string.punctuation))
        sentences = tuple(Sentence(sentence, position) for position, sentence in enumerate(self.sentences))
        self.assertEqual(None, preprocessor._preprocess_sentences(sentences))
        self.assertTrue(all(sentence.get_preprocessed() for sentence in sentences))

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_preprocess_sentences_value_error(self):
        stop_words = ('sed', 'ex', 'ut', 'est', 'non', 'in', 'qui', 'esse', 'ea', 'et', 'do')
        preprocessor = SentencePreprocessor(stop_words, tuple(string.punctuation))
        self.assertRaises(ValueError, preprocessor._preprocess_sentences, self.sentences)
