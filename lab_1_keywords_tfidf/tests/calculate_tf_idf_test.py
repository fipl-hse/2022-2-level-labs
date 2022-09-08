"""
Checks the first lab calculate tf-idf function
"""

import math
import unittest

import pytest

from lab_1_keywords_tfidf.main import calculate_tfidf


class CalculateTFIDFTest(unittest.TestCase):
    """
    Tests calculating tf-idf function
    """

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tfidf_ideal(self):
        """
        Ideal calculate tf-idf scenario
        """
        term_f = {'this': 2, 'is': 4, 'an': 5, 'example': 2, 'of': 2, 'test': 2,
                  'text': 2, 'contains': 2, 'two': 2, 'parts': 2}

        idf = {
            'this': math.log(3 / 2),
            'is': math.log(3 / 3),
            'an': math.log(3 / 1),
            'example': math.log(3 / 1),
            'of': math.log(3 / 1),
            'test': math.log(3 / 2),
            'text': math.log(3 / 2),
            'contains': math.log(3 / 1),
            'two': math.log(3 / 1),
            'sentences': math.log(3 / 1),
            'written': math.log(3 / 1),
            'on': math.log(3 / 1),
            'english': math.log(3 / 1),
            'simple': math.log(3 / 1),
            'third': math.log(3 / 1),
            'one': math.log(3 / 1),
            'there': math.log(3 / 1),
            'no': math.log(3 / 1),
            'much': math.log(3 / 1),
            'sense': math.log(3 / 1),
            'parts': math.log(3 / 1)
        }

        expected = {'an': 1.0986122886681098,
                    'contains': 1.0986122886681098,
                    'example': 1.0986122886681098,
                    'is': 0.0,
                    'of': 1.0986122886681098,
                    'parts': 1.0986122886681098,
                    'test': 0.4054651081081644,
                    'text': 0.4054651081081644,
                    'this': 0.4054651081081644,
                    'two': 1.0986122886681098}

        actual = calculate_tfidf(term_f, idf)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tfidf_no_tf(self):
        """
        calculate tf-idf, empty tf
        """
        term_f = {}

        idf = {
            'this': math.log(3 / 2),
            'is': math.log(3 / 3),
            'an': math.log(3 / 1),
            'example': math.log(3 / 1),
            'of': math.log(3 / 1),
            'test': math.log(3 / 2),
            'text': math.log(3 / 2),
            'contains': math.log(3 / 1),
            'two': math.log(3 / 1),
            'sentences': math.log(3 / 1),
            'written': math.log(3 / 1),
            'on': math.log(3 / 1),
            'english': math.log(3 / 1),
            'simple': math.log(3 / 1),
            'third': math.log(3 / 1),
            'one': math.log(3 / 1),
            'there': math.log(3 / 1),
            'no': math.log(3 / 1),
            'much': math.log(3 / 1),
            'sense': math.log(3 / 1),
            'parts': math.log(3 / 1)
        }

        expected = None

        actual = calculate_tfidf(term_f, idf)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tfidf_no_idf(self):
        """
        calculate tf-idf, empty idf
        """
        term_f = {'this': 2, 'is': 4, 'an': 5, 'example': 2, 'of': 2, 'test': 2,
                  'text': 2, 'contains': 2, 'two': 2, 'parts': 2}

        idf = {}

        expected = {'an': 3.8501476017100584,
                    'contains': 3.8501476017100584,
                    'example': 3.8501476017100584,
                    'is': 3.8501476017100584,
                    'of': 3.8501476017100584,
                    'parts': 3.8501476017100584,
                    'test': 3.8501476017100584,
                    'text': 3.8501476017100584,
                    'this': 3.8501476017100584,
                    'two': 3.8501476017100584}

        actual = calculate_tfidf(term_f, idf)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tfidf_bad_input(self):
        """
        calculate tf-idf invalid input
        """
        bad_inputs = ['string', (), None, 9, 9.34, True, [None]]
        expected = None
        for bad_input in bad_inputs:
            actual = calculate_tfidf(bad_input, {'a': 2, 'b': 3})
            self.assertEqual(expected, actual)

        for bad_input in bad_inputs:
            actual = calculate_tfidf({'a': 2, 'b': 3}, bad_input)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tfidf_containing_bad_input(self):
        """
        calculate tf-idf scenario with text containing bad input
        """
        term_f = {'this': 0.09091, 'is': 0.09091, 'an': 0.09091, 'example': 0.09091, 'of': 0.09091, 'test': 0.09091,
                  'text': 0.18182, 'contains': 0.09091, 'two': 0.09091, 'parts': 0.09091, None: 0.32}

        idf = {
            'this': math.log(3 / 2),
            'is': math.log(3 / 3),
            'an': math.log(3 / 1),
            'example': math.log(3 / 1),
            'of': math.log(3 / 1),
            'test': math.log(3 / 2),
            'text': math.log(3 / 2),
            'contains': math.log(3 / 1),
            'two': math.log(3 / 1),
            'sentences': math.log(3 / 1),
            'written': math.log(3 / 1),
            'on': math.log(3 / 1),
            'english': math.log(3 / 1),
            'simple': math.log(3 / 1),
            'third': math.log(3 / 1),
            'one': math.log(3 / 1),
            'there': math.log(3 / 1),
            'no': math.log(3 / 1),
            'much': math.log(3 / 1),
            'sense': math.log(3 / 1),
            'parts': math.log(3 / 1)
        }

        expected = None
        actual = calculate_tfidf(term_f, idf)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_1_keywords_tfidf
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_tfidf_no_idf_for_term(self):
        """
        calculate tf-idf when no idf for a term
        """
        term_f = {'this': 2, 'is': 4, 'an': 5, 'example': 2, 'of': 2, 'test': 2,
                  'text': 2, 'contains': 2, 'two': 2, 'parts': 2}

        idf = {
            'is': math.log(3 / 3),
            'an': math.log(3 / 1),
            'example': math.log(3 / 1),
            'of': math.log(3 / 1),
            'test': math.log(3 / 2),
            'text': math.log(3 / 2),
            'contains': math.log(3 / 1),
            'two': math.log(3 / 1),
            'sentences': math.log(3 / 1),
            'written': math.log(3 / 1),
            'on': math.log(3 / 1),
            'english': math.log(3 / 1),
            'simple': math.log(3 / 1),
            'third': math.log(3 / 1),
            'one': math.log(3 / 1),
            'there': math.log(3 / 1),
            'no': math.log(3 / 1),
            'much': math.log(3 / 1),
            'sense': math.log(3 / 1),
            'parts': math.log(3 / 1)
        }

        expected = {'an': 1.0986122886681098,
                    'contains': 1.0986122886681098,
                    'example': 1.0986122886681098,
                    'is': 0.0,
                    'of': 1.0986122886681098,
                    'parts': 1.0986122886681098,
                    'test': 0.4054651081081644,
                    'text': 0.4054651081081644,
                    'this': 3.8501476017100584,
                    'two': 1.0986122886681098}

        actual = calculate_tfidf(term_f, idf)
        self.assertEqual(expected, actual)
