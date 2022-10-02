"""
Checks the second lab generate_stop_words function
"""

import unittest

import pytest

from lab_2_keywords_cooccurrence.main import generate_stop_words


class GenerateStopWordsTest(unittest.TestCase):
    """
    Tests generate_stop_words function
    """

    text = '''aaaaaaaaaa, bbbbbbbbb, cccccccc, ddddddd, eeeeee, fffff,
    gggg, hhh, ll, m, aaaaaaaaaa, bbbbbbbbb, cccccccc, ddddddd, eeeeee, 
    fffff, gggg, hhh, ll, aaaaaaaaaa, bbbbbbbbb, cccccccc, ddddddd,
    eeeeee, fffff, gggg, hhh, aaaaaaaaaa, bbbbbbbbb, cccccccc, ddddddd,
    eeeeee, fffff, gggg, aaaaaaaaaa, bbbbbbbbb, cccccccc, ddddddd, eeeeee,
    fffff, aaaaaaaaaa, bbbbbbbbb, cccccccc, ddddddd, eeeeee, aaaaaaaaaa,
    bbbbbbbbb, cccccccc, ddddddd, aaaaaaaaaa, bbbbbbbbb, cccccccc,
    aaaaaaaaaa, bbbbbbbbb, aaaaaaaaaa'''

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark10
    def test_generate_stop_words_basic(self):
        expected = ['aaaaaaaaaa', 'bbbbbbbbb', 'cccccccc']
        actual = generate_stop_words(self.text, 20)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark10
    def test_generate_stop_words_basic_length_limited(self):
        expected = ['cccccccc']
        actual = generate_stop_words(self.text, 8)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark10
    def test_generate_stop_words_empty_arguments(self):
        expected = None
        actual = generate_stop_words('', 10)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark10
    def test_generate_stop_words_wrong_types(self):
        wrong_types = [4, 4.0, ['l', 'i', 's', 't'], {'dict': 'dict'}]
        expected = None
        for wrong_type in wrong_types:
            actual = generate_stop_words(wrong_type, 10)
            self.assertEqual(expected, actual)
        wrong_types = ['4', 4.0, -2, ['l', 'i', 's', 't'], {'dict': 'dict'}]
        for wrong_type in wrong_types:
            actual = generate_stop_words(self.text, wrong_type)
            self.assertEqual(expected, actual)
