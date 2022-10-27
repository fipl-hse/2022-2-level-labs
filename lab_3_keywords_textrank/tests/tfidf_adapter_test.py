"""
Checks the third lab TFIDFAdapter
"""

import unittest
from unittest import mock

import pytest

from lab_3_keywords_textrank.main import TFIDFAdapter


class TFIDFAdapterTest(unittest.TestCase):
    """
    Tests TFIDFAdapter
    """

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark10
    def test_pipeline(self):
        tokens = ('weather', 'sunny', 'man', 'happy', 'weather', 'man')
        idf = {
            'weather': 0.8,
            'sunny': 0.2,
            'man': 0.1,
            'happy': 0.3
        }

        tfidf = TFIDFAdapter(tokens, idf)
        tfidf.train()
        top = tfidf.get_top_keywords(1)
        self.assertIsInstance(top[0], str)
        self.assertEqual(top, ('weather',))

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark10
    def test_empty_outputs(self):

        funcs_to_mock = ('calculate_frequencies', 'calculate_tf', 'calculate_tfidf')

        for func in funcs_to_mock:
            with mock.patch(f'lab_3_keywords_textrank.main.{func}') as mock_func:
                mock_func.return_value = None
                tokens = ('weather', 'sunny', 'man', 'happy', 'weather', 'man')
                idf = {
                    'weather': 0.8,
                    'sunny': 0.2,
                    'man': 0.1,
                    'happy': 0.3
                }

                tfidf = TFIDFAdapter(tokens, idf)

                self.assertEqual(tfidf.train(), -1)
