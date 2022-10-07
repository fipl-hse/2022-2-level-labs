"""
Checks the second lab load_stop_words function
"""

from pathlib import Path
import unittest

import pytest

from lab_2_keywords_cooccurrence.main import load_stop_words


class LoadStopWordsTest(unittest.TestCase):
    """
    Tests load_stop_words function
    """

    PROJECT_ROOT = Path(__file__).parent.parent
    ASSETS_PATH = PROJECT_ROOT / 'assets'
    STOP_WORDS_PATH = ASSETS_PATH / 'stopwords.json'
    stop_words_extracted = load_stop_words(STOP_WORDS_PATH)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark10
    def test_load_stop_words_basic_type_verification(self):
        expected = dict
        actual = type(self.stop_words_extracted)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark10
    def test_load_stop_words_basic_keys_verification(self):
        expected = ['bg', 'cs', 'da', 'de', 'el', 'en',
                    'es', 'fa', 'fi', 'fr', 'ga', 'hr',
                    'hu', 'id', 'it', 'lt', 'lv', 'nl',
                    'no', 'pl', 'pt', 'ro', 'ru', 'sk',
                    'sv', 'tr', 'uk']
        actual = list(self.stop_words_extracted.keys())
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark10
    def test_load_stop_words_basic_values_verification(self):
        expected = 7313
        all_stop_words = []
        for stopwords in self.stop_words_extracted.values():
            all_stop_words.extend(stopwords)
        actual = len(all_stop_words)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark10
    def test_load_stop_words_incorrect_path(self):
        expected = None
        actual = load_stop_words('coorrupt path')
        self.assertEqual(expected, actual)
