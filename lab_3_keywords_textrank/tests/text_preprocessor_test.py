"""
Checks the third lab TextPreprocessor
"""

import unittest

import pytest

from lab_3_keywords_textrank.main import TextPreprocessor


class TextPreprocessorTest(unittest.TestCase):
    """
    Tests TextPreprocessor
    """

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_preprocess_text(self):
        text = "Во времена, Советского Союза."
        text_preprocessor = TextPreprocessor(('во',), (',', '.', '/'))
        preprocessed = text_preprocessor.preprocess_text(text)
        self.assertEqual(preprocessed, ('времена', 'советского', 'союза'))

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_preprocess_text_no_punctuation(self):
        text = "Во времена, Советского Союза."
        text_preprocessor = TextPreprocessor(('во',), ())
        preprocessed = text_preprocessor.preprocess_text(text)
        self.assertEqual(preprocessed, ('времена,', 'советского', 'союза.'))
