"""
Checks the fourth lab Sentence abstraction
"""

import unittest

import pytest

from lab_4_summarization_textrank.main import Sentence


class SentenceTest(unittest.TestCase):
    """
    Tests Sentence
    """

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_sentence_initialization_ideal(self):
        sentence = Sentence('Мама мыла раму', 0)
        for attribute in ["_text", "_preprocessed", "_encoded", "_position"]:
            self.assertTrue(hasattr(sentence, attribute), f"Sentence misses {attribute} attribute")

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_sentence_initialization_value_error(self):

        def sentence_initialization(argument1, argument2):
            _ = Sentence(argument1, argument2)

        text_inputs = [1, True, (1, 2,), [True, 8], 4.0]
        position_inputs = ['ааа', True, (1, 2,), [True, 8]]
        for wrong_input in text_inputs:
            self.assertRaises(ValueError, sentence_initialization, wrong_input, 0)
        for wrong_input in position_inputs:
            self.assertRaises(ValueError, sentence_initialization, 'Мама мыла раму', wrong_input)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_sentence_setting_getting_attributes(self):
        text1 = "Мама мыла раму"
        text2 = "Рама мыла маму"
        preprocessed = ("мама", "мыла", "раму")
        encoded = (1, 2, 3)

        sentence = Sentence('Мама мыла раму', 0)
        self.assertEqual(sentence.get_text(), text1)

        sentence.set_preprocessed(preprocessed)
        self.assertEqual(sentence.get_preprocessed(), preprocessed)

        sentence.set_encoded(encoded)
        self.assertEqual(sentence.get_encoded(), encoded)

        sentence.set_text(text2)
        self.assertEqual(sentence.get_text(), text2)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_sentence_setting_attributes_value_error(self):
        text_wrong_inputs = [1, (1, 2,), True, ['d', 'a']]
        preprocessed_wrong_inputs = [1, (1, 2,), True, ['d', 'a']]
        encoded_wrong_inputs = [1, ('a', 'b',), True, ['d', 'a']]

        sentence = Sentence('Мама мыла раму', 0)

        for text_wrong_input in text_wrong_inputs:
            self.assertRaises(ValueError, sentence.set_text, text_wrong_input)

        for preprocessed_wrong_input in preprocessed_wrong_inputs:
            self.assertRaises(ValueError, sentence.set_preprocessed, preprocessed_wrong_input)

        for encoded_wrong_input in encoded_wrong_inputs:
            self.assertRaises(ValueError, sentence.set_encoded, encoded_wrong_input)
