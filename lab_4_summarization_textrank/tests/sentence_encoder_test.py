# pylint: disable=protected-access
"""
Checks the fourth lab SentenceEncoder abstraction
"""

from collections import Counter
import string
import unittest

import pytest

from lab_4_summarization_textrank.main import SentenceEncoder, SentencePreprocessor, Sentence


class SentenceEncoderTest(unittest.TestCase):
    """
    Tests SentenceEncoder
    """

    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor " \
           "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud " \
           "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure " \
           "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. " \
           "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit " \
           "anim id est laborum."

    preprocessor = SentencePreprocessor(('sed', 'ex', 'ut', 'est', 'non', 'in', 'qui', 'esse', 'ea', 'et', 'do'),
                                         tuple(string.punctuation))
    sentences = preprocessor.get_sentences(text)

    encoded_sentences = [(1011, 1014, 1010, 1015, 1006, 1008, 1007, 1002, 1003, 1001, 1012, 1009, 1004, 1005, 1013),
                         (1027, 1028, 1025, 1016, 1019, 1017, 1020, 1023, 1021, 1022, 1026, 1024, 1018),
                         (1036, 1038, 1037, 1010, 1029, 1033, 1031, 1039, 1004, 1034, 1032, 1035, 1030),
                         (1041, 1046, 1051, 1052, 1050, 1047, 1042, 1044, 1048, 1040, 1045, 1043, 1049)]

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_sentence_encoder_initialization_ideal(self):
        encoder = SentenceEncoder()
        for attribute in ["_word2id", "_id2word"]:
            self.assertTrue(hasattr(encoder, attribute), f"Sentence encoder misses {attribute} attribute")

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_encode_sentences_return_value(self):
        encoder = SentenceEncoder()
        self.assertEqual(None, encoder.encode_sentences(self.sentences))

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_encode_sentences_ideal(self):
        encoder = SentenceEncoder()
        encoder.encode_sentences(self.sentences)
        all_identifiers_actual = []
        for sentence in self.sentences:
            all_identifiers_actual.extend(list(sentence.get_encoded()))
        all_identifiers_expected = []
        for sentence in self.encoded_sentences:
            all_identifiers_expected.extend(list(sentence))
        actual_counts = Counter(all_identifiers_actual).values()
        expected_counts = Counter(all_identifiers_expected).values()
        self.assertCountEqual(expected_counts, actual_counts)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_encode_sentences_value_error(self):
        encoder = SentenceEncoder()
        for wrong_input in ['text', 1, 4.0, ['word1', 'word2'], [Sentence('word1', 0), Sentence('word2', 1)]]:
            self.assertRaises(ValueError, encoder.encode_sentences, wrong_input)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_learn_indices_return_value(self):
        encoder = SentenceEncoder()
        invalid_inputs = [['alksla'], 'мама мыла раму', True, 1, 0.0, {6: 5}]
        for invalid_input in invalid_inputs:
            self.assertRaises(ValueError, encoder._learn_indices, invalid_input)
