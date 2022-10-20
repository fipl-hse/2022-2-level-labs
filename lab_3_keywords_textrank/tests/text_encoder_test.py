"""
Checks the third lab TextEncoder
"""

import unittest

import pytest

from lab_3_keywords_textrank.main import TextEncoder


class TextEncoderTest(unittest.TestCase):
    """
    Tests TextEncoder
    """

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_encode(self):
        text_encoder = TextEncoder()
        encoded = text_encoder.encode(('время', 'советского', 'союза'))
        self.assertEqual(len(encoded), 3)
        for encoded_token in encoded:
            self.assertIsInstance(encoded_token, int)
            self.assertGreaterEqual(encoded_token, 1000)

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_encode_incorrect_input(self):
        text_encoder = TextEncoder()

        expected = None
        actual = text_encoder.encode(())
        self.assertEqual(expected, actual)

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_decode(self):
        text_encoder = TextEncoder()
        encoded = text_encoder.encode(('время', 'советского', 'союза'))

        decoded = text_encoder.decode(encoded)
        self.assertEqual(decoded, ('время', 'советского', 'союза'))

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_decode_nonexistent_id(self):
        text_encoder = TextEncoder()
        encoded = text_encoder.encode(('время', 'советского', 'союза'))

        decoded = text_encoder.decode((*encoded, 127469123))
        self.assertEqual(decoded, None)
