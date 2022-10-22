"""
Checks the third lab extract_pairs function
"""

import unittest

import pytest

from lab_3_keywords_textrank.main import extract_pairs


class ExtractPairsTest(unittest.TestCase):
    """
    Tests extract_pairs function
    """

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_ideal_case(self):
        actual = extract_pairs((1, 2, 3, 4, 3, 2), 3)
        pairs = ((1, 2), (1, 3), (2, 3), (3, 4), (2, 4), (4, 3))
        self.assertCountEqual(actual, pairs)

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_incorrect_input(self):
        expected = None
        actual = extract_pairs((), 3)
        self.assertEqual(expected, actual)

        wrong_inputs = [0, 1, -100, 2.24]
        for wrong_input in wrong_inputs:
            actual = extract_pairs((1, 2, 3, 4, 3), wrong_input)
            self.assertEqual(expected, actual)
