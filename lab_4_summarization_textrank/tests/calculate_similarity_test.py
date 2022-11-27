"""
Checks the fourth lab similarity calculation function
"""

import unittest

import pytest

from lab_4_summarization_textrank.main import calculate_similarity


class CalculateSimilarityTest(unittest.TestCase):
    """
    Tests calculate_similarity function
    """

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_similarity_ideal(self):
        self.assertEqual(0.5714285714285714,
                         calculate_similarity([1, 2, 3, 4, 5],
                                              [5, 8, 3, 7, 2, 1, 8]))
        self.assertEqual(calculate_similarity([1, 2, 3, 4, 5], [5, 8, 3, 7, 2, 1, 8]),
                         calculate_similarity([5, 8, 3, 7, 2, 1, 8], [1, 2, 3, 4, 5]))
        self.assertEqual(0, calculate_similarity([], [5, 8, 3, 7, 2, 1, 8]))
        self.assertEqual(0, calculate_similarity([], []))
        self.assertEqual(0, calculate_similarity([1, 2, 3], [4, 5, 6]))

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_similarity_non_iterable(self):
        sequence = [1, 2, 3, 4]
        for wrong_input in [4, 4.0, True]:
            self.assertRaises(ValueError, calculate_similarity, wrong_input, sequence)
            self.assertRaises(ValueError, calculate_similarity, sequence, wrong_input)
