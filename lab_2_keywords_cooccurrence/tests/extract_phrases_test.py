"""
Checks the second lab extract_phrases function
"""

import unittest

import pytest

from lab_2_keywords_cooccurrence.main import extract_phrases


class ExtractPhrasesTest(unittest.TestCase):
    """
    Tests extract phrases function
    """

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_phrases_basic(self):
        text = "Во времена Советского Союза исследование космоса было " \
                "одной из важнейших задач. И именно из СССР была отправлена " \
                "ракета, совершившая первый полет в космос! " \
                "Произошло это 12 апреля 1961 года?"
        expected = ['Во времена Советского Союза исследование космоса было одной из важнейших задач',
                    'И именно из СССР была отправлена ракета',
                    'совершившая первый полет в космос',
                    'Произошло это 12 апреля 1961 года']
        actual = extract_phrases(text)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_phrases_a_lot_of_punctuation(self):
        text = "–~—]Во ;:¡!¿?времена]( ) ⟨⟩}{ &]«»Советского Союза"
        expected = ['Во', 'времена', 'Советского Союза']
        actual = extract_phrases(text)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_phrases_punctuation_only(self):
        text = r""".,;:¡!¿?…⋯‹›«»\\"“”\[\]()⟨⟩}{&]|[-–~—]"""
        expected = []
        actual = extract_phrases(text)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_phrases_empty_input(self):
        text = r""
        expected = None
        actual = extract_phrases(text)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_phrases_wrong_input_type(self):
        wrong_inputs = [4, 4.0, False, ['list'], {'dict': 0}]
        expected = None
        for wrong_input in wrong_inputs:
            actual = extract_phrases(wrong_input)
            self.assertEqual(expected, actual)
