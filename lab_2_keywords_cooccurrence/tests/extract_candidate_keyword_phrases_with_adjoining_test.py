"""
Checks the second lab extract_candidate_keyword_phrases_with_adjoining function
"""

import unittest

import pytest

from lab_2_keywords_cooccurrence.main import extract_candidate_keyword_phrases_with_adjoining


class ExtractCandidateKeywordPhrasesWithAdjoiningTest(unittest.TestCase):
    """
    Tests extract_candidate_keyword_phrases_with_adjoining function
    """

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_candidate_keyword_phrases_with_adjoining_basic(self):
        candidate_keyword_phrases = [('времена', 'советского', 'союза', 'исследование', 'космоса'),
                                     ('одной',), ('важнейших', 'задач'), ('времена', 'союза', 'прошли'),
                                     ('одной',), ('важнейших', 'задач'), ('исследование', 'космоса'),
                                     ('ящик',), ('железа',), ('самое',), ('ящик',), ('железа',)]
        phrases = ['Во времена Советского Союза исследование космоса было одной из важнейших задач',
                   'И именно из СССР была отправлена ракета',
                   'совершившая первый полет в космос',
                   'Произошло это 12 апреля 1961 года',
                   'было это одной из важнейших задач',
                   'Ящик для железа не то же самое, что ящик из железа ']

        expected = [('одной', 'из', 'важнейших', 'задач')]
        actual = extract_candidate_keyword_phrases_with_adjoining(candidate_keyword_phrases, phrases)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_candidate_keyword_phrases_with_adjoining_phrases_do_not_contain_keywords(self):
        candidate_keyword_phrases = [('времена', 'советского', 'союза', 'исследование', 'космоса'),
                                     ('одной',), ('важнейших', 'задач'), ('времена', 'союза', 'прошли'),
                                     ('одной',), ('важнейших', 'задач'), ('исследование', 'космоса'),
                                     ('ящик',), ('железа',), ('самое',), ('ящик',), ('железа',)]
        phrases = ['Во времена Советского Союза исследование космоса было одной из важнейших задач']
        expected = []
        actual = extract_candidate_keyword_phrases_with_adjoining(candidate_keyword_phrases, phrases)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_candidate_keyword_phrases_with_adjoining_phrases_empty_phrase(self):
        candidate_keyword_phrases = [('времена', 'советского', 'союза', 'исследование', 'космоса'),
                                     ('одной',), ('важнейших', 'задач'), ('времена', 'союза', 'прошли'),
                                     ('одной',), ('важнейших', 'задач'), ('исследование', 'космоса'),
                                     ('ящик',), ('железа',), ('самое',), ('ящик',), ('железа',)]
        phrases = ['']
        expected = []
        actual = extract_candidate_keyword_phrases_with_adjoining(candidate_keyword_phrases, phrases)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_candidate_keyword_phrases_with_adjoining_empty_arguments(self):
        candidate_keyword_phrases = [('времена', 'советского', 'союза', 'исследование', 'космоса'),
                                     ('одной',), ('важнейших', 'задач'), ('времена', 'союза', 'прошли'),
                                     ('одной',), ('важнейших', 'задач'), ('исследование', 'космоса'),
                                     ('ящик',), ('железа',), ('самое',), ('ящик',), ('железа',)]
        phrases = ['Во времена Советского Союза исследование космоса было одной из важнейших задач',
                   'И именно из СССР была отправлена ракета',
                   'совершившая первый полет в космос',
                   'Произошло это 12 апреля 1961 года',
                   'было это одной из важнейших задач',
                   'Ящик для железа не то же самое, что ящик из железа ']
        expected = None

        actual = extract_candidate_keyword_phrases_with_adjoining(candidate_keyword_phrases, [])
        self.assertEqual(expected, actual)
        actual = extract_candidate_keyword_phrases_with_adjoining([], phrases)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_candidate_keyword_phrases_with_adjoining_wrong_argument_types(self):
        candidate_keyword_phrases = [('времена', 'советского', 'союза', 'исследование', 'космоса'),
                                     ('одной',), ('важнейших', 'задач'), ('времена', 'союза', 'прошли'),
                                     ('одной',), ('важнейших', 'задач'), ('исследование', 'космоса'),
                                     ('ящик',), ('железа',), ('самое',), ('ящик',), ('железа',)]
        phrases = ['Во времена Советского Союза исследование космоса было одной из важнейших задач',
                   'И именно из СССР была отправлена ракета',
                   'совершившая первый полет в космос',
                   'Произошло это 12 апреля 1961 года',
                   'было это одной из важнейших задач',
                   'Ящик для железа не то же самое, что ящик из железа ']
        expected = None

        wrong_inputs = [4, 4.0, False, {0: 1}]
        for wrong_input in wrong_inputs:
            actual = extract_candidate_keyword_phrases_with_adjoining(candidate_keyword_phrases, wrong_input)
            self.assertEqual(expected, actual)
            actual = extract_candidate_keyword_phrases_with_adjoining(wrong_input, phrases)
            self.assertEqual(expected, actual)
