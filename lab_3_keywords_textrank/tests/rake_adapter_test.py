"""
Checks the third lab RAKEAdapter
"""

import unittest
from unittest import mock

import pytest

from lab_3_keywords_textrank.main import RAKEAdapter


class RAKEAdapterTest(unittest.TestCase):
    """
    Tests RAKEAdapter
    """

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark10
    def test_pipeline(self):
        text = "Во времена Советского Союза исследование космоса было " \
                "одной из важнейших задач. И именно из СССР была отправлена " \
                "ракета, совершившая первый полет в космос! " \
                "Произошло это 12 апреля 1961 года?"
        stop_words = ('во', 'было', 'из', 'и', 'в')

        rake = RAKEAdapter(text, stop_words)
        rake.train()
        top = rake.get_top_keywords(11)

        self.assertTrue(all(isinstance(keyword, str) for keyword in top))

        top_words = ('12', '1961', 'апреля', 'года', 'произошло', 'это',
                     'времена', 'исследование', 'космоса', 'советского', 'союза')

        self.assertEqual(top, top_words)

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark10
    def test_empty_outputs(self):

        funcs_to_mock = ('extract_phrases', 'extract_candidate_keyword_phrases',
                         'calculate_frequencies_for_content_words',
                         'calculate_word_degrees', 'calculate_word_scores')

        for func in funcs_to_mock:
            with mock.patch(f'lab_3_keywords_textrank.main.{func}') as mock_func:
                mock_func.return_value = None
                text = "Во времена Советского Союза исследование космоса было " \
                       "одной из важнейших задач. И именно из СССР была отправлена " \
                       "ракета, совершившая первый полет в космос! " \
                       "Произошло это 12 апреля 1961 года?"
                stop_words = ('во', 'было', 'из', 'и', 'в')

                rake = RAKEAdapter(text, stop_words)

                rake.train()

                self.assertEqual(rake.train(), -1)
