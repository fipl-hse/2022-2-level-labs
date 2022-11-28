# pylint: disable=protected-access
"""
Checks the fourth lab SimilarityMatrix abstraction
"""

import string
import unittest

import pytest

from lab_4_summarization_textrank.main import SimilarityMatrix, SentenceEncoder, SentencePreprocessor


class SimilarityMatrixTest(unittest.TestCase):
    """
    Tests SimilarityMatrix
    """

    text = "Мама мыла раму? Раму долго мыла. Идти можно долго! Можно долго использовать раму?"
    preprocessor = SentencePreprocessor((), tuple(string.punctuation))
    sentences = preprocessor.get_sentences(text)
    encoder = SentenceEncoder()
    encoder.encode_sentences(sentences)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_similarity_matrix_initialization(self):
        matrix = SimilarityMatrix()
        self.assertTrue(hasattr(matrix, "_matrix"))
        self.assertIsInstance(matrix._matrix, list)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_add_edge_ideal(self):

        matrix = SimilarityMatrix()
        matrix.add_edge(self.sentences[0], self.sentences[1])
        matrix.add_edge(self.sentences[0], self.sentences[2])
        matrix.add_edge(self.sentences[1], self.sentences[2])
        matrix.add_edge(self.sentences[3], self.sentences[0])
        self.assertEqual(4, len(matrix._matrix))

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_add_edge_ideal_value_error(self):
        matrix = SimilarityMatrix()
        invalid_inputs = [True, False, (1, 2, 3), {5: 6}, SentencePreprocessor((), ()), SimilarityMatrix()]
        for invalid_input in invalid_inputs:
            self.assertRaises(ValueError, matrix.add_edge, invalid_input, invalid_input)


    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_similarity_score_ideal(self):
        matrix = SimilarityMatrix()
        for index1, sentence1 in enumerate(self.sentences):
            for index2, sentence2 in enumerate(self.sentences):
                if not index1 == index2:
                    matrix.add_edge(sentence1, sentence2)

        self.assertTrue(matrix.get_similarity_score(self.sentences[0], self.sentences[1]))
        self.assertTrue(matrix.get_similarity_score(self.sentences[1], self.sentences[2]))
        self.assertTrue(matrix.get_similarity_score(self.sentences[3], self.sentences[0]))
        self.assertTrue(matrix.get_similarity_score(self.sentences[1], self.sentences[3]))

        self.assertFalse(matrix.get_similarity_score(self.sentences[0], self.sentences[2]))

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_similarity_score_value_error(self):
        matrix = SimilarityMatrix()
        for index1, sentence1 in enumerate(self.sentences):
            for index2, sentence2 in enumerate(self.sentences):
                if not index1 == index2:
                    matrix.add_edge(sentence1, sentence2)
        invalid_inputs = [True, False, (1, 2, 3), {5: 6}, SentencePreprocessor((), ()), SimilarityMatrix()]
        for invalid_input in invalid_inputs:
            self.assertRaises(ValueError, matrix.get_similarity_score, invalid_input, invalid_input)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_similarity_score_nonexistent_vertex(self):
        matrix = SimilarityMatrix()
        self.assertRaises(ValueError, matrix.get_similarity_score, self.sentences[0], self.sentences[2])

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_add_edge_cannot_add_loops(self):
        matrix = SimilarityMatrix()
        self.assertRaises(ValueError, matrix.add_edge, self.sentences[0], self.sentences[0])

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_vertices(self):
        matrix = SimilarityMatrix()
        for index1, sentence1 in enumerate(self.sentences):
            for index2, sentence2 in enumerate(self.sentences):
                if not index1 == index2:
                    matrix.add_edge(sentence1, sentence2)
        self.assertCountEqual(self.sentences, matrix.get_vertices())

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_inout_score(self):
        matrix = SimilarityMatrix()
        for index1, sentence1 in enumerate(self.sentences):
            for index2, sentence2 in enumerate(self.sentences):
                if not index1 == index2:
                    matrix.add_edge(sentence1, sentence2)
        self.assertEqual(matrix.calculate_inout_score(self.sentences[0]), 2)
        self.assertEqual(matrix.calculate_inout_score(self.sentences[1]), 3)
        self.assertEqual(matrix.calculate_inout_score(self.sentences[2]), 2)
        self.assertEqual(matrix.calculate_inout_score(self.sentences[3]), 3)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_inout_score_nonexistent_vertex(self):
        matrix = SimilarityMatrix()
        self.assertRaises(ValueError, matrix.calculate_inout_score, self.sentences[0])

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_inout_score_invalid_input(self):
        matrix = SimilarityMatrix()
        for index1, sentence1 in enumerate(self.sentences):
            for index2, sentence2 in enumerate(self.sentences):
                if not index1 == index2:
                    matrix.add_edge(sentence1, sentence2)
        invalid_inputs = [True, False, (1, 2, 3), {5: 6}, SentencePreprocessor((), ()), SimilarityMatrix()]
        for invalid_input in invalid_inputs:
            self.assertRaises(ValueError, matrix.calculate_inout_score, invalid_input)

    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_fill_from_sentences(self):
        matrix = SimilarityMatrix()
        matrix.fill_from_sentences(self.sentences)
        self.assertEqual(2, matrix.calculate_inout_score(self.sentences[0]))
        self.assertEqual(4, len(matrix.get_vertices()))


    @pytest.mark.lab_4_summarization_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_fill_from_sentences_invalid_input(self):
        invalid_inputs = [True, 1, 4.0, 'мама', (), []]
        matrix = SimilarityMatrix()
        for invalid_input in invalid_inputs:
            self.assertRaises(ValueError, matrix.fill_from_sentences, invalid_input)
