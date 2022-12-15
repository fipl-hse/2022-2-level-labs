"""
Lab 4
Summarize text using TextRank algorithm
"""
from typing import Union
import re

from lab_3_keywords_textrank.main import TextEncoder,\
    TextPreprocessor

PreprocessedSentence = tuple[str, ...]
EncodedSentence = tuple[int, ...]


class Sentence:
    """
    An abstraction over the real-world sentences
    """

    def __init__(self, text: str, position: int) -> None:
        """
        Constructs all the necessary attributes
        """
        if not isinstance(text, str) or not isinstance(position, int):
            raise ValueError
        self._text = text
        self._position = position
        self._preprocessed: tuple[str, ...] = ()
        self._encoded: tuple[int, ...] = ()

    def get_position(self) -> int:
        """
        Returns the attribute
        :return: the position of the sentence in the text
        """
        return self._position

    def set_text(self, text: str) -> None:
        """
        Sets the attribute
        :param text: the text
        :return: None
        """
        if not isinstance(text, str):
            raise ValueError
        self._text = text

    def get_text(self) -> str:
        """
        Returns the attribute
        :return: the text
        """
        return self._text

    def set_preprocessed(self, preprocessed_sentence: PreprocessedSentence) -> None:
        """
        Sets the attribute
        :param preprocessed_sentence: the preprocessed sentence (a sequence of tokens)
        :return: None
        """
        if not isinstance(preprocessed_sentence, tuple):
            raise ValueError
        for i in preprocessed_sentence:
            if not isinstance(i, str):
                raise ValueError
        self._preprocessed = preprocessed_sentence

    def get_preprocessed(self) -> PreprocessedSentence:
        """
        Returns the attribute
        :return: the preprocessed sentence (a sequence of tokens)
        """
        return self._preprocessed

    def set_encoded(self, encoded_sentence: EncodedSentence) -> None:
        """
        Sets the attribute
        :param encoded_sentence: the encoded sentence (a sequence of numbers)
        :return: None
        """
        if not isinstance(encoded_sentence, tuple):
            raise ValueError
        for i in encoded_sentence:
            if not isinstance(i, int):
                raise ValueError
        self._encoded = encoded_sentence

    def get_encoded(self) -> EncodedSentence:
        """
        Returns the attribute
        :return: the encoded sentence (a sequence of numbers)
        """
        return self._encoded


class SentencePreprocessor(TextPreprocessor):
    """
    Class for sentence preprocessing
    """

    def __init__(self, stop_words: tuple[str, ...], punctuation: tuple[str, ...]) -> None:
        """
        Constructs all the necessary attributes
        """
        self._stop_words = stop_words
        if not isinstance(stop_words, tuple):
            raise ValueError
        for i in stop_words:
            if not isinstance(i, str):
                raise ValueError
        self._punctuation = punctuation
        if not isinstance(punctuation, tuple):
            raise ValueError
        for i in punctuation:
            if not isinstance(i, str):
                raise ValueError
        super().__init__(stop_words, punctuation)

    def _split_by_sentence(self, text: str) -> tuple[Sentence, ...]:
        """
        Splits the provided text by sentence
        :param text: the raw text
        :return: a sequence of sentences
        """
        if not isinstance(text, str):
            raise ValueError
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ])", text)
        final = []
        for position, i in enumerate(sentences):
            final.append(Sentence(i.strip(), position))
        return tuple(final)

    def _preprocess_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their preprocessed versions
        :param sentences: a list of sentences
        :return:
        """
        if not isinstance(sentences, tuple):
            raise ValueError
        for one in sentences:
            preprocessed = self.preprocess_text(one.get_text())
            one.set_preprocessed(preprocessed)

    def get_sentences(self, text: str) -> tuple[Sentence, ...]:
        """
        Extracts the sentences from the given text & preprocesses them
        :param text: the raw text
        :return:
        """
        if not isinstance(text, str):
            raise ValueError
        final = self._split_by_sentence(text)
        self._preprocess_sentences(final)
        return final


class SentenceEncoder(TextEncoder):
    """
    A class to encode string sequence into matching integer sequence
    """

    def __init__(self) -> None:
        """
        Constructs all the necessary attributes
        """
        super().__init__()
        self.last_number = 1000

    def _learn_indices(self, tokens: tuple[str, ...]) -> None:
        """
        Fills attributes mapping words and integer equivalents to each other
        :param tokens: a sequence of string tokens
        :return:
        """
        if not isinstance(tokens, tuple):
            raise ValueError
        for i in tokens:
            if not isinstance(i, str):
                raise ValueError
        new_tokens = []
        for i in tokens:
            if i not in self._word2id:
                new_tokens.append(i)
        last_index = len(new_tokens)+self.last_number
        for number, token in enumerate(new_tokens, last_index):
            self._word2id[token] = number
            self._id2word[number] = token

    def encode_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their encoded versions
        :param sentences: a sequence of sentences
        :return: a list of sentences with their preprocessed versions
        """
        if not isinstance(sentences, tuple):
            raise ValueError
        for i in sentences:
            preprocessed = i.get_preprocessed()
            self._learn_indices(preprocessed)
            for one in preprocessed:
                i.set_encoded(tuple(self._word2id[one]))


def calculate_similarity(sequence: Union[list, tuple], other_sequence: Union[list, tuple]) -> float:
    """
    Calculates similarity between two sequences using Jaccard index
    :param sequence: a sequence of items
    :param other_sequence: a sequence of items
    :return: similarity score
    """
    if not isinstance(sequence, dict) or not isinstance(other_sequence, dict):
        raise ValueError
    if not sequence or other_sequence:
        return 0
    similar = 0
    different = 0
    for i in sequence:
        if i in other_sequence:
            similar += 1
        different += 1
    for i in other_sequence:
        if i not in sequence:
            different += 1
    jaccard_index = similar/different
    return jaccard_index


class SimilarityMatrix:
    """
    A class to represent relations between sentences
    """

    _matrix: list[list[float]]

    def __init__(self) -> None:
        """
        Constructs necessary attributes
        """
        self._matrix = []
        self._vertices = []

    def get_vertices(self) -> tuple[Sentence, ...]:
        """
        Returns a sequence of all vertices present in the graph
        :return: a sequence of vertices
        """
        return tuple(self._vertices)

    def calculate_inout_score(self, vertex: Sentence) -> int:
        """
        Retrieves a number of vertices that are similar (i.e. have similarity score > 0) to the input one
        :param vertex
        :return:
        """
        if vertex not in self._vertices:
            raise ValueError
        summary = 0
        i = self._vertices.index(vertex)
        for number in self._matrix[i]:
            if number > 0:
                summary += 1
        return summary

    def add_edge(self, vertex1: Sentence, vertex2: Sentence) -> None:
        """
        Adds or overwrites an edge in the graph between the specified vertices
        :param vertex1:
        :param vertex2:
        :return:
        """
        if not isinstance(vertex1, Sentence) or not isinstance(vertex2, Sentence):
            raise ValueError
        if vertex1 == vertex2:
            raise ValueError
        for vertex in vertex1, vertex2:
            if vertex not in self._matrix:
                self._matrix.append([])

        for edges_list in self._matrix:
            if len(edges_list) < len(self._vertices):
                edges_list.extend([0 for _ in range(len(self._vertices) - len(edges_list))])

        idx1 = self._vertices.index(vertex1)
        idx2 = self._vertices.index(vertex2)
        jaccard_index = calculate_similarity(vertex1.get_encoded(), vertex2.get_encoded())
        self._matrix[idx1][idx2] = jaccard_index
        self._matrix[idx2][idx1] = jaccard_index

    def get_similarity_score(self, sentence: Sentence, other_sentence: Sentence) -> float:
        """
        Gets the similarity score for two sentences from the matrix
        :param sentence
        :param other_sentence
        :return: the similarity score
        """
        if sentence not in self._matrix or other_sentence not in self._matrix:
            raise ValueError
        first = self._vertices.index(sentence)
        second = self._vertices.index(other_sentence)
        return self._matrix[first][second]

    def fill_from_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Updates graph instance with vertices and edges extracted from sentences
        :param sentences
        :return:
        """
        if not isinstance(sentences, tuple):
            raise ValueError
        for sentence in sentences:
            for other_sentence in sentences:
                if sentence.get_encoded() != other_sentence.get_encoded():
                    self.add_edge(sentence, other_sentence)


class TextRankSummarizer:
    """
    TextRank for summarization
    """

    _scores: dict[Sentence, float]
    _graph: SimilarityMatrix

    def __init__(self, graph: SimilarityMatrix) -> None:
        """
        Constructs all the necessary attributes
        :param graph: the filled instance of the similarity matrix
        """
        self._graph = graph
        self._damping_factor = 0.85
        self._convergence_threshold = 0.0001
        self._max_iter = 50
        self._scores = {}

    def update_vertex_score(
            self, vertex: Sentence, incidental_vertices: list[Sentence], scores: dict[Sentence, float]
    ) -> None:
        """
        Changes vertex significance score using algorithm-specific formula
        :param vertex: a sentence
        :param incidental_vertices: vertices with similarity score > 0 for vertex
        :param scores: current vertices scores
        :return:
        """
        summa = sum((1 / self._graph.calculate_inout_score(inc_vertex)) * scores[inc_vertex]
                    for inc_vertex in incidental_vertices)
        self._scores[vertex] = summa * self._damping_factor + (1 - self._damping_factor)

    def train(self) -> None:
        """
        Iteratively computes significance scores for vertices
        """
        vertices = self._graph.get_vertices()
        for vertex in vertices:
            self._scores[vertex] = 1.0

        for iteration in range(self._max_iter):
            prev_score = self._scores.copy()
            for scored_vertex in vertices:
                similar_vertices = [vertex for vertex in vertices
                                    if self._graph.get_similarity_score(scored_vertex, vertex) > 0]
                self.update_vertex_score(scored_vertex, similar_vertices, prev_score)
            abs_score_diff = [abs(i - j) for i, j in zip(prev_score.values(), self._scores.values())]

            if sum(abs_score_diff) <= self._convergence_threshold:  # convergence condition
                print("Converging at iteration " + str(iteration) + "...")
                break

    def get_top_sentences(self, n_sentences: int) -> tuple[Sentence, ...]:
        """
        Retrieves top n most important sentences in the encoded text
        :param n_sentences: number of sentence to retrieve
        :return: a sequence of sentences
        """
        if not isinstance(n_sentences, int):
            raise ValueError
        sort = sorted(self._scores, key=lambda i: self._scores[i])[:n_sentences]
        return tuple(sort)

    def make_summary(self, n_sentences: int) -> str:
        """
        Constructs summary from the most important sentences
        :param n_sentences: number of sentences to include in the summary
        :return: summary
        """
        if not isinstance(n_sentences, int):
            raise ValueError
        sentences = sorted(self.get_top_sentences(n_sentences), key=lambda elem: elem.get_position())
        final = []
        for one in sentences:
            final.append(one.get_text())
        return '\n'.join(final)

class Buddy:
    """
    (Almost) All-knowing entity
    """

    def __init__(
            self,
            paths_to_texts: list[str],
            stop_words: tuple[str, ...],
            punctuation: tuple[str, ...],
            idf_values: dict[str, float],
    ):
        """
        Constructs all the necessary attributes
        :param paths_to_texts: paths to the texts from which to learn
        :param stop_words: a sequence of stop words
        :param punctuation: a sequence of punctuation symbols
        :param idf_values: pre-computed IDF values
        """
        pass

    def add_text_to_database(self, path_to_text: str) -> None:
        """
        Adds the given text to the existing database
        :param path_to_text
        :return:
        """
        pass

    def _find_texts_close_to_keywords(self, keywords: tuple[str, ...], n_texts: int) -> tuple[str, ...]:
        """
        Finds texts that are similar (i.e. contain the same keywords) to the given keywords
        :param keywords: a sequence of keywords
        :param n_texts: number of texts to find
        :return: the texts' ids
        """
        pass

    def reply(self, query: str, n_summaries: int = 3) -> str:
        """
        Replies to the query
        :param query: the query
        :param n_summaries: the number of summaries to include in the answer
        :return: the answer
        """
        pass
