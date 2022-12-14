"""
Lab 4
Summarize text using TextRank algorithm
"""
from typing import Union, Any, Type
import re
from lab_3_keywords_textrank.main import TextEncoder, \
    TextPreprocessor

PreprocessedSentence = tuple[str, ...]
EncodedSentence = tuple[int, ...]


def check_type(user_input: Any, correct_type: Type) -> bool:
    """
    Checks type of users input
    """
    if isinstance(user_input, correct_type) and not isinstance(user_input, bool):
        return True
    raise ValueError


def check_sequence(user_input: Any, correct_type: Type, element_type: Type) -> bool:
    """
    Checks type of users input and its elements
    """
    if isinstance(user_input, correct_type) and all(isinstance(element, element_type) for element in user_input):
        return True
    raise ValueError


class Sentence:
    """
    An abstraction over the real-world sentences
    """

    def __init__(self, text: str, position: int) -> None:
        """
        Constructs all the necessary attributes
        """
        if check_type(text, str) and check_type(position, int):
            self._text = text
            self._position = position
            self._preprocessed: tuple[str, ...] = ('',)
            self._encoded: tuple[int, ...] = (0,)

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
        if check_type(text, str):
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
        if check_sequence(preprocessed_sentence, tuple, str):
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
        if check_sequence(encoded_sentence, tuple, int):
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
        if check_sequence(stop_words, tuple, str) and punctuation and check_sequence(punctuation, tuple, str):
            super().__init__(stop_words, punctuation)
            self._stop_words = stop_words
            self._punctuation = punctuation

    def _split_by_sentence(self, text: str) -> tuple[Sentence, ...]:
        """
        Splits the provided text by sentence
        :param text: the raw text
        :return: a sequence of sentences
        """
        check_type(text, str)
        list_of_sentences = []
        text = text.replace('\n', ' ').replace('  ', ' ')
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZА-Я])', text)
        for number, sentence in enumerate(sentences):
            if sentence:
                list_of_sentences.append(Sentence(sentence, number))
        return tuple(list_of_sentences)

    def _preprocess_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their preprocessed versions
        :param sentences: a list of sentences
        :return:
        """
        if sentences and check_sequence(sentences, tuple, Sentence):
            for sentence in sentences:
                preprocessed_sentence = self.preprocess_text(sentence.get_text())
                sentence.set_preprocessed(preprocessed_sentence)

    def get_sentences(self, text: str) -> tuple[Sentence, ...]:
        """
        Extracts the sentences from the given text & preprocesses them
        :param text: the raw text
        :return:
        """
        sentences = self._split_by_sentence(text)
        self._preprocess_sentences(sentences)
        return sentences


class SentenceEncoder(TextEncoder):
    """
    A class to encode string sequence into matching integer sequence
    """

    def _learn_indices(self, tokens: tuple[str, ...]) -> None:
        """
        Fills attributes mapping words and integer equivalents to each other
        :param tokens: a sequence of string tokens
        :return:
        """
        if check_sequence(tokens, tuple, str):
            start = 1000
            tokens2 = (token for token in tokens if token not in self._word2id)
            for index, token in enumerate(tokens2, start + len(self._word2id)):
                self._word2id[token] = index
                self._id2word[index] = token

    def encode_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their encoded versions
        :param sentences: a sequence of sentences
        :return: a list of sentences with their preprocessed versions
        """
        if check_sequence(sentences, tuple, Sentence):
            for sentence in sentences:
                self._learn_indices(sentence.get_preprocessed())
                sentence.set_encoded(tuple(self._word2id[word] for word in sentence.get_preprocessed()))


def calculate_similarity(sequence: Union[list, tuple], other_sequence: Union[list, tuple]) -> float:
    """
    Calculates similarity between two sequences using Jaccard index
    :param sequence: a sequence of items
    :param other_sequence: a sequence of items
    :return: similarity score
    """
    if not isinstance(sequence, (list, tuple)) or not isinstance(other_sequence, (list, tuple)):
        raise ValueError
    if not sequence or not other_sequence:
        return 0
    one_set = set(sequence)
    other_set = set(other_sequence)
    return len(one_set.intersection(other_set)) / len(one_set.union(other_set))


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
        check_type(vertex, Sentence)
        if vertex not in self._vertices:
            raise ValueError
        index = self._vertices.index(vertex)
        inout_score = len(self._matrix) - self._matrix[index].count(0) - 1
        return inout_score

    def add_edge(self, vertex1: Sentence, vertex2: Sentence) -> None:
        """
        Adds or overwrites an edge in the graph between the specified vertices
        :param vertex1:
        :param vertex2:
        :return:
        """
        if check_type(vertex1, Sentence) and check_type(vertex2, Sentence):
            if vertex1 == vertex2:
                raise ValueError
            for vertex in vertex1, vertex2:
                if vertex not in self._vertices:
                    self._vertices.append(vertex)
                    new_row = [calculate_similarity(vertex.get_encoded(), other.get_encoded())
                               for other in self._vertices]
                    self._matrix.append(new_row)
                    for i in range(len(self._matrix) - 1):
                        self._matrix[i].append(self._matrix[-1][i])

    def get_similarity_score(self, sentence: Sentence, other_sentence: Sentence) -> float:
        """
        Gets the similarity score for two sentences from the matrix
        :param sentence
        :param other_sentence
        :return: the similarity score
        """
        check_type(sentence, Sentence)
        check_type(other_sentence, Sentence)
        if sentence not in self._vertices or other_sentence not in self._vertices:
            raise ValueError
        index1 = self._vertices.index(sentence)
        index2 = self._vertices.index(other_sentence)
        return self._matrix[index1][index2]

    def fill_from_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Updates graph instance with vertices and edges extracted from sentences
        :param sentences
        :return:
        """
        if check_sequence(sentences, tuple, Sentence):
            if not sentences:
                raise ValueError
            if check_type(sentences, tuple):
                for index1, sentence1 in enumerate(sentences):
                    for index2, sentence2 in enumerate(sentences):
                        if index2 == index1:
                            break
                        self.add_edge(sentence1, sentence2)


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
        if check_type(graph, SimilarityMatrix):
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
        if check_type(vertex, Sentence) and check_type(scores, dict):
            summary = sum(self._scores[inc_vertex] / (1 + self._graph.calculate_inout_score(inc_vertex))
                          for inc_vertex in incidental_vertices)
        self._scores[vertex] = summary * self._damping_factor + (1 - self._damping_factor)

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
        check_type(n_sentences, int)
        sorted_sentences = sorted(self._scores, key=lambda sent: self._scores[sent], reverse=True)
        return tuple(sorted_sentences[:n_sentences])

    def make_summary(self, n_sentences: int) -> str:
        """
        Constructs summary from the most important sentences
        :param n_sentences: number of sentences to include in the summary
        :return: summary
        """
        check_type(n_sentences, int)
        top_sentences = sorted(self.get_top_sentences(n_sentences), key=lambda x: x.get_position())
        return '\n'.join(sentence.get_text() for sentence in top_sentences)


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
