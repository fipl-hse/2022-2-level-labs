"""
Lab 4
Summarize text using TextRank algorithm
"""
from typing import Union

from lab_3_keywords_textrank.main import TextEncoder, \
    TextPreprocessor
import re

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
        if not isinstance(text, str):
            raise ValueError
        if not isinstance(position, int) or isinstance(position, bool):
            raise ValueError
        self._text = text
        self._position = position
        self._preprocessed = ()
        self._encoded = ()

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
        if not preprocessed_sentence:
            return None
        if not isinstance(preprocessed_sentence, tuple):
            raise ValueError
        for element in preprocessed_sentence:
            if not isinstance(element, str):
                raise ValueError
        self._preprocessed = preprocessed_sentence
        return None

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
        if not encoded_sentence:
            return None
        if not isinstance(encoded_sentence, tuple):
            raise ValueError
        for element in encoded_sentence:
            if not isinstance(element, int):
                raise ValueError
        self._encoded = encoded_sentence
        return None

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
        if not isinstance(stop_words, tuple) or not isinstance(punctuation, tuple):
            raise ValueError
        if stop_words:
           for element in stop_words:
                if not isinstance(element, str):
                    raise ValueError
        for element in punctuation:
            if not isinstance(element, str):
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
        sent_list = []
        pattern = re.compile('((?<=[.?!]\s))')
        split_text = re.split(pattern, text)
        idx = 0
        for txt_element in split_text:
            if txt_element:
                sent_list.append(Sentence(txt_element.strip(), idx))
                idx += 1
        return tuple(sent_list) # this function doesn't work correctly but i don't know why

    def _preprocess_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their preprocessed versions
        :param sentences: a list of sentences
        :return:
        """
        if not isinstance(sentences, tuple):
            raise ValueError
        for one_sentence in sentences:
            if not isinstance(one_sentence, Sentence):
                raise ValueError
            txt = one_sentence.get_text()
            preprocessing = TextPreprocessor.preprocess_text(self, txt)
            one_sentence.set_preprocessed(preprocessing)

    def get_sentences(self, text: str) -> tuple[Sentence, ...]:
        """
        Extracts the sentences from the given text & preprocesses them
        :param text: the raw text
        :return:
        """
        if not isinstance(text, str):
            raise ValueError

        sentences = self._split_by_sentence(text)
        self._preprocess_sentences(sentences)
        return tuple(sentences)

class SentenceEncoder(TextEncoder):
    """
    A class to encode string sequence into matching integer sequence
    """

    def __init__(self) -> None:
        """
        Constructs all the necessary attributes
        """
        super().__init__()
        self._last_idx = 1000

    def _learn_indices(self, tokens: tuple[str, ...]) -> None:
        """
        Fills attributes mapping words and integer equivalents to each other
        :param tokens: a sequence of string tokens
        :return:
        """
        if not isinstance(tokens, tuple):
            raise ValueError

        for one_token in tokens:
            if not isinstance(one_token, str):
                raise ValueError
            self._word2id[one_token] = self._word2id.get(one_token, self._last_idx)
            self._id2word[self._last_idx] = self._id2word.get(self._last_idx, one_token)
            self._last_idx += 1

    def encode_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their encoded versions
        :param sentences: a sequence of sentences
        :return: a list of sentences with their preprocessed versions
        """
        if not isinstance(sentences, tuple):
            raise ValueError
        coded_sent = []
        for one_sentence in sentences:
            if not isinstance(one_sentence, Sentence):
                raise ValueError
            preprocessed_tokens = one_sentence.get_preprocessed()
            self._learn_indices(preprocessed_tokens)
            for words in preprocessed_tokens:
                coded_sent.append(self._word2id[words])
            one_sentence.set_encoded(tuple(coded_sent))
            coded_sent = []


def calculate_similarity(sequence: Union[list, tuple], other_sequence: Union[list, tuple]) -> float:
    """
    Calculates similarity between two sequences using Jaccard index
    :param sequence: a sequence of items
    :param other_sequence: a sequence of items
    :return: similarity score
    """
    for variable in (sequence, other_sequence):
        if not isinstance(variable, (list, tuple)):
            raise ValueError

    if not sequence or not other_sequence:
        return 0

    sequences = sequence + other_sequence
    intersection = len(set(sequences))
    association = []

    if len(set(sequence)) < len(set(other_sequence)):
        small_seq = set(sequence)
        big_seq = set(other_sequence)
    else:
        small_seq = set(other_sequence)
        big_seq = set(sequence)

    for one_elem in small_seq:
        if one_elem in big_seq:
            association.append(one_elem)

    return len(association)/intersection


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
        self._codes = []

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
        idx = self._vertices.index(vertex)
        return len([item for item in self._matrix[idx] if item > 0 and item]) - 1

    def add_edge(self, vertex1: Sentence, vertex2: Sentence) -> None:
        """
        Adds or overwrites an edge in the graph between the specified vertices
        :param vertex1:
        :param vertex2:
        :return:
        """
        for variable in (vertex1, vertex2):
            if not isinstance(variable, Sentence):
                raise ValueError

        if vertex1 == vertex2:
            raise ValueError

        seq1 = vertex1.get_encoded()
        seq2 = vertex2.get_encoded()

        for vertex in (vertex1, vertex2):
            vert_code = sorted(vertex.get_encoded())
            if vert_code not in self._codes:
                self._vertices.append(vertex)
                self._codes.append(vert_code)
                self._matrix.append([])

        for edges_list in self._matrix:
            if len(edges_list) < len(self._vertices):
                edges_list.extend([0 for _ in range(len(self._vertices) - len(edges_list))])

        idx1 = self._codes.index(sorted(seq1))
        idx2 = self._codes.index(sorted(seq2))

        self._matrix[idx1][idx1] = 1
        self._matrix[idx2][idx2] = 1

        self._matrix[idx1][idx2] = calculate_similarity(seq1, seq2)
        self._matrix[idx2][idx1] = calculate_similarity(seq1, seq2)

    def get_similarity_score(self, sentence: Sentence, other_sentence: Sentence) -> float:
        """
        Gets the similarity score for two sentences from the matrix
        :param sentence
        :param other_sentence
        :return: the similarity score
        """
        if not isinstance(sentence, Sentence) or not isinstance(other_sentence, Sentence):
            raise ValueError

        seq1 = sorted(sentence.get_encoded())
        seq2 = sorted(other_sentence.get_encoded())

        if seq1 not in self._codes or seq2 not in self._codes:
            raise ValueError

        idx1 = self._codes.index(seq1)
        idx2 = self._codes.index(seq2)
        return self._matrix[idx1][idx2]

    def fill_from_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Updates graph instance with vertices and edges extracted from sentences
        :param sentences
        :return:
        """
        if not isinstance(sentences, tuple) or not sentences:
            raise ValueError

        for idx1 in range(len(sentences)-1):
            for idx2 in range(1, len(sentences)):
                sent1 = sentences[idx1]
                sent2 = sentences[idx2]
                if not isinstance(sent1, Sentence) or not isinstance(sent2, Sentence):
                    raise ValueError
                if sorted(sent1.get_encoded()) == sorted(sent2.get_encoded()):
                    continue
                self.add_edge(sent1, sent2)


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
        if not isinstance(graph, SimilarityMatrix):
            raise ValueError
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
        if not isinstance(vertex, Sentence) or not isinstance(incidental_vertices, list) \
                or not isinstance(scores, dict):
            raise ValueError

        for inc_vert in incidental_vertices:
            if not isinstance(inc_vert, Sentence):
                raise ValueError

        summa = sum((1 / (1 + self._graph.calculate_inout_score(inc_vertex))) * scores[inc_vertex]
                    for inc_vertex in incidental_vertices)
        self._scores[vertex] = summa * self._damping_factor + 1 - self._damping_factor

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
        if not isinstance(n_sentences, int) or isinstance(n_sentences, bool):
            raise ValueError
        sort_list = sorted(self._scores, key=lambda x: self._scores[x], reverse=True)[:n_sentences]
        return tuple(sort_list)

    def make_summary(self, n_sentences: int) -> str:
        """
        Constructs summary from the most important sentences
        :param n_sentences: number of sentences to include in the summary
        :return: summary
        """
        if not isinstance(n_sentences, int) or isinstance(n_sentences, bool):
            raise ValueError

        important_list = sorted(self.get_top_sentences(n_sentences), key=lambda x: x.get_position())
        result = []
        for element in important_list:
            result.append(element.get_text())
        return '\n'.join(result)


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
