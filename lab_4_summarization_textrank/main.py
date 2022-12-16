"""
Lab 4
Summarize text using TextRank algorithm
"""
from typing import Union, Iterable, Type, Any, Optional
import re
from itertools import permutations
from lab_3_keywords_textrank.main import TextEncoder, \
    TextPreprocessor


PreprocessedSentence = tuple[str, ...]
EncodedSentence = tuple[int, ...]
def check_types(variable: Any, possible_var_type: Type, container_value_type: Optional[Type] = None) -> None:
    """
    Checks if the variable is of an appropriate type
    param: variable
    param: possible_var_type
    param: container_value_type (default = None)
    return:
    """
    if not isinstance(variable, possible_var_type) or isinstance(variable, bool):
        raise ValueError
    if not container_value_type:
        return None
    for element in variable:
        if not isinstance(element, container_value_type):
            raise ValueError
    return None
def check_iter(iterable: Iterable, val_type: Type) -> bool:
    """
    Checks, if all values in iterable are of needed type
    """
    if not all(isinstance(val, val_type) for val in iterable):
        return False
    return True

class Sentence:
    """
    An abstraction over the real-world sentences
    """

    def __init__(self, text: str, position: int) -> None:
        """
        Constructs all the necessary attributes
        """
        if not (isinstance(text, str) and isinstance(position, int) and
            not isinstance(position, bool)):
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
        if not (text and isinstance(text, str)):
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
        if not (isinstance(preprocessed_sentence, tuple) and
            check_iter(preprocessed_sentence, str)):
            raise ValueError
        if check_iter(self._preprocessed, str):
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
        if not (isinstance(encoded_sentence, tuple) and
                check_iter(encoded_sentence, int)):
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
        if not (isinstance(stop_words, tuple) and
                isinstance(punctuation, tuple) and
            check_iter(stop_words, str) and check_iter(punctuation, str)):
            raise ValueError
        super().__init__(stop_words, punctuation)

    def _split_by_sentence(self, text: str) -> tuple[Sentence, ...]:
        """
        Splits the provided text by sentence
        :param text: the raw text
        :return: a sequence of sentences
        """
        if not (text and isinstance(text, str)):
            raise ValueError
        text_list = re.split(r'(?<=[.!?])\s+(?=[А-ЯA-Z])', text)
        return tuple(Sentence(sent.replace('\n', ' ').replace('  ', ' '), pos)
                     for pos, sent in enumerate(text_list) if sent)

    def _preprocess_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their preprocessed versions
        :param sentences: a list of sentences
        :return:
        """
        if not (sentences and isinstance(sentences, tuple) and
        check_iter(sentences, Sentence)):
            raise ValueError
        for one_sentence in sentences:
            txt = one_sentence.get_text()
            preprocessing = TextPreprocessor.preprocess_text(self, txt)
            one_sentence.set_preprocessed(preprocessing)

    def get_sentences(self, text: str) -> tuple[Sentence, ...]:
        """
        Extracts the sentences from the given text & preprocesses them
        :param text: the raw text
        :return:
        """
        if not (text and isinstance(text, str)):
            raise ValueError
        tuple_of_sentences = self._split_by_sentence(text)
        self._preprocess_sentences(tuple_of_sentences)
        return tuple_of_sentences



class SentenceEncoder(TextEncoder):
    """
    A class to encode string sequence into matching integer sequence
    """

    def __init__(self) -> None:
        """
        Constructs all the necessary attributes
        """
        super().__init__()
        self._last_num = 0

    def _learn_indices(self, tokens: tuple[str, ...]) -> None:
        """
        Fills attributes mapping words and integer equivalents to each other
        :param tokens: a sequence of string tokens
        :return:
        """
        if not (tokens and isinstance(tokens, tuple) and
        check_iter(tokens, str)):
            raise ValueError
        for idx, token in enumerate(tokens):
            if not self._last_num and token not in self._word2id:
                self._word2id[token] = 1000 + idx
            elif token not in self._word2id:
                self._word2id[token] = self._last_num + idx + 1
        self._last_num = list(self._word2id.values())[-1]
        self._id2word = {num: token for token, num in self._word2id.items()}

    def encode_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their encoded versions
        :param sentences: a sequence of sentences
        :return: a list of sentences with their preprocessed versions
        """
        if not (sentences and isinstance(sentences, tuple) and
        check_iter(sentences, Sentence)):
            raise ValueError
        for sent in sentences:
            preproc_sentence = sent.get_preprocessed()
            self._learn_indices(preproc_sentence)
            tokens = tuple(self._word2id[token] for token in preproc_sentence)
            sent.set_encoded(tokens)


def calculate_similarity(sequence: Union[list, tuple], other_sequence: Union[list, tuple]) -> float:
    """
    Calculates similarity between two sequences using Jaccard index
    :param sequence: a sequence of items
    :param other_sequence: a sequence of items
    :return: similarity score
    """
    if not (isinstance(sequence, (list, tuple)) and
    isinstance(other_sequence, (tuple, list))):
        raise ValueError
    if not (sequence and other_sequence):
        return 0.0
    seq_set = frozenset(sequence)
    other_seq_set = frozenset(other_sequence)
    intersection = seq_set.intersection(other_seq_set)
    union = seq_set.union(other_seq_set)
    return len(intersection) / len(union)



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
        if not (isinstance(vertex, Sentence) and vertex in self._vertices):
            raise ValueError
        v_idx = self._vertices.index(vertex)
        counter = 0
        for elem1 in self._matrix:
            try:
                if 0 < elem1[v_idx] < 1:
                    counter += 1
            except IndexError:
                continue
        for elem2 in self._matrix[v_idx]:
            if 0 < elem2 < 1:
                counter += 1
        return counter

    def add_edge(self, vertex1: Sentence, vertex2: Sentence) -> None:
        """
        Adds or overwrites an edge in the graph between the specified vertices
        :param vertex1:
        :param vertex2:
        :return:
        """
        if not ((isinstance(vertex1, Sentence)) and isinstance(vertex2, Sentence)):
            raise ValueError
        enc1 = vertex1.get_encoded()
        enc2 = vertex2.get_encoded()
        if enc1 == enc2 and vertex1 == vertex2:
            raise ValueError
        for vertex in (vertex1, vertex2):
            if vertex in self._vertices:
                continue
            self._vertices.append(vertex)
            self._matrix.append([])
        for idx, line in enumerate(self._matrix):
            while len(line) != idx + 1:
                self._matrix[idx].append(0)
            self._matrix[idx][-1] = 1
        v1_idx = self._vertices.index(vertex1)
        v2_idx = self._vertices.index(vertex2)
        similarity = calculate_similarity(vertex1.get_encoded(), vertex2.get_encoded())
        try:
            self._matrix[v1_idx][v2_idx] = similarity
        except IndexError:
            self._matrix[v2_idx][v1_idx] = similarity

    def get_similarity_score(self, sentence: Sentence, other_sentence: Sentence) -> float:
        """
        Gets the similarity score for two sentences from the matrix
        :param sentence
        :param other_sentence
        :return: the similarity score
        """
        if not (isinstance(sentence, Sentence) and isinstance(other_sentence, Sentence)):
            raise ValueError
        if not (sentence in self._vertices and other_sentence in self._vertices):
            raise ValueError
        return calculate_similarity(sentence.get_encoded(), other_sentence.get_encoded())

    def fill_from_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Updates graph instance with vertices and edges extracted from sentences
        :param sentences
        :return:
        """
        if not (sentences and isinstance(sentences, tuple) and check_iter(sentences, Sentence)):
            raise ValueError
        pairs = list(permutations(sentences, r=2))
        for pair in pairs:
            self.add_edge(pair[0], pair[1])

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
        self._scores = {}
        self._damping_factor = 0.85
        self._convergence_threshold = 0.0001
        self._max_iter = 50

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
        if not (vertex and isinstance(vertex, Sentence) and incidental_vertices and
        isinstance(incidental_vertices, list) and check_iter(incidental_vertices, Sentence) and
        scores and isinstance(scores, dict) and scores and isinstance(scores, dict) and
        check_iter(scores, Sentence) and all(isinstance(val, float) for val in scores.values())):
            raise ValueError
        summa = sum((1 / (1 + self._graph.calculate_inout_score(inc_vertex))) * scores[inc_vertex]
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
        if not (n_sentences and isinstance(n_sentences, int) and not isinstance(n_sentences, bool)):
            raise ValueError
        return tuple(sorted(self._scores, key=lambda x: self._scores[x], reverse=True))[:n_sentences]

    def make_summary(self, n_sentences: int) -> str:
        """
        Constructs summary from the most important sentences
        :param n_sentences: number of sentences to include in the summary
        :return: summary
        """

        if not (n_sentences and isinstance(n_sentences, int) and not isinstance(n_sentences, bool)):
            raise ValueError
        sent_positions = {sent.get_text(): sent.get_position() for sent in self.get_top_sentences(n_sentences)}
        sent_sorted = sorted(sent_positions, key=lambda x: sent_positions[x])
        summary = '\n'.join(sent_sorted)
        summary.replace('\n \n', '')
        return summary

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
