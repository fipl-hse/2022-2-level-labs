"""
Lab 4
Summarize text using TextRank algorithm
"""
from typing import Union, Any
import re
import itertools as it
from lab_3_keywords_textrank.main import TextEncoder, \
    TextPreprocessor


PreprocessedSentence = tuple[str, ...]
EncodedSentence = tuple[int, ...]
# написать функцию для проверки типов


def check_type(seq: Any, seq_type:Any, elem_type:Any) -> bool:
    """
    Checks iterable argument type
    return: bool
    """
    if not isinstance(seq, seq_type):
        return False
    for elem in seq:
        if not isinstance(elem, elem_type):
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
        if not isinstance(text, str):
            raise ValueError
        if not isinstance(position, int) or isinstance(position, bool):
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
        if not isinstance(text, str) or not text:
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
        if not check_type(preprocessed_sentence, tuple, str):
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
        if not check_type(encoded_sentence, tuple, int):
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

        if not check_type(stop_words, tuple, str):
            raise ValueError
        if not check_type(punctuation,tuple, str):
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
        # чтобы вспомнить, что значит регулярка, см. lookahead и lookbehind
        sentences = re.split(r'(?<=[?!.])\s+(?=[A-ZА-Я])', text)
        tuple_with_sent = []
        for count, value in enumerate(sentences):
            sent = Sentence(value.replace('\n', ' ').replace('  ', ' '), count)
            if sent:
                tuple_with_sent.append(sent)
        print(tuple_with_sent)
        return tuple(tuple_with_sent)


    def _preprocess_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their preprocessed versions
        :param sentences: a list of sentences
        :return:
        """
        if not check_type(sentences, tuple, Sentence):
            raise ValueError
        for sentence in sentences:
            text = sentence.get_text()
            new_sentence = super().preprocess_text(text)
            sentence.set_preprocessed(new_sentence)

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
        if not check_type(tokens, tuple, str):
            raise ValueError
        if self._id2word:
            start_v = max(self._id2word.keys())
        else:
            start_v = 1000
        for count, token in enumerate(tokens, start=start_v):
            if token not in (self._word2id.keys() and self._id2word.keys()):
                self._word2id[token] = count
                self._id2word[count] = token

    def encode_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their encoded versions
        :param sentences: a sequence of sentences
        :return: a list of sentences with their preprocessed versions
        """
        if not check_type(sentences, tuple, Sentence):
            raise ValueError
        words = []
        for sentence in sentences:
            prepr = sentence.get_preprocessed()
            for word in prepr:
                words.append(word)
        self._learn_indices(tuple(words))
        for sentence in sentences:
            prepr = sentence.get_preprocessed()
            enc_sent = []
            for word in prepr:
                enc_sent.append(self._word2id[word])
            sentence.set_encoded(tuple(enc_sent))


def calculate_similarity(sequence: Union[list, tuple], other_sequence: Union[list, tuple]) -> float:
    """
    Calculates similarity between two sequences using Jaccard index
    :param sequence: a sequence of items
    :param other_sequence: a sequence of items
    :return: similarity score
    """
    if not isinstance(sequence, (list, tuple)):
        raise ValueError
    if not isinstance(other_sequence, (list, tuple)):
        raise ValueError
    if not sequence or not other_sequence:
        return 0
    numerator = []
    denominator = set(tuple(sequence) + tuple(other_sequence))
    for i in sequence:
        if i in other_sequence:
            numerator.append(i)
    set(numerator)
    j_val = len(numerator) / len(denominator)
    return j_val

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
        if not isinstance(vertex, Sentence):
            raise ValueError
        if vertex not in self._vertices:
            raise ValueError
        vert_ind = self._vertices.index(vertex)
        inout_score = len(self._matrix) - self._matrix[vert_ind].count(0)
        return inout_score


    def add_edge(self, vertex1: Sentence, vertex2: Sentence) -> None:
        """
        Adds or overwrites an edge in the graph between the specified vertices
        :param vertex1:
        :param vertex2:
        :return:
        """
        if not (isinstance(vertex1, Sentence) or isinstance(vertex2, Sentence)):
            raise ValueError
        if vertex1.get_encoded() == vertex2.get_encoded():
            raise ValueError
        for vertex in vertex1, vertex2:
            if vertex not in self._vertices:
                self._vertices.append(vertex)
                self._matrix.append([0])
        for i in range(len(self._matrix)):
            for _ in range((len(self._vertices)) - len((self._matrix[i]))):
                self._matrix[i].append(0)
        vert1 = vertex1.get_encoded()
        vert2 = vertex2.get_encoded()
        v1_index = self._vertices.index(vertex1)
        v2_index = self._vertices.index(vertex2)

        self._matrix[v1_index][v2_index] = calculate_similarity(vert1, vert2)
        self._matrix[v2_index][v1_index] = calculate_similarity(vert1, vert2)


    def get_similarity_score(self, sentence: Sentence, other_sentence: Sentence) -> float:
        """
        Gets the similarity score for two sentences from the matrix
        :param sentence
        :param other_sentence
        :return: the similarity score
        """
        if not isinstance(sentence, Sentence) or not isinstance(other_sentence, Sentence):
            raise ValueError
        if (sentence or other_sentence) not in self._vertices:
            raise ValueError
        sim_score = self._matrix[self._vertices.index(sentence)][self._vertices.index(other_sentence)]
        return sim_score


    def fill_from_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Updates graph instance with vertices and edges extracted from sentences
        :param sentences
        :return:
        """
        if not sentences:
            raise ValueError
        if not check_type(sentences, tuple, Sentence):
            raise ValueError
        pairs = list(it.combinations(sentences, 2))
        for pair in pairs:
            if pair[0].get_encoded() != pair[1].get_encoded():
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
        if not (isinstance(vertex, Sentence) or check_type(incidental_vertices, list, Sentence)
                or isinstance(scores, dict)):
            raise ValueError
        for key, value in scores.items():
            if not (isinstance(key, Sentence) or isinstance(value, float)):
                raise ValueError
        summa = sum((1 / self._graph.calculate_inout_score(inc_vertex)) * scores[inc_vertex]
                    for inc_vertex in incidental_vertices)
        self._scores[vertex] = (summa * self._damping_factor +
                                (1 - self._damping_factor))

    def train(self) -> None:
        """
        Iteratively computes significance scores for vertices
        """
        # кортеж с предложениями
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
        top_n = sorted(self._scores.items(), key=lambda item: item[1],
                                      reverse=True)[:n_sentences]

        return tuple(k for k, v in top_n)

    def make_summary(self, n_sentences: int) -> str:
        """
        Constructs summary from the most important sentences
        :param n_sentences: number of sentences to include in the summary
        :return: summary
        """
        # выводить предложения в хронологическом порядке!!! Нужно исправить
        if not isinstance(n_sentences, int) or isinstance(n_sentences, bool):
            raise ValueError
        sent_dict = {sentence: sentence.get_position() for sentence in self.get_top_sentences(n_sentences)}
        right_order = (k for k, v in sorted(sent_dict.items(), key=lambda item: item[1])[:n_sentences])
        return '\n'.join(sentence.get_text().strip() for sentence in right_order)
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
