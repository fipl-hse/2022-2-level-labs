"""
Lab 4
Summarize text using TextRank algorithm
"""
from typing import Union, Any
import re

from lab_3_keywords_textrank.main import TextEncoder, TFIDFAdapter, TextPreprocessor

PreprocessedSentence = tuple[str, ...]
EncodedSentence = tuple[int, ...]


def check_type(user_input: Any, user_input_type: type, can_be_empty: bool) -> None:
    """
    Checks weather object has the correct type
    """
    if not isinstance(user_input, user_input_type):
        raise ValueError
    if not user_input and can_be_empty is False:
        raise ValueError


def check_inner_types(user_input: Any, user_input_type: type, elements_type: type, can_be_empty: bool) -> None:
    """
    Checks weather object has the correct type
    and elements within are of certain type
    """
    if not isinstance(user_input, user_input_type):
        raise ValueError
    if not user_input and can_be_empty is False:
        raise ValueError
    for element in user_input:
        if not isinstance(element, elements_type):
            raise ValueError


def check_dict(user_input: dict, key_type: type, value_type: type, can_be_empty: bool) -> None:
    """
    Checks weather object is dictionary
    hat has keys and values of certain type
    """
    if not isinstance(user_input, dict):
        raise ValueError
    if not user_input and can_be_empty is False:
        raise ValueError
    for key, value in user_input.items():
        if not isinstance(key, key_type) or not isinstance(value, value_type):
            raise ValueError


class NoRelevantTextsError(Exception):
    """
    This error raises when there aren't texts related to the user's query
    """

    def __str__(self):
        return 'Texts that are related to the query were not found. Try another query.'


class IncorrectQueryError(Exception):
    """
    This error raises when user enters the wrong query
    """

    def __str__(self):
        return 'Incorrect query. Use string as input.'


class Sentence:
    """
    An abstraction over the real-world sentences
    """

    def __init__(self, text: str, position: int) -> None:
        """
        Constructs all the necessary attributes
        """
        if not isinstance(text, str) or not isinstance(position, int) or isinstance(position, bool):
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
        check_type(text, str, False)
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
        check_inner_types(preprocessed_sentence, tuple, str, True)
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
        check_inner_types(encoded_sentence, tuple, int, True)
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
        super().__init__(stop_words, punctuation)
        check_inner_types(self._stop_words, tuple, str, True)
        check_inner_types(self._punctuation, tuple, str, True)

    def _split_by_sentence(self, text: str) -> tuple[Sentence, ...]:
        """
        Splits the provided text by sentence
        :param text: the raw text
        :return: a sequence of sentences
        """
        check_type(text, str, False)
        text = text.replace("\n", ' ').replace("  ", " ")
        phrases = re.split(r'(?<=[.?!])\s+(?=[A-ZА-Я])', text)
        sentences = []
        for ind, phrase in enumerate(phrases):
            sentences.append(Sentence(phrase, ind))
        return tuple(sentences)

    def _preprocess_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their preprocessed versions
        :param sentences: a list of sentences
        :return:
        """
        check_inner_types(sentences, tuple, Sentence, False)
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
        check_type(text, str, False)
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
        check_inner_types(tokens, tuple, str, True)
        new_tokens = (elem for elem in tokens if elem not in self._word2id)

        for ind, element in enumerate(new_tokens, start=1000 + len(self._word2id)):
            self._word2id[element] = ind
            self._id2word[ind] = element

    def encode_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their encoded versions
        :param sentences: a sequence of sentences
        :return: a list of sentences with their preprocessed versions
        """
        check_inner_types(sentences, tuple, Sentence, False)
        for sentence in sentences:
            self._learn_indices(sentence.get_preprocessed())
            sentence.set_encoded(tuple(self._word2id[sent] for sent in sentence.get_preprocessed()))


def calculate_similarity(sequence: Union[list, tuple], other_sequence: Union[list, tuple]) -> float:
    """
    Calculates similarity between two sequences using Jaccard index
    :param sequence: a sequence of items
    :param other_sequence: a sequence of items
    :return: similarity score
    """
    if not isinstance(sequence, (list, tuple)) or not isinstance(other_sequence, (list, tuple)):
        raise ValueError
    if len(sequence) == 0 or len(other_sequence) == 0:
        return 0
    whole_sequence = tuple(sequence) + tuple(other_sequence)
    all_unique_element = set(whole_sequence)
    the_same_elements = {elem for elem in whole_sequence if elem in sequence and elem in other_sequence}
    return len(the_same_elements) / len(all_unique_element)


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
        check_type(vertex, Sentence, False)
        if vertex not in self._vertices:
            raise ValueError
        idx = self._vertices.index(vertex)
        return len([element for element in self._matrix[idx] if element > 0])

    def add_edge(self, vertex1: Sentence, vertex2: Sentence) -> None:
        """
        Adds or overwrites an edge in the graph between the specified vertices
        :param vertex1:
        :param vertex2:
        :return:
        """
        check_type(vertex1, Sentence, False)
        check_type(vertex2, Sentence, False)
        if vertex1.get_encoded() == vertex2.get_encoded():
            raise ValueError

        for vertex in vertex1, vertex2:
            if vertex not in self._vertices:
                self._vertices.append(vertex)
                self._matrix.append([])

        for edges_list in self._matrix:
            if len(edges_list) < len(self._vertices):
                edges_list.extend([0 for _ in range(len(self._vertices) - len(edges_list))])

        idx1 = self._vertices.index(vertex1)
        idx2 = self._vertices.index(vertex2)
        self._matrix[idx1][idx2] = calculate_similarity(vertex1.get_encoded(), vertex2.get_encoded())
        self._matrix[idx2][idx1] = calculate_similarity(vertex1.get_encoded(), vertex2.get_encoded())

    def get_similarity_score(self, sentence: Sentence, other_sentence: Sentence) -> float:
        """
        Gets the similarity score for two sentences from the matrix
        :param sentence
        :param other_sentence
        :return: the similarity score
        """
        check_type(sentence, Sentence, False)
        check_type(other_sentence, Sentence, False)
        if sentence not in self._vertices or other_sentence not in self._vertices:
            raise ValueError
        idx1 = self._vertices.index(sentence)
        idx2 = self._vertices.index(other_sentence)
        return self._matrix[idx1][idx2]

    def fill_from_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Updates graph instance with vertices and edges extracted from sentences
        :param sentences
        :return:
        """
        check_inner_types(sentences, tuple, Sentence, False)

        pairs = []
        for idx1, token1 in enumerate(sentences):
            for token2 in sentences[idx1 + 1:idx1 + len(sentences)]:
                if token1.get_encoded() != token2.get_encoded():
                    pairs.append((token1, token2))

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
        check_type(graph, SimilarityMatrix, False)
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
        check_type(vertex, Sentence, False)
        check_inner_types(incidental_vertices, list, Sentence, True)
        check_dict(scores, Sentence, float, False)

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
        check_type(n_sentences, int, True)
        if isinstance(n_sentences, bool):
            raise ValueError
        return tuple(sorted(self._scores, key=lambda elem: self._scores[elem], reverse=True)[:n_sentences])

    def make_summary(self, n_sentences: int) -> str:
        """
        Constructs summary from the most important sentences
        :param n_sentences: number of sentences to include in the summary
        :return: summary
        """
        check_type(n_sentences, int, False)
        summery = sorted(self.get_top_sentences(n_sentences), key=lambda elem: elem.get_position())
        return '\n'.join(element.get_text() for element in summery)


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
        self._paths_to_texts = paths_to_texts
        self._stop_words = stop_words
        self._punctuation = punctuation
        self._idf_values = idf_values
        self._text_preprocessor = TextPreprocessor(self._stop_words, self._punctuation)
        self._sentence_encoder = SentenceEncoder()
        self._sentence_preprocessor = SentencePreprocessor(self._stop_words, self._punctuation)
        self._knowledge_database = {}

        for path in paths_to_texts:
            self.add_text_to_database(path)

    def add_text_to_database(self, path_to_text: str) -> None:
        """
        Adds the given text to the existing database
        :param path_to_text
        :return:
        """
        check_type(path_to_text, str, False)
        with open(path_to_text, "r", encoding='utf-8') as texts:
            text = texts.read()
            sentences = self._sentence_preprocessor.get_sentences(text)
            self._sentence_encoder.encode_sentences(sentences)

            tuple_words = self._text_preprocessor.preprocess_text(text)
            adapter = TFIDFAdapter(tuple_words, self._idf_values)
            adapter.train()
            keywords = adapter.get_top_keywords(100)

            similarity_matrix = SimilarityMatrix()
            similarity_matrix.fill_from_sentences(sentences)
            text_rank_summarizer = TextRankSummarizer(similarity_matrix)
            text_rank_summarizer.train()
            summery = text_rank_summarizer.make_summary(5)
            self._knowledge_database[path_to_text] = {'sentences': sentences, 'keywords': keywords, 'summary': summery}

    def _find_texts_close_to_keywords(self, keywords: tuple[str, ...], n_texts: int) -> tuple[str, ...]:
        """
        Finds texts that are similar (i.e. contain the same keywords) to the given keywords
        :param keywords: a sequence of keywords
        :param n_texts: number of texts to find
        :return: the texts' ids
        """
        check_inner_types(keywords, tuple, str, True)
        check_type(n_texts, int, False)
        if isinstance(n_texts, bool):
            raise ValueError
        top_similar_texts = {}
        for element in self._knowledge_database.items():
            number = calculate_similarity(element[1]['keywords'], keywords)
            top_similar_texts[element[0]] = number
        if not any(top_similar_texts.values()):
            raise NoRelevantTextsError
        sorted_top_similar_texts = dict(sorted(top_similar_texts.items(), key=lambda x: (x[1], x[0]), reverse=True))
        return tuple(sorted_top_similar_texts.keys())[:n_texts]

    def reply(self, query: str, n_summaries: int = 3) -> str:
        """
        Replies to the query
        :param query: the query
        :param n_summaries: the number of summaries to include in the answer
        :return: the answer
        """
        if not query or not isinstance(query, str):
            raise IncorrectQueryError
        users_key_words = self._text_preprocessor.preprocess_text(query)
        summaries = self._find_texts_close_to_keywords(users_key_words, n_summaries)
        return "Ответ:\n" + "\n\n".join(self._knowledge_database[summary]['summary'] for summary in summaries)
