"""
Lab 4
Summarize text using TextRank algorithm
"""
from typing import Union, Type, Any
import re

from lab_3_keywords_textrank.main import TextEncoder, \
    TextPreprocessor, TFIDFAdapter

PreprocessedSentence = tuple[str, ...]
EncodedSentence = tuple[int, ...]


def type_check(check_object: Any, object_type: Union[Type, tuple[Type, ...]], token_type: Any = None,
               value_type: Any = None, can_be_empty: bool = True) -> None:
    """
    Checks types of the object and its contents. Also checks if it is empty.
    :param check_object: the object we need to check
    :param object_type: expected type of the object
    :param token_type: expected type of its tokens (or keys if the object is a dictionary)
    :param value_type: expected type of its values (for dictionaries)
    :param can_be_empty: True if the object can be empty, False otherwise
    :return: None
    """
    if not can_be_empty and not check_object:
        raise ValueError
    if not isinstance(check_object, object_type):
        raise ValueError
    if object_type == 'int' and isinstance(check_object, bool):
        raise ValueError
    if isinstance(check_object, dict) and token_type and value_type:
        for token, value in check_object.items():
            if not isinstance(token, token_type) or not isinstance(value, value_type):
                raise ValueError
    else:
        if token_type:
            for token in check_object:
                if not isinstance(token, token_type):
                    raise ValueError


class NoRelevantTextsError(Exception):
    """
    An error which is raised if there are no relevant texts.
    """
    pass


class IncorrectQueryError(Exception):
    """
    An error which is raised if the query is incorrect.
    """
    pass


class Sentence:
    """
    An abstraction over the real-world sentences
    """

    _preprocessed: tuple[str, ...]
    _encoded: tuple[int, ...]

    def __init__(self, text: str, position: int) -> None:
        """
        Constructs all the necessary attributes
        """
        type_check(text, str)
        type_check(position, int)
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
        type_check(text, str)
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
        type_check(preprocessed_sentence, tuple, str)
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
        type_check(encoded_sentence, tuple, int)
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
        type_check(stop_words, tuple, str)
        type_check(punctuation, tuple, str)
        super().__init__(stop_words, punctuation)

    def _split_by_sentence(self, text: str) -> tuple[Sentence, ...]:
        """
        Splits the provided text by sentence
        :param text: the raw text
        :return: a sequence of sentences
        """
        type_check(text, str)
        text = text.replace('\n', ' ').replace('  ', ' ')
        split_text = re.split(r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ])', text)
        return tuple(Sentence(sentence.strip(), position) for position, sentence in enumerate(split_text))

    def _preprocess_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their preprocessed versions
        :param sentences: a list of sentences
        :return:
        """
        type_check(sentences, tuple, Sentence)
        for sentence in sentences:
            preprocessed_sentence = self.preprocess_text(sentence.get_text())
            sentence.set_preprocessed(preprocessed_sentence)

    def get_sentences(self, text: str) -> tuple[Sentence, ...]:
        """
        Extracts the sentences from the given text & preprocesses them
        :param text: the raw text
        :return:
        """
        type_check(text, str)
        split_text = self._split_by_sentence(text)
        self._preprocess_sentences(split_text)
        return split_text


class SentenceEncoder(TextEncoder):
    """
    A class to encode string sequence into matching integer sequence
    """

    def __init__(self) -> None:
        """
        Constructs all the necessary attributes
        """
        super().__init__()
        self._last_idx = 999

    def _learn_indices(self, tokens: tuple[str, ...]) -> None:
        """
        Fills attributes mapping words and integer equivalents to each other
        :param tokens: a sequence of string tokens
        :return:
        """
        type_check(tokens, tuple, str)
        new_tokens = (token for token in tokens if token not in self._word2id)
        new_index = 1 + self._last_idx
        for ind, token in enumerate(new_tokens):
            self._word2id[token] = ind + new_index
            self._id2word[ind + new_index] = token
        self._last_idx = max(self._id2word)

    def encode_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their encoded versions
        :param sentences: a sequence of sentences
        :return: a list of sentences with their preprocessed versions
        """
        type_check(sentences, tuple, Sentence)
        for sentence in sentences:
            preprocessed = sentence.get_preprocessed()
            self._learn_indices(preprocessed)
            encoded = tuple(self._word2id[word] for word in preprocessed)
            sentence.set_encoded(encoded)


def calculate_similarity(sequence: Union[list, tuple], other_sequence: Union[list, tuple]) -> float:
    """
    Calculates similarity between two sequences using Jaccard index
    :param sequence: a sequence of items
    :param other_sequence: a sequence of items
    :return: similarity score
    """
    type_check(sequence, (list, tuple))
    type_check(other_sequence, (list, tuple))
    if not sequence or not other_sequence:
        return 0.0
    ab_common = set(sequence) & set(other_sequence)
    ab_total = set(sequence) | set(other_sequence)
    return len(ab_common) / len(ab_total)


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
        type_check(vertex, Sentence)
        if vertex not in self._vertices:
            raise ValueError
        idx = self._vertices.index(vertex)
        return sum(i > 0 for i in self._matrix[idx])

    def add_edge(self, vertex1: Sentence, vertex2: Sentence) -> None:
        """
        Adds or overwrites an edge in the graph between the specified vertices
        :param vertex1:
        :param vertex2:
        :return:
        """
        type_check(vertex1, Sentence)
        type_check(vertex2, Sentence)
        if vertex1 == vertex2:
            raise ValueError
        for vertex in vertex1, vertex2:
            if vertex not in self._vertices:
                self._vertices.append(vertex)
                self._matrix.append([])
        for edges_list in self._matrix:
            if len(edges_list) < len(self._vertices):
                edges_list.extend([0 for _ in range(len(self._vertices) - len(edges_list))])
        encoded1 = vertex1.get_encoded()
        encoded2 = vertex2.get_encoded()
        similarity = calculate_similarity(encoded1, encoded2)
        idx1 = self._vertices.index(vertex1)
        idx2 = self._vertices.index(vertex2)
        self._matrix[idx1][idx2] = similarity
        self._matrix[idx2][idx1] = similarity

    def get_similarity_score(self, sentence: Sentence, other_sentence: Sentence) -> float:
        """
        Gets the similarity score for two sentences from the matrix
        :param sentence
        :param other_sentence
        :return: the similarity score
        """
        type_check(sentence, Sentence)
        type_check(other_sentence, Sentence)
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
        type_check(sentences, tuple, Sentence, can_be_empty=False)
        for sentence1 in sentences:
            for sentence2 in sentences:
                if sentence1.get_encoded() != sentence2.get_encoded():
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
        type_check(graph, SimilarityMatrix)
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
        type_check(vertex, Sentence)
        type_check(incidental_vertices, list, Sentence)
        type_check(scores, dict, Sentence, float)
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
        type_check(n_sentences, int)
        return tuple(sorted(self._scores, key=lambda sent: self._scores[sent], reverse=True)[:n_sentences])

    def make_summary(self, n_sentences: int) -> str:
        """
        Constructs summary from the most important sentences
        :param n_sentences: number of sentences to include in the summary
        :return: summary
        """
        type_check(n_sentences, int)
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
        type_check(paths_to_texts, list, str)
        type_check(stop_words, tuple, str)
        type_check(punctuation, tuple, str)
        type_check(idf_values, dict, str, float)
        self._paths_to_texts = paths_to_texts
        self._stop_words = stop_words
        self._punctuation = punctuation
        self._idf_values = idf_values
        self._text_preprocessor = TextPreprocessor(self._stop_words, self._punctuation)
        self._sentence_encoder = SentenceEncoder()
        self._sentence_preprocessor = SentencePreprocessor(self._stop_words, self._punctuation)
        self._knowledge_database = {}
        for text in paths_to_texts:
            self.add_text_to_database(text)

    def add_text_to_database(self, path_to_text: str) -> None:
        """
        Adds the given text to the existing database
        :param path_to_text
        :return:
        """
        type_check(path_to_text, str)
        with open(path_to_text, 'r', encoding='utf-8') as f:
            text = f.read()

        tokens = self._text_preprocessor.preprocess_text(text)
        sentences = self._sentence_preprocessor.get_sentences(text)
        self._sentence_encoder.encode_sentences(sentences)

        tfidf = TFIDFAdapter(tokens, self._idf_values)
        tfidf.train()
        keywords = tfidf.get_top_keywords(100)

        graph = SimilarityMatrix()
        graph.fill_from_sentences(sentences)
        summarizer = TextRankSummarizer(graph)
        summarizer.train()
        summary = summarizer.make_summary(5)

        self._knowledge_database[path_to_text] = {
            'sentences': sentences,
            'keywords': keywords,
            'summary': summary
        }

    def _find_texts_close_to_keywords(self, keywords: tuple[str, ...], n_texts: int) -> tuple[str, ...]:
        """
        Finds texts that are similar (i.e. contain the same keywords) to the given keywords
        :param keywords: a sequence of keywords
        :param n_texts: number of texts to find
        :return: the texts' ids
        """
        type_check(keywords, tuple, str)
        type_check(n_texts, int)
        similarity = {text: calculate_similarity(data['keywords'], keywords)
                      for text, data in self._knowledge_database.items()}
        if all(not value for value in similarity.values()):
            raise NoRelevantTextsError('Texts that are related to the query were not found. Try another query.')
        similarity_sorted = sorted(similarity, reverse=True)
        return tuple(sorted(similarity_sorted, key=lambda x: similarity[x], reverse=True)[:n_texts])

    def reply(self, query: str, n_summaries: int = 3) -> str:
        """
        Replies to the query
        :param query: the query
        :param n_summaries: the number of summaries to include in the answer
        :return: the answer
        """
        if not query or not isinstance(query, str):
            raise IncorrectQueryError('Incorrect query. Use string as input.')
        type_check(n_summaries, int)
        if len(self._knowledge_database) < n_summaries:
            raise ValueError
        keywords = self._text_preprocessor.preprocess_text(query)
        texts = self._find_texts_close_to_keywords(keywords, n_summaries)
        reply = '\n\n'.join(self._knowledge_database[text]['summary'] for text in texts)
        return f'Ответ:\n{reply}'
