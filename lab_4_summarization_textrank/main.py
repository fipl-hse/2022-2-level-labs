"""
Lab 4
Summarize text using TextRank algorithm
"""
from typing import Any, Union

from lab_3_keywords_textrank.main import (TextEncoder,
                                          TextPreprocessor,
                                          TFIDFAdapter)


PreprocessedSentence = tuple[str, ...]
EncodedSentence = tuple[int, ...]


def check_type(var: Any, var_type: (type, tuple), el_type: (type, tuple) = None, can_be_empty : bool = False) -> None:
    """
    Checks if the given variable is of a certain type,
    raises ValueError in case it's not
    """
    if not isinstance(var, var_type):
        raise ValueError
    if el_type and not all(isinstance(el, el_type) for el in var):
        raise ValueError
    if var_type == int and isinstance(var, bool):
        raise ValueError
    if isinstance(var, var_type) and not can_be_empty and not var and var != 0:
        raise ValueError


class NoRelevantTextsError(Exception):
    pass


class IncorrectQueryError(Exception):
    pass


class Sentence:
    """
    An abstraction over the real-world sentences
    """

    def __init__(self, text: str, position: int) -> None:
        """
        Constructs all the necessary attributes
        """
        check_type(text, str)
        check_type(position, int)
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
        check_type(text, str)
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
        check_type(preprocessed_sentence, tuple, str)
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
        check_type(encoded_sentence, tuple, int)
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
        check_type(stop_words, tuple, str, can_be_empty=True)
        check_type(punctuation, tuple, str)
        super().__init__(stop_words, punctuation)

    def _split_by_sentence(self, text: str) -> tuple[Sentence, ...]:
        """
        Splits the provided text by sentence
        :param text: the raw text
        :return: a sequence of sentences
        """
        check_type(text, str)
        clean_text = (' '.join(text.split())).replace('!', '.').replace('?', '.')
        sentences = []
        sentence_start_idx = 0
        sentence_counter = 0

        for idx in range(len(clean_text) - 2):

            if clean_text[idx] == '.' and clean_text[idx + 1].isspace() and clean_text[idx + 2].isupper():
                sentence = clean_text[sentence_start_idx:idx + 1]
                sentences.append(Sentence(sentence, sentence_counter))

                sentence_start_idx = idx + 2
                sentence_counter += 1

            elif idx + 3 == len(clean_text):
                sentence = clean_text[sentence_start_idx:]
                sentences.append(Sentence(sentence, sentence_counter))

        return tuple(sentences)

    def _preprocess_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their preprocessed versions
        :param sentences: a list of sentences
        :return:
        """
        check_type(sentences, tuple, Sentence)
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

    def __init__(self) -> None:
        """
        Constructs all the necessary attributes
        """
        super().__init__()
        self._last_ident = 0

    def _learn_indices(self, tokens: tuple[str, ...]) -> None:
        """
        Fills attributes mapping words and integer equivalents to each other
        :param tokens: a sequence of string tokens
        :return:
        """
        check_type(tokens, tuple, str)
        for token, idx in zip(tokens, range(1000 + len(tokens))):
            self._word2id[token] = idx

        for token, idx in self._word2id.items():
            self._id2word[idx] = token

    def encode_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their encoded versions
        :param sentences: a sequence of sentences
        :return: a list of sentences with their preprocessed versions
        """
        check_type(sentences, tuple, Sentence)

        for sentence in sentences:
            tokens = sentence.get_preprocessed()
            self._learn_indices(sentence.get_preprocessed())
            encoded_sentence = []

            for token in tokens:
                encoded_sentence.append(self._word2id.get(token))

            sentence.set_encoded(tuple(encoded_sentence))


def calculate_similarity(sequence: Union[list, tuple], other_sequence: Union[list, tuple]) -> float:
    """
    Calculates similarity between two sequences using Jaccard index
    :param sequence: a sequence of items
    :param other_sequence: a sequence of items
    :return: similarity score
    """
    check_type(sequence, (list, tuple))
    check_type(other_sequence, (list, tuple))

    if not(sequence or other_sequence):
        return 0.0
    return (len(set(sequence) & set(other_sequence))) / (len(set(sequence) | set(other_sequence)))


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

        idx = self._vertices.index(vertex)
        return sum(row[idx] > 0 for row in self._matrix)

    def add_edge(self, vertex1: Sentence, vertex2: Sentence) -> None:
        """
        Adds or overwrites an edge in the graph between the specified vertices
        :param vertex1:
        :param vertex2:
        :return:
        """
        if vertex1.get_encoded() == vertex2.get_encoded():
            raise ValueError

        for vertex in vertex1, vertex2:
            check_type(vertex, Sentence)
            if vertex not in self._vertices:
                self._vertices.append(vertex)
                self._matrix.append([])

        for row in self._matrix:
            if len(row) < len(self._vertices):
                row.extend([0 for _ in range(len(self._vertices) - len(row))])

        idx1, idx2 = self._vertices.index(vertex1), self._vertices.index(vertex2)
        self._matrix[idx1][idx2] = self._matrix[idx2][idx1] = \
            calculate_similarity(vertex1.get_encoded(), vertex2.get_encoded())

    def get_similarity_score(self, sentence: Sentence, other_sentence: Sentence) -> float:
        """
        Gets the similarity score for two sentences from the matrix
        :param sentence
        :param other_sentence
        :return: the similarity score
        """
        check_type(sentence, Sentence)
        check_type(other_sentence, Sentence)
        if not(sentence in self._vertices and other_sentence in self._vertices):
            raise ValueError

        idx1, idx2 = self._vertices.index(sentence), self._vertices.index(other_sentence)
        return self._matrix[idx1][idx2]

    def fill_from_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Updates graph instance with vertices and edges extracted from sentences
        :param sentences
        :return:
        """
        check_type(sentences, tuple, Sentence)
        for sent1 in sentences[:len(sentences)]:
            for sent2 in sentences[1:]:
                if sent1.get_encoded != sent2.get_encoded:
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
        check_type(graph, SimilarityMatrix)
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
        check_type(vertex, Sentence)
        check_type(incidental_vertices, list, Sentence, can_be_empty=True)
        check_type(scores, dict, can_be_empty=True)

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
        check_type(n_sentences, int)
        return tuple(sorted(self._scores, key=lambda s: self._scores[s], reverse=True)[:n_sentences])

    def make_summary(self, n_sentences: int = 5) -> str:
        """
        Constructs summary from the most important sentences
        :param n_sentences: number of sentences to include in the summary
        :return: summary
        """
        check_type(n_sentences, int)
        top_sentences = self.get_top_sentences(n_sentences)
        position_srtd_sentences = [s.get_text() for s in sorted(top_sentences, key=lambda s: s.get_position())]
        return '\n'.join(position_srtd_sentences)


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
        self._stop_words = stop_words
        self._punctuation = punctuation
        self._idf_values = idf_values
        self._text_preprocessor = TextPreprocessor(self._stop_words, self._punctuation)
        self._sentence_encoder = SentenceEncoder()
        self._sentence_preprocessor = SentencePreprocessor(self._stop_words, self._punctuation)
        self._paths_to_texts = paths_to_texts
        self._knowledge_database = {}

        for path in self._paths_to_texts:
            self.add_text_to_database(path)

    def add_text_to_database(self, path_to_text: str) -> None:
        """
        Adds the given text to the existing database
        :param path_to_text
        :return:
        """
        check_type(path_to_text, str)
        with open(path_to_text, 'r', encoding='utf-8') as file:
            text = ' '.join([line.strip() for line in file.readline()])
            sentences = self._sentence_preprocessor.get_sentences(text)
            self._sentence_encoder.encode_sentences(sentences)
            tokens = self._text_preprocessor.preprocess_text(text)

            tfidf_adapter = TFIDFAdapter(tokens, self._idf_values)
            tfidf_adapter.train()
            keywords = tfidf_adapter.get_top_keywords(100)

            graph = SimilarityMatrix()
            graph.fill_from_sentences(sentences)

            summarizer = TextRankSummarizer(graph)
            summarizer.train()
            summary = summarizer.make_summary(5)

            inner_dct = {'sentences': sentences, 'keywords': keywords, 'summary': summary}
            self._knowledge_database[path_to_text] = inner_dct

    def _find_texts_close_to_keywords(self, keywords: tuple[str, ...], n_texts: int) -> tuple[str, ...]:
        """
        Finds texts that are similar (i.e. contain the same keywords) to the given keywords
        :param keywords: a sequence of keywords
        :param n_texts: number of texts to find
        :return: the texts' ids
        """
        check_type(keywords, tuple, str)
        check_type(n_texts, int)

        similarity_scores = {}
        for path, info in self._knowledge_database.items():
            text_keywords = info.get('keywords')
            similarity_scores[path] = calculate_similarity(text_keywords, keywords)

        if all(val == 0 for val in similarity_scores.values()):
            raise NoRelevantTextsError('Texts that are related to the query were not found. Try another query.')

        return tuple(sorted(similarity_scores, key=lambda p: (similarity_scores[p], p),
                            reverse=True)[:n_texts])

    def reply(self, query: str, n_summaries: int = 3) -> str:
        """
        Replies to the query
        :param query: the query
        :param n_summaries: the number of summaries to include in the answer
        :return: the answer
        """
        if n_summaries > len(self._knowledge_database):
            raise ValueError
        try:
            check_type(query, str)
        except ValueError as error:
            raise IncorrectQueryError('Incorrect query. Use string as input.') from error
        summaries = self._find_texts_close_to_keywords(TextPreprocessor.preprocess_text(query), n_summaries)
        return 'Ответ:\n,' + '\n\n'.join(list(summaries))
