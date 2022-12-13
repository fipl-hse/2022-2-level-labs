"""
Lab 4
Summarize text using TextRank algorithm
"""
from typing import Union

import re
from lab_3_keywords_textrank.main import TextEncoder, \
    TextPreprocessor, TFIDFAdapter


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
        if not isinstance(text, str) or isinstance(position, bool) or not isinstance(position, int):
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
        if not isinstance(text, str) and text:
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
        for word in preprocessed_sentence:
            if not isinstance(word, str):
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
        for number in encoded_sentence:
            if not isinstance(number, int):
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

        if not isinstance(stop_words, tuple) or not isinstance(punctuation, tuple):
            raise ValueError

        for stop_word in stop_words:
            if not isinstance(stop_word, str):
                raise ValueError

        for sign in punctuation:
            if not isinstance(sign, str):
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
        text = text.replace('\n', ' ')
        text = text.replace('  ', ' ')
        sentences = re.split(r'(?<=[?!.])\s+(?=[А-ЯA-Z])', text)
        sentences_tuple = []
        for position, value in enumerate(sentences):
            sent = Sentence(value, position)
            sentences_tuple.append(sent)
        return tuple(sentences_tuple)

    def _preprocess_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their preprocessed versions
        :param sentences: a list of sentences
        :return:
        """
        if not isinstance(sentences, tuple):
            raise ValueError
        for sentence in sentences:
            sentence.set_preprocessed(super().preprocess_text(sentence.get_text()))

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

    def __init__(self) -> None:
        """
        Constructs all the necessary attributes
        """

        super().__init__()
        self.last_index = 999


    def _learn_indices(self, tokens: tuple[str, ...]) -> None:
        """
        Fills attributes mapping words and integer equivalents to each other
        :param tokens: a sequence of string tokens
        :return:
        """
        if not isinstance(tokens, tuple):
            raise ValueError
        updated_tokens = []
        for token in tokens:
            if token not in self._word2id:
                updated_tokens.append(token)
        for index, token in enumerate(updated_tokens, self.last_index + 1):
            self._word2id[token] = index
            self._id2word[index] = token
        self.last_index = max(self._id2word)

    def encode_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their encoded versions
        :param sentences: a sequence of sentences
        :return: a list of sentences with their preprocessed versions
        """
        if not isinstance(sentences, tuple):
            raise ValueError
        for token in sentences:
            self._learn_indices(token.get_preprocessed())
            token.set_encoded(tuple(self._word2id[word] for word in token.get_preprocessed()))



def calculate_similarity(sequence: Union[list, tuple], other_sequence: Union[list, tuple]) -> float:
    """
    Calculates similarity between two sequences using Jaccard index
    :param sequence: a sequence of items
    :param other_sequence: a sequence of items
    :return: similarity score
    """
    if not isinstance(sequence, list | tuple) or not isinstance(other_sequence, list | tuple):
        raise ValueError
    if not sequence or not sequence:
        return 0

    sequence1 = set(sequence)
    sequence2 = set(other_sequence)
    junction = sequence1.intersection(sequence2)
    return float(len(junction)) / (len(sequence1) + len(sequence2) - len(junction))


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
        summarization = 0
        for index in self._matrix[self._vertices.index(vertex)]:
            if index > 0:
                summarization += 1
        return summarization

    def add_edge(self, vertex1: Sentence, vertex2: Sentence) -> None:
        """
        Adds or overwrites an edge in the graph between the specified vertices
        :param vertex1:
        :param vertex2:
        :return:
        """
        if not (isinstance(vertex1, Sentence) or isinstance(vertex2, Sentence)):
            raise ValueError
        if vertex1 == vertex2:
            raise ValueError

        for vertex in vertex1, vertex2:
            if vertex not in self._vertices:
                self._vertices.append(vertex)
                self._matrix.append([])

        for i in self._matrix:
            for _ in self._vertices:
                if len(i) < len(self._vertices):
                    i.append(0)

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
        if (sentence or other_sentence) not in self._vertices:
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
        if not isinstance(sentences, tuple) or not sentences:
            raise ValueError
        pairs = []
        for sentence1 in sentences:
            for sentence2 in sentences:
                if sentence1.get_encoded() != sentence2.get_encoded():
                    pairs.append((sentence1, sentence2))
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
        if not isinstance(vertex, Sentence) or not isinstance(scores, dict) and isinstance(incidental_vertices, list)\
                and isinstance(incidental_vertices, Sentence):
            raise ValueError
        summa = sum((1 / self._graph.calculate_inout_score(vertex)) * scores[vertex]
                    for vertex in incidental_vertices)
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
        if not isinstance(n_sentences, int) or isinstance(n_sentences, bool):
            raise ValueError
        return tuple(sorted(self._scores, key=lambda token: self._scores[token], reverse=True))[:n_sentences]

    def make_summary(self, n_sentences: int) -> str:
        """
        Constructs summary from the most important sentences
        :param n_sentences: number of sentences to include in the summary
        :return: summary
        """
        if not isinstance(n_sentences, int) or isinstance(n_sentences, bool):
            raise ValueError
        top_sent = sorted(self.get_top_sentences(n_sentences), key=lambda x: x.get_position())
        return '\n'.join([sentence.get_text() for sentence in top_sent])


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
        self._paths_to_texts: list[str] = paths_to_texts
        self._stop_words: tuple[str, ...] = stop_words
        self._punctuation: tuple[str, ...] = punctuation
        self._idf_values: dict[str, float] = idf_values
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
        if not isinstance(path_to_text, str):
            raise ValueError
        with open(path_to_text, encoding='utf-8') as file:
            text = file.read()


        processed_sentences = self._sentence_preprocessor.get_sentences(text)
        self._sentence_encoder.encode_sentences(processed_sentences)

        tf_idf = TFIDFAdapter(self._text_preprocessor.preprocess_text(text), self._idf_values)
        tf_idf.train()
        keywords = tf_idf.get_top_keywords(100)

        matrix = SimilarityMatrix()
        matrix.fill_from_sentences(processed_sentences)

        text_rank = TextRankSummarizer(matrix)
        text_rank.train()
        summary = text_rank.make_summary(5)

        self._knowledge_database[path_to_text] = {'sentences': processed_sentences, 'keywords': keywords, 'summary': summary}

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
