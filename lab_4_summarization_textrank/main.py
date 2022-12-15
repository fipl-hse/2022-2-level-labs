"""
Lab 4
Summarize text using TextRank algorithm
"""
from typing import Union

from lab_3_keywords_textrank.main import TextEncoder, \
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
        if not (text and isinstance(text, str) and isinstance(position, int) and not isinstance(position, bool)):
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
        if not all(isinstance(i, str) for i in preprocessed_sentence):
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
        if not (isinstance(encoded_sentence, tuple) and all(isinstance(i, int) for i in encoded_sentence)):
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
        if not (isinstance(stop_words, tuple) and all(isinstance(i, str) for i in stop_words)):
            raise ValueError
        if not (isinstance(punctuation, tuple) and all(isinstance(j, str) for j in punctuation)):
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
        final_sentences = []
        clean_txt = ''
        new_txt = text.replace('\n', ' ').replace('  ', ' ')
        for idx, elem in enumerate(new_txt[:-1]):
            if not (elem == ' ' and new_txt[idx - 1] in '.?!' and new_txt[idx + 1].isupper()):
                clean_txt += elem
            else:
                clean_txt += elem
                clean_txt += '  '
        clean_txt += new_txt[-1:]
        sentences = clean_txt.split('  ')
        for index, sent in enumerate(list(sentences)):
            sentence = Sentence(sent.strip(), index)
            final_sentences.append(sentence)
        return tuple(final_sentences)

    def _preprocess_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their preprocessed versions
        :param sentences: a list of sentences
        :return:
        """
        if not (isinstance(sentences, tuple) and all(isinstance(i, Sentence) for i in sentences)):
            raise ValueError
        for sentence in sentences:
            preprocessed = self.preprocess_text(sentence.get_text())
            sentence.set_preprocessed(preprocessed)

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
        if not (isinstance(tokens, tuple) and all(isinstance(i, str) for i in tokens)):
            raise ValueError
        unique_tokens = (i for i in tokens if i not in self._word2id)
        for idx, token in enumerate(unique_tokens, 1000 + len(self._word2id)):
            self._word2id[token] = idx
        for idx, token in self._word2id.items():
            self._id2word[idx] = token

    def encode_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their encoded versions
        :param sentences: a sequence of sentences
        :return: a list of sentences with their preprocessed versions
        """
        if not (isinstance(sentences, tuple) and all(isinstance(i, Sentence) for i in sentences)):
            raise ValueError
        for sent in sentences:
            self._learn_indices(sent.get_preprocessed())
            sent.set_encoded(tuple(self._word2id[token] for token in sent.get_preprocessed()))


def calculate_similarity(sequence: Union[list, tuple], other_sequence: Union[list, tuple]) -> float:
    """
    Calculates similarity between two sequences using Jaccard index
    :param sequence: a sequence of items
    :param other_sequence: a sequence of items
    :return: similarity score
    """
    if not (isinstance(sequence, (list, tuple)) and isinstance(other_sequence, (list, tuple))):
        raise ValueError
    if not sequence or not other_sequence:
        return 0.
    return len(set(sequence) & set(other_sequence)) / len(set(sequence) | set(other_sequence))


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
        count = 0
        for val in self._matrix[self._vertices.index(vertex)]:
            if val > 0:
                count += 1
        return count - 1

    def add_edge(self, vertex1: Sentence, vertex2: Sentence) -> None:
        """
        Adds or overwrites an edge in the graph between the specified vertices
        :param vertex1:
        :param vertex2:
        :return:
        """
        if not (isinstance(vertex1, Sentence) and isinstance(vertex2, Sentence)):
            raise ValueError
        if vertex1.get_encoded() == vertex2.get_encoded():
            raise ValueError
        for vrtx in vertex1, vertex2:
            if vrtx not in self._vertices:
                self._vertices.append(vrtx)
                self._matrix.append([])

        for edges in self._matrix:
            if len(edges) < len(self._vertices):
                edges.extend([0 for _ in range(len(self._vertices) - len(edges))])

        idx1 = self._vertices.index(vertex1)
        idx2 = self._vertices.index(vertex2)
        self._matrix[idx1][idx2] = calculate_similarity(vertex1.get_encoded(), vertex2.get_encoded())
        self._matrix[idx2][idx1] = calculate_similarity(vertex2.get_encoded(), vertex1.get_encoded())
        self._matrix[idx1][idx1] = 1
        self._matrix[idx2][idx2] = 1

    def get_similarity_score(self, sentence: Sentence, other_sentence: Sentence) -> float:
        """
        Gets the similarity score for two sentences from the matrix
        :param sentence
        :param other_sentence
        :return: the similarity score
        """
        if not (isinstance(sentence, Sentence) and isinstance(other_sentence, Sentence)):
            raise ValueError
        if not sentence or not other_sentence:
            raise ValueError
        return self._matrix[self._vertices.index(sentence)][self._vertices.index(other_sentence)]

    def fill_from_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Updates graph instance with vertices and edges extracted from sentences
        :param sentences
        :return:
        """
        if not (sentences and isinstance(sentences, tuple) and all(isinstance(i, Sentence) for i in sentences)):
            raise ValueError
        for sent in sentences:
            for sent2 in sentences:
                if sent.get_encoded() != sent2.get_encoded():
                    self.add_edge(sent, sent2)


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
        if not (isinstance(vertex, Sentence) and isinstance(incidental_vertices, list) and isinstance(scores, dict)):
            raise ValueError
        summa = sum(self._scores[inc] / (1 + self._graph.calculate_inout_score(inc))
                    for inc in incidental_vertices)
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
        return tuple(sorted(self._scores, key=lambda x: self._scores[x], reverse=True)[:n_sentences])

    def make_summary(self, n_sentences: int) -> str:
        """
        Constructs summary from the most important sentences
        :param n_sentences: number of sentences to include in the summary
        :return: summary
        """
        if not (n_sentences and isinstance(n_sentences, int) and not isinstance(n_sentences, bool)):
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
