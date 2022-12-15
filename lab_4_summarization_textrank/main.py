"""
Lab 4
Summarize text using TextRank algorithm
"""
from typing import Union, Any, Type
import re

from lab_3_keywords_textrank.main import TextEncoder, \
    TextPreprocessor, TFIDFAdapter


class NoRelevantTextsError(Exception):
    pass


class IncorrectQueryError(Exception):
    pass


PreprocessedSentence = tuple[str, ...]
EncodedSentence = tuple[int, ...]


def check_type(var: Any, type_of_object: Type) -> None:
    """
    Checks whether type of var is expected one,
    if input is not correct - ValueError is occurred
    """
    if not isinstance(var, type_of_object):
        raise ValueError
    if isinstance(var, int) and isinstance(var, bool):
        raise ValueError


def check_object_and_type(var: Any, object_type: Type, element_type: Type) -> None:
    """
    Checks whether type of var is expected one;
    checks whether type of elements in var are expected one
    if input is not correct - ValueError is occurred
    """
    check_type(var, object_type)
    for ele in var:
        check_type(ele, element_type)


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
        check_object_and_type(preprocessed_sentence, tuple, str)
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
        check_object_and_type(encoded_sentence, tuple, int)
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
        check_object_and_type(stop_words, tuple, str)
        self._stop_words = stop_words
        check_object_and_type(punctuation, tuple, str)
        self._punctuation = punctuation

    def _split_by_sentence(self, text: str) -> tuple[Sentence, ...]:
        """
        Splits the provided text by sentence
        :param text: the raw text
        :return: a sequence of sentences
        """
        check_type(text, str)
        text = text.replace('\n', ' ').replace('  ', ' ')
        text_two = re.split(r'(?<=[.!?])\s+(?=[A-ZА-Я])', text)
        list_of_sentences = []
        for idx, sentence in enumerate(text_two):
            list_of_sentences.append(Sentence(sentence, idx))
        return tuple(list_of_sentences)

    def _preprocess_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their preprocessed versions
        :param sentences: a list of sentences
        :return:
        """

        check_object_and_type(sentences, tuple, Sentence)
        for i in sentences:
            preprocessed = self.preprocess_text(i.get_text())
            i.set_preprocessed(preprocessed)

    def get_sentences(self, text: str) -> tuple[Sentence, ...]:
        """
        Extracts the sentences from the given text & preprocesses them
        :param text: the raw text
        :return:
        """
        check_type(text, str)
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
        self.last_id = 999

    def _learn_indices(self, tokens: tuple[str, ...]) -> None:
        """
        Fills attributes mapping words and integer equivalents to each other
        :param tokens: a sequence of string tokens
        :return:
        """
        check_object_and_type(tokens, tuple, str)
        id_tokens = []
        for token in tokens:
            if token not in self._word2id:
                id_tokens.append(token)
        for idx, word in enumerate(id_tokens, self.last_id + 1):
            self._word2id[word] = idx
            self._id2word[idx] = word
        self.last_id = max(self._word2id.values())

    def encode_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their encoded versions
        :param sentences: a sequence of sentences
        :return: a list of sentences with their preprocessed versions
        """
        check_object_and_type(sentences, tuple, Sentence)
        for one_sentence in sentences:
            self._learn_indices(one_sentence.get_preprocessed())
            one_sentence.set_encoded(tuple(self._word2id[word] for word in one_sentence.get_preprocessed()))


def calculate_similarity(sequence: Union[list, tuple], other_sequence: Union[list, tuple]) -> float:
    """
    Calculates similarity between two sequences using Jaccard index
    :param sequence: a sequence of items
    :param other_sequence: a sequence of items
    :return: similarity score
    """
    if not isinstance(sequence, (list, tuple)) or not isinstance(other_sequence, (list, tuple)):
        raise ValueError
    if not sequence or not sequence:
        return 0
    intersection = list(set(sequence) & set(other_sequence))
    unification = list(set().union(sequence, other_sequence))
    measure_of_jacquard = float(len(intersection) / len(unification))
    return measure_of_jacquard


class SimilarityMatrix():
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
        summa = 0
        for score in self._matrix[self._vertices.index(vertex)]:
            if score > 0:
                summa += 1
        return summa - 1

    def add_edge(self, vertex1: Sentence, vertex2: Sentence) -> None:
        """
        Adds or overwrites an edge in the graph between the specified vertices
        :param vertex1:
        :param vertex2:
        :return:
        """
        if vertex1 == vertex2:
            raise ValueError
        for vertex in vertex1, vertex2:
            check_type(vertex, Sentence)
            if vertex not in self._vertices:
                self._vertices.append(vertex)
                self._matrix.append([0])
        for edges_list in self._matrix:
            if len(edges_list) < len(self._vertices):
                edges_list.extend([0 for _ in range(len(self._vertices) - len(edges_list))])
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
        check_type(sentence, Sentence)
        check_type(other_sentence, Sentence)
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

        check_object_and_type(sentences, tuple, Sentence)
        if not sentences:
            raise ValueError
        check_type(sentences, tuple)
        for idx1, first_sentence in enumerate(sentences):
            for idx2, second_sentence in enumerate(sentences):
                if idx1 == idx2:
                    break
                self.add_edge(first_sentence, second_sentence)


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
        check_type(scores, dict)
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
        check_type(n_sentences, int)
        return tuple(sorted(self._scores, key=lambda item: self._scores[item], reverse=True)[:n_sentences])

    def make_summary(self, n_sentences: int) -> str:
        """
        Constructs summary from the most important sentences
        :param n_sentences: number of sentences to include in the summary
        :return: summary
        """
        check_type(n_sentences, int)
        top_sentences = sorted(self.get_top_sentences(n_sentences), key=lambda item: item.get_position())
        return '\n'.join([i.get_text() for i in top_sentences])


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

        for one_path in paths_to_texts:
            self.add_text_to_database(one_path)

    def add_text_to_database(self, path_to_text: str) -> None:
        """
        Adds the given text to the existing database
        :param path_to_text
        :return:
        """
        check_type(path_to_text, str)
        with open(path_to_text, encoding='utf-8') as file:
            text = file.read()

        preprocessor = SentencePreprocessor(self._stop_words, self._punctuation)
        sentences = preprocessor.get_sentences(text) #preprocessing texts, remove puctuation and stop_words
        self._sentence_encoder.encode_sentences(sentences) #encode sentences

        preprocess_text = self._text_preprocessor.preprocess_text(text)
        tf_idf_adapter = TFIDFAdapter(preprocess_text, self._idf_values)
        tf_idf_adapter.train()
        keywords = tf_idf_adapter.get_top_keywords(100) #extracting keywords using TFIDF

        matrix = SimilarityMatrix()
        matrix.fill_from_sentences(sentences)
        summarizer = TextRankSummarizer(matrix)
        summarizer.train()
        summary = summarizer.make_summary(5) #create summary using TextRankSummarizer

        self._knowledge_database[path_to_text] = {'sentences': sentences, 'keywords': keywords, 'summary': summary}

    def _find_texts_close_to_keywords(self, keywords: tuple[str, ...], n_texts: int) -> tuple[str, ...]:
        """
        Finds texts that are similar (i.e. contain the same keywords) to the given keywords
        :param keywords: a sequence of keywords
        :param n_texts: number of texts to find
        :return: the texts' ids
        """
        check_object_and_type(keywords, tuple, str)
        check_type(n_texts, int)
        close_sim_texts = {}
        for path_to_text, info in self._knowledge_database.items():
            close_sim_texts[path_to_text] = calculate_similarity(info['keywords'], keywords)
        if not any(close_sim_texts.values()):
            raise NoRelevantTextsError('Texts that are related to the query were not found. Try another query.')
        return tuple(sorted(close_sim_texts, key=lambda path: (close_sim_texts[path], path), reverse=True)[:n_texts])

    def reply(self, query: str, n_summaries: int = 3) -> str:
        """
        Replies to the query
        :param query: the query
        :param n_summaries: the number of summaries to include in the answer
        :return: the answer
        """
        if not isinstance(query, str) or not query:
            raise IncorrectQueryError('Incorrect query. Use string as input.')
        if not isinstance(n_summaries, int):
            raise ValueError
        if len(self._knowledge_database) < n_summaries:
            raise ValueError
        user_question = self._text_preprocessor.preprocess_text(query)
        summaries = self._find_texts_close_to_keywords(user_question, n_summaries)
        answer = []
        for sentence in summaries:
            answer.append(self._knowledge_database[sentence]['summary'])
        return 'Ответ:\n' + '\n\n'.join(answer)
