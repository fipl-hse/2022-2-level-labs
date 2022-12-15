"""
Lab 4
Summarize text using TextRank algorithm
"""
from typing import Union, Any
import re
from itertools import chain, combinations

from lab_3_keywords_textrank.main import TextEncoder, \
    TextPreprocessor, TFIDFAdapter

PreprocessedSentence = tuple[str, ...]
EncodedSentence = tuple[int, ...]


class NoRelevantTextsError(Exception):
    """
    Raised if there are 0 relevant texts.
    """


class IncorrectQueryError(Exception):
    """
    Raised if the query is incorrect.
    """


def arg_check(*args: Any) -> bool:
    """
    Accepts tuples with objects and expected types.
    Raises a ValueError if any object is empty when it should not be or has the wrong type.
    Or if arguments are not tuples. Or tuples are too short.
    Returns True if everything is okay.
    Positions in tuples:
    0 = data
    1 = expected type of data
    2 = expected type of (a) content if data is a list or a tuple, (b) keys if data is a dict
    3 = expected type of values if data is a dict
    None in tuple = allowed to be falsy
    """
    for i in args:
        if not isinstance(i, tuple):
            raise ValueError
        length = len(i) - i.count(None)  # working tuple length (everything but Nones)
        if length < 2 or not isinstance(i[0], i[1]) or i[1] == int and isinstance(i[0], bool):
            raise ValueError
        if isinstance(i[0], (bool, list, tuple, dict)) and None not in i and not i[0]:
            raise ValueError
        if length > 2 and isinstance(i[0], (list, tuple, dict)) and not arg_check(*[(item, i[2]) for item in i[0]]):
            raise ValueError
        if length == 4 and isinstance(i[1], dict) and not arg_check(*[(value, i[3]) for value in i[0].values()]):
            raise ValueError
    return True


class Sentence:
    """
    An abstraction over the real-world sentences
    """

    def __init__(self, text: str, position: int) -> None:
        """
        Constructs all the necessary attributes
        """
        arg_check((text, str), (position, int))
        self._text = text
        self._position = position
        self._preprocessed: tuple[str, ...] = ('', )
        self._encoded: tuple[int, ...] = (0, )

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
        arg_check((text, str))
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
        arg_check((preprocessed_sentence, tuple, str, None))
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
        arg_check((encoded_sentence, tuple, int, None))
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
        arg_check((stop_words, tuple, str, None), (punctuation, tuple, str, None))
        super().__init__(stop_words, punctuation)

    def _split_by_sentence(self, text: str) -> tuple[Sentence, ...]:
        """
        Splits the provided text by sentence
        :param text: the raw text
        :return: a sequence of sentences
        """
        arg_check((text, str))
        text = text.replace('\n', ' ').replace('  ', ' ')
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZА-Я])', text)
        return tuple(Sentence(sentence, count) for count, sentence in enumerate(sentences) if sentence)

    def _preprocess_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their preprocessed versions
        :param sentences: a list of sentences
        :return:
        """
        arg_check((sentences, tuple, Sentence))
        for sent in sentences:
            sent.set_preprocessed(super().preprocess_text(sent.get_text()))

    def get_sentences(self, text: str) -> tuple[Sentence, ...]:
        """
        Extracts the sentences from the given text & preprocesses them
        :param text: the raw text
        :return:
        """
        arg_check((text, str))
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
        arg_check((tokens, tuple, str, None))
        for count, new_token in enumerate((token for token in tokens if token not in self._word2id), self.last_id + 1):
            self._word2id[new_token] = count
            self._id2word[count] = new_token
        self.last_id = max(self._id2word)

    def encode_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their encoded versions
        :param sentences: a sequence of sentences
        :return: a list of sentences with their preprocessed versions
        """
        arg_check((sentences, tuple, Sentence))
        for sentence, preprocessed in [(sent, sent.get_preprocessed()) for sent in sentences]:
            self._learn_indices(preprocessed)
            sentence.set_encoded(tuple(self._word2id[token] for token in preprocessed))


def calculate_similarity(sequence: Union[list, tuple], other_sequence: Union[list, tuple]) -> float:
    """
    Calculates similarity between two sequences using Jaccard index
    :param sequence: a sequence of items
    :param other_sequence: a sequence of items
    :return: similarity score
    """
    arg_check((sequence, (list, tuple), None), (other_sequence, (list, tuple), None))
    try:
        return sum(1 for i in sequence if i in other_sequence) / len(set(chain(sequence, other_sequence)))
    except ZeroDivisionError:
        return 0


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
        self._encoded = []

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
        arg_check((vertex, Sentence))
        encoded = vertex.get_encoded()
        arg_check((encoded in self._encoded, bool))
        return sum(1 for i in self._matrix[self._encoded.index(encoded)] if i > 0)

    def add_edge(self, vertex1: Sentence, vertex2: Sentence) -> None:
        """
        Adds or overwrites an edge in the graph between the specified vertices
        :param vertex1:
        :param vertex2:
        :return:
        """
        arg_check((vertex1, Sentence), (vertex2, Sentence))
        encoded = (vertex1.get_encoded(), vertex2.get_encoded())
        arg_check((encoded[0] != encoded[1], bool))
        for vertex in (vertex1, encoded[0]), (vertex2, encoded[1]):
            if vertex[1] not in self._encoded:
                self._encoded.append(vertex[1])
                self._vertices.append(vertex[0])
                for row in self._matrix:
                    row.append(0)
                self._matrix.append([0 for _ in self._vertices])
        index1 = self._encoded.index(encoded[0])
        index2 = self._encoded.index(encoded[1])
        self._matrix[index1][index2] = self._matrix[index2][index1] = calculate_similarity(encoded[0], encoded[1])

    def get_similarity_score(self, sentence: Sentence, other_sentence: Sentence) -> float:
        """
        Gets the similarity score for two sentences from the matrix
        :param sentence
        :param other_sentence
        :return: the similarity score
        """
        arg_check((sentence, Sentence), (other_sentence, Sentence))
        encoded = (sentence.get_encoded(), other_sentence.get_encoded())
        arg_check((encoded[0] in self._encoded, bool), (encoded[1] in self._encoded, bool))
        return self._matrix[self._encoded.index(encoded[0])][self._encoded.index(encoded[1])]

    def fill_from_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Updates graph instance with vertices and edges extracted from sentences
        :param sentences
        :return:
        """
        arg_check((sentences, tuple, Sentence))
        for pair in combinations(sentences, 2):
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
        arg_check((graph, SimilarityMatrix))
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
        arg_check((vertex, Sentence), (incidental_vertices, list, Sentence, None), (scores, dict, Sentence, float))
        multiplier = sum(scores[i] / self._graph.calculate_inout_score(i) for i in incidental_vertices) - 1
        self._scores[vertex] = 1 + self._damping_factor * multiplier

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
        arg_check((n_sentences, int))
        return tuple(sorted(self._scores, key=lambda x: self._scores[x], reverse=True)[:n_sentences])

    def make_summary(self, n_sentences: int) -> str:
        """
        Constructs summary from the most important sentences
        :param n_sentences: number of sentences to include in the summary
        :return: summary
        """
        arg_check((n_sentences, int))
        sentences_sorted = sorted(self.get_top_sentences(n_sentences), key=lambda x: x.get_position())
        return '\n'.join(i.get_text() for i in sentences_sorted)


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
        arg_check((paths_to_texts, list, str), (stop_words, tuple, str),
                  (punctuation, tuple, str), (idf_values, dict, str, float))
        self._stop_words = stop_words
        self._punctuation = punctuation
        self._idf_values = idf_values
        self._paths_to_texts = paths_to_texts
        self._text_preprocessor = TextPreprocessor(self._stop_words, self._punctuation)
        self._sentence_encoder = SentenceEncoder()
        self._sentence_preprocessor = SentencePreprocessor(self._stop_words, self._punctuation)
        self._knowledge_database = {}
        for path_to_text in paths_to_texts:
            self.add_text_to_database(path_to_text)

    def add_text_to_database(self, path_to_text: str) -> None:
        """
        Adds the given text to the existing database
        :param path_to_text
        :return:
        """
        arg_check((path_to_text, str))
        file = open(path_to_text, 'r', encoding='utf-8')
        text = file.read()

        sentences = self._sentence_preprocessor.get_sentences(text)
        self._sentence_encoder.encode_sentences(sentences)

        tfidf = TFIDFAdapter(self._text_preprocessor.preprocess_text(text), self._idf_values)
        tfidf.train()
        keywords = tfidf.get_top_keywords(100)

        similarity_matrix = SimilarityMatrix()
        similarity_matrix.fill_from_sentences(sentences)
        text_rank_summarizer = TextRankSummarizer(similarity_matrix)
        text_rank_summarizer.train()
        summary = text_rank_summarizer.make_summary(5)

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
        arg_check((n_texts, int), (keywords, tuple, str))
        texts = {k: calculate_similarity(keywords, v['keywords']) for k, v in self._knowledge_database.items()}
        if all(not texts[text] for text in texts):
            raise NoRelevantTextsError('Texts that are related to the query were not found. Try another query.')
        return tuple(sorted(sorted(texts, reverse=True), key=lambda x: texts[x], reverse=True)[:n_texts])

    def reply(self, query: str, n_summaries: int = 3) -> str:
        """
        Replies to the query
        :param query: the query
        :param n_summaries: the number of summaries to include in the answer
        :return: the answer
        """
        if not isinstance(query, str) or not query:
            raise IncorrectQueryError('Incorrect query. Use a non-emtpy string as an input.')
        arg_check((n_summaries, int))
        arg_check((n_summaries <= len(self._knowledge_database), bool))
        keywords = tuple(word for word in query.lower().split() if word not in self._stop_words)
        paths = self._find_texts_close_to_keywords(keywords, n_summaries)
        return 'Ответ:\n' + '\n\n'.join(self._knowledge_database[path]['summary'] for path in paths)
