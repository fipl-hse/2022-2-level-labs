"""
Lab 4
Summarize text using TextRank algorithm
"""
from typing import Union, Any, Type

import re
from lab_3_keywords_textrank.main import TextEncoder, \
    TextPreprocessor, TFIDFAdapter


PreprocessedSentence = tuple[str, ...]
EncodedSentence = tuple[int, ...]


def check_types(variable: Any, possible_var_type: list[Type]) -> None:
    """
    Checks if the variable is of an appropriate type
    param: variable
    param: possible_var_type
    param: container_value_type (default = None)
    return:
    """
    # error_result = []  # save 'not' bools of check result
    # first_type = type(possible_var_type[0])
    #
    # if not variable:
    #     raise ValueError
    #
    # if len(possible_var_type) == 1:
    #     if isinstance(possible_var_type[0], bool):
    #         a_check = not (isinstance(variable, first_type))
    #         error_result.append(a_check)
    #     else:
    #         a_check = not (isinstance(variable, first_type) and not isinstance(variable, bool))
    #         error_result.append(a_check)
    # else:
    #     second_type = type(possible_var_type[1])
    #     a_check = not (isinstance(variable, (first_type, second_type)) and not isinstance(variable, bool))
    #     error_result.append(a_check)
    # if any(error_result):
    #     raise ValueError
    error_result = []  # save 'not' bools of check result
    for one_type in possible_var_type:
        if isinstance(one_type, bool):
            a_check = not (isinstance(variable, one_type))
            error_result.append(a_check)
        else:
            a_check = not isinstance(variable, one_type) or isinstance(variable, bool)
            error_result.append(a_check)

    if any(error_result):
        raise ValueError


def check_dicts(variable: Any, container_types: list[Type]) -> None:
    """
    checks the dictionary to be a dictionary and to have
    appropriate types of keys and values
    """
    check_types(variable, [dict])
    key_type = container_types[0]
    val_type = container_types[1]
    for key, value in variable.items():
        check_types(key, [key_type])
        check_types(value, [val_type])


def check_iterable_var(variable: Any, possible_var_type: list[Type], container_value_type: [list[Type]]) -> None:
    """
    checks the iterable variables (except dicts)
    """
    check_types(variable, possible_var_type)

    for element in variable:
        check_types(element, container_value_type)



class Sentence:
    """
    An abstraction over the real-world sentences
    """

    def __init__(self, text: str, position: int) -> None:
        """
        Constructs all the necessary attributes
        """
        check_types(text, [str])
        check_types(position, [int])
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
        check_types(text, [str])
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
        check_iterable_var(preprocessed_sentence, [tuple], [str])
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
        check_iterable_var(encoded_sentence, [tuple], [int])
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
        check_iterable_var(stop_words, [tuple], [str])
        check_iterable_var(punctuation, [tuple], [str])
        super().__init__(stop_words, punctuation)

    def _split_by_sentence(self, text: str) -> tuple[Sentence, ...]:
        """
        Splits the provided text by sentence
        :param text: the raw text
        :return: a sequence of sentences
        """
        check_types(text, [str])

        sent_list = []
        split_text = re.split(r'(?<=[.?!])\s+(?=[А-Я A-Z])', text)

        for idx, txt_element in enumerate(split_text):
            txt_element = txt_element.replace('\n', ' ').replace('  ', ' ')
            if txt_element:
                sent_list.append(Sentence(txt_element, idx))
        return tuple(sent_list)

    def _preprocess_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their preprocessed versions
        :param sentences: a list of sentences
        :return:
        """
        check_iterable_var(sentences, [tuple], [Sentence])
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
        check_types(text, [str])

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
        check_iterable_var(tokens, [tuple], [str])

        for one_token in tokens:
            self._word2id[one_token] = self._word2id.get(one_token, self._last_idx)
            self._id2word[self._last_idx] = self._id2word.get(self._last_idx, one_token)
            self._last_idx += 1

    def encode_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their encoded versions
        :param sentences: a sequence of sentences
        :return: a list of sentences with their preprocessed versions
        """
        check_iterable_var(sentences, [tuple], [Sentence])

        coded_sent = []
        for one_sentence in sentences:
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
    if not sequence or not other_sequence:
        return 0

    # check_types(sequence, [list, tuple])
    # check_types(other_sequence, [list, tuple])
    for variable in (sequence, other_sequence):
        if not isinstance(variable, (list, tuple)):
            raise ValueError

    sequences = [*sequence, *other_sequence]
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
            check_types(variable, [Sentence])

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
        check_types(sentence, [Sentence])
        check_types(other_sentence, [Sentence])

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
        if not sentences:
            raise ValueError
        check_types(sentences, [tuple])

        for idx1 in range(len(sentences)-1):
            for idx2 in range(1, len(sentences)):
                sent1 = sentences[idx1]
                sent2 = sentences[idx2]

                check_types(sent1, [Sentence])
                check_types(sent2, [Sentence])

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
        check_types(graph, [SimilarityMatrix])
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
        check_types(vertex, [Sentence])
        check_iterable_var(incidental_vertices, [list], [Sentence])
        check_dicts(scores, [Sentence, float])

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
        check_types(n_sentences, [int])
        sort_list = sorted(self._scores, key=lambda x: self._scores[x], reverse=True)[:n_sentences]
        return tuple(sort_list)

    def make_summary(self, n_sentences: int) -> str:
        """
        Constructs summary from the most important sentences
        :param n_sentences: number of sentences to include in the summary
        :return: summary
        """
        check_types(n_sentences, [int])

        important_list = sorted(self.get_top_sentences(n_sentences), key=lambda x: x.get_position())
        result = []

        for element in important_list:
            result.append(element.get_text())

        return '\n'.join(result)


class NoRelevantTextsError(Exception):
    """
    Raises when no relevant texts for summary is found
    """
    pass


class IncorrectQueryError(Exception):
    """
    Raises when
    """
    pass


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
        check_iterable_var(paths_to_texts, [list], [str])
        check_iterable_var(stop_words, [tuple], [str])
        check_iterable_var(punctuation, [tuple], [str])
        check_dicts(idf_values, [str, float])
        # for key, values in idf_values.items():
        #     if not isinstance(key, str) or not isinstance(values, float):
        #         raise ValueError
        self._stop_words = stop_words
        self._punctuation = punctuation
        self._idf_values = idf_values
        self._text_preprocessor = TextPreprocessor(stop_words, punctuation)
        self._sentence_encoder = SentenceEncoder()
        self._sentence_preprocessor = SentencePreprocessor(stop_words, punctuation)
        self._paths_to_texts = paths_to_texts
        self._knowledge_database = {}

        for file_path in paths_to_texts:
            self.add_text_to_database(file_path)

    def add_text_to_database(self, path_to_text: str) -> None:
        """
        Adds the given text to the existing database
        :param path_to_text
        :return:
        """
        check_types(path_to_text, [str])
        with open(path_to_text, encoding='utf-8') as file:
            text = file.read()
        sentences = self._sentence_preprocessor.get_sentences(text)
        self._sentence_encoder.encode_sentences(sentences)

        tokens = self._text_preprocessor.preprocess_text(text)
        tfidf_adapter = TFIDFAdapter(tokens, self._idf_values)
        train_result = tfidf_adapter.train()
        if train_result == -1:
            raise ValueError
        keywords = tfidf_adapter.get_top_keywords(100)

        matrix = SimilarityMatrix()
        matrix.fill_from_sentences(sentences)
        summarizer = TextRankSummarizer(matrix)
        summarizer.train()
        the_summary = summarizer.make_summary(5)

        self._knowledge_database[path_to_text] = {
            'sentences': sentences,
            'keywords': keywords,
            'summary': the_summary
        }

    def _find_texts_close_to_keywords(self, keywords: tuple[str, ...], n_texts: int) -> tuple[str, ...]:
        """
        Finds texts that are similar (i.e. contain the same keywords) to the given keywords
        :param keywords: a sequence of keywords
        :param n_texts: number of texts to find
        :return: the texts' ids
        """
        check_iterable_var(keywords, [tuple], [str])
        check_types(n_texts, [int])
        similar_texts = {}

        for key, value in self._knowledge_database.items():
            text_keywords = value['keywords']
            similarity_val = calculate_similarity(keywords, text_keywords)
            similar_texts[key] = similarity_val

        if not any(similar_texts.values()):
            raise NoRelevantTextsError('Texts that are related to the query were not found. Try another query.')

        sorted_similar_texts = sorted(similar_texts, key=lambda x: (similar_texts[x], x), reverse=True)[:n_texts]

        return tuple(sorted_similar_texts)

    def reply(self, query: str, n_summaries: int = 3) -> str:
        """
        Replies to the query
        :param query: the query
        :param n_summaries: the number of summaries to include in the answer
        :return: the answer
        """
        if not isinstance(query, str) or not query:
            raise IncorrectQueryError('Incorrect query. Use string as input')

        check_types(n_summaries, [int])

        if len(self._knowledge_database) < n_summaries:
            raise ValueError

        query_keywords = self._text_preprocessor.preprocess_text(query)

        summaries = self._find_texts_close_to_keywords(query_keywords, n_summaries)
        answer = []
        for element in summaries:
            answer.append(self._knowledge_database[element]['summary'])
        return 'Ответ:\n' + '\n\n'.join(answer)
