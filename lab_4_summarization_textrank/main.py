"""
Lab 4
Summarize text using TextRank algorithm
"""
from typing import Union
import re
from lab_3_keywords_textrank.main import TextEncoder, TextPreprocessor, TFIDFAdapter

PreprocessedSentence = tuple[str, ...]
EncodedSentence = tuple[int, ...]


class NoRelevantTextsError(Exception):
    """
    Raised if there are no relevant texts.
    """
    print ('NoRelevantTextsError')


class IncorrectQueryError(Exception):
    """
    Raised if query is incorrect.
    """
    print (' IncorrectQueryError')


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
        self._preprocessed: tuple[str, ...] = ('', )
        self._encoded: tuple[int, ...] = (0, )

    def get_position(self) -> int:
        """
        Returns the position of the sentence in the text.
        """
        return self._position

    def set_text(self, text: str) -> None:
        """
        Sets the attribute.
        """
        if not isinstance(text, str):
            raise ValueError
        self._text = text

    def get_text(self) -> str:
        """
        Returns the text.
        """
        return self._text

    def set_preprocessed(self, preprocessed_sentence: PreprocessedSentence) -> None:
        """
        Sets the attribute.
        """
        if not isinstance(preprocessed_sentence, tuple):
            raise ValueError
        for i in preprocessed_sentence:
            if not isinstance(i, str):
                raise ValueError
        self._preprocessed = preprocessed_sentence

    def get_preprocessed(self) -> PreprocessedSentence:
        """
        Returns the attribute.
        """
        return self._preprocessed

    def set_encoded(self, encoded_sentence: EncodedSentence) -> None:
        """
        Sets the attribute (None).
        """
        if not isinstance(encoded_sentence, tuple):
            raise ValueError
        for i in encoded_sentence:
            if not isinstance(i, int):
                raise ValueError
        self._encoded = encoded_sentence

    def get_encoded(self) -> EncodedSentence:
        """
        Returns the encoded sentence (a sequence of numbers).
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
        for word in stop_words:
            if not isinstance(word, str):
                raise ValueError
        for i in punctuation:
            if not isinstance(i, str):
                raise ValueError
        super().__init__(stop_words, punctuation)

    def _split_by_sentence(self, text: str) -> tuple[Sentence, ...]:
        """
        Splits the provided text by sentence.
        """
        if not isinstance(text, str):
            raise ValueError
        text = text.replace('\n', ' ').replace('  ', ' ')
        splited_text = re.split(r'(?<=[.?!])\s+(?=[A-ZА-Я])', text)
        sentences = []
        for index, value in enumerate(splited_text):
            sentences.append(Sentence(value, index))
        return tuple(sentences)

    def _preprocess_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their versions.
        """
        if not isinstance(sentences, tuple):
            raise ValueError
        for sentence in sentences:
            preprocessed_sentence = self.preprocess_text(sentence.get_text())
            sentence.set_preprocessed(preprocessed_sentence)

    def get_sentences(self, text: str) -> tuple[Sentence, ...]:
        """
        Extracts the sentences from the given text and preprocesses them.
        """
        if not isinstance(text, str):
            raise ValueError
        splited_text = self._split_by_sentence(text)
        self._preprocess_sentences(splited_text)
        return splited_text


class SentenceEncoder(TextEncoder):
    """
    A class to encode string sequence into matching integer sequence.
    """

    def __init__(self) -> None:
        """
        Constructs all the necessary attributes.
        """
        super().__init__()
        self.last_value = 1000

    def _learn_indices(self, tokens: tuple[str, ...]) -> None:
        """
        Fills attributes mapping words and integer equivalents to each other.
        """
        if not isinstance(tokens, tuple):
            raise ValueError

        for index, token in enumerate((token for token in tokens if token not in self._word2id), self.last_value):
            self._word2id[token] = index
            self._id2word[self.last_value] = token
            self.last_value += 1

    def encode_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Enriches the instances of sentences with their encoded versions.
        """
        if not isinstance(sentences, tuple):
            raise ValueError

        for sentence in sentences:
            preprocessed_sent = sentence.get_preprocessed()
            self._learn_indices(preprocessed_sent)
            sentence.set_encoded(tuple(self._word2id[word] for word in preprocessed_sent))


def calculate_similarity(sequence: Union[list, tuple], other_sequence: Union[list, tuple]) -> float:
    """
    Calculates similarity between two sequences using Jaccard index.
    """
    if not isinstance(sequence, (list,tuple)) or not isinstance(other_sequence, (list,tuple)):
        raise ValueError
    if not sequence or not other_sequence:
        return 0
    return len(set(sequence) & set(other_sequence)) / len(set(sequence) | set(other_sequence))


class SimilarityMatrix:
    """
    A class to represent relations between sentences.
    """

    _matrix: list[list[float]]

    def __init__(self) -> None:
        """
        Constructs necessary attributes.
        """
        self._matrix = []
        self._vertices = []

    def get_vertices(self) -> tuple[Sentence, ...]:
        """
        Returns a sequence of all vertices present in the graph.
        """
        return tuple(self._vertices)

    def calculate_inout_score(self, vertex: Sentence) -> int:
        """
        Retrieves a number of vertices that are similar (i.e. have similarity score > 0) to the input one.
        """
        if vertex not in self._vertices:
            raise ValueError
        index = self._vertices.index(vertex)
        return len([i for i in self._matrix[index] if i > 0])

    def add_edge(self, vertex1: Sentence, vertex2: Sentence) -> None:
        """
        Adds or overwrites an edge in the graph between the specified vertices.
        """
        if vertex1 == vertex2:
            raise ValueError
        for vertex in vertex1, vertex2:
            if vertex in self._vertices:
                continue
            self._vertices.append(vertex)
            self._matrix.append([])
            for i in self._matrix:
                if len(i) < len(self._vertices):
                    i.extend([0 for _ in range(len(self._vertices) - len(i))])

        encoded1 = vertex1.get_encoded()
        encoded2 = vertex2.get_encoded()
        vert1_index = self._vertices.index(vertex1)
        vert2_index = self._vertices.index(vertex2)
        self._matrix[vert1_index][vert2_index] = calculate_similarity(encoded1, encoded2)
        self._matrix[vert2_index][vert1_index] = calculate_similarity(encoded1, encoded2)


    def get_similarity_score(self, sentence: Sentence, other_sentence: Sentence) -> float:
        """
        Gets the similarity score for two sentences from the matrix.
        """
        if sentence not in self._vertices or other_sentence not in self._vertices:
            raise ValueError
        index1 = self._vertices.index(sentence)
        index2 = self._vertices.index(other_sentence)
        return self._matrix[index1][index2]

    def fill_from_sentences(self, sentences: tuple[Sentence, ...]) -> None:
        """
        Updates graph instance with vertices and edges extracted from sentences.
        """
        if not isinstance(sentences, tuple) or not sentences:
            raise ValueError
        for sentence1 in sentences:
            for sentence2 in sentences:
                if sentence1.get_encoded() != sentence2.get_encoded():
                    self.add_edge(sentence1, sentence2)


class TextRankSummarizer:
    """
    TextRank for summarization.
    """

    _scores: dict[Sentence, float]
    _graph: SimilarityMatrix

    def __init__(self, graph: SimilarityMatrix) -> None:
        """
        Constructs all the necessary attributes or None.
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
        Changes vertex significance score with algorithm-specific formula.
        """
        sum_list = []
        for vert in incidental_vertices:
            inout_score = self._graph.calculate_inout_score(vert)
            sum_list.append(1 / inout_score * self._scores[vert])
        summa = sum(sum_list)
        vertex_score = (1 - self._damping_factor) + self._damping_factor * summa
        self._scores[vertex] = vertex_score

    def train(self) -> None:
        """
        Iteratively computes significance scores for vertices.
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

            if sum(abs_score_diff) <= self._convergence_threshold:
                print("Converging at iteration " + str(iteration) + "...")
                break

    def get_top_sentences(self, n_sentences: int) -> tuple[Sentence, ...]:
        """
        Retrieves top n most important sentences in the encoded text.
        """
        if not isinstance(n_sentences, int) or isinstance(n_sentences, bool):
            raise ValueError
        return tuple(sorted(self._scores, reverse=True, key=lambda sentence: self._scores[sentence])[:n_sentences])

    def make_summary(self, n_sentences: int) -> str:
        """
        Constructs summary from the most important sentences.
        """
        if not isinstance(n_sentences, int) or isinstance(n_sentences, bool):
            raise ValueError
        sorted_lst = sorted(self.get_top_sentences(n_sentences), key=lambda sentence: sentence.get_position())
        summary = [i.get_text() for i in sorted_lst]
        return '\n'.join(summary)


class Buddy:
    """
    All-knowing entity
    """

    def __init__(self, paths_to_texts: list[str], stop_words: tuple[str, ...], punctuation: tuple[str, ...], idf_values: dict[str, float]):
        """
        Constructs all the necessary attributes.
        """
        self._stop_words = stop_words
        self._punctuation = punctuation
        self._idf_values = idf_values
        self._text_preprocessor = TextPreprocessor(self._stop_words, self._punctuation)
        self._sentence_encoder = SentenceEncoder()
        self._sentence_preprocessor = SentencePreprocessor(self._stop_words, self._punctuation)
        self._paths_to_texts = paths_to_texts
        self._knowledge_database = {}
        for text in paths_to_texts:
            self.add_text_to_database(text)

    def add_text_to_database(self, path_to_text: str) -> None:
        """
        Adds the given text to the existing database.
        """
        if not isinstance(path_to_text, str):
            raise ValueError
        with open(path_to_text, 'r', encoding='utf-8') as f:
            text = f.read()
        preprocessed_text = self._text_preprocessor.preprocess_text(text)
        sentences = self._sentence_preprocessor.get_sentences(text)
        self._sentence_encoder.encode_sentences(sentences)
        tfidf = TFIDFAdapter(preprocessed_text, self._idf_values)
        tfidf.train()
        keywords = tfidf.get_top_keywords(100)
        matrix = SimilarityMatrix()
        matrix.fill_from_sentences(sentences)
        summarizer = TextRankSummarizer(matrix)
        summarizer.train()
        summary = summarizer.make_summary(5)
        self._knowledge_database[path_to_text] = {
            'sentences': sentences,
            'keywords': keywords,
            'summary': summary
        }

    def _find_texts_close_to_keywords(self, keywords: tuple[str, ...], n_texts: int) -> tuple[str, ...]:
        """
        Contains the same keyword to the given keywords.
        """
        if not isinstance(keywords, tuple) or not isinstance(n_texts, int):
            raise ValueError
        close_texts = {}
        for k, val in self._knowledge_database.items():
            similarity = calculate_similarity(keywords, val['keywords'])
            close_texts[k] = similarity
        if not any(close_texts.values()):
            raise NoRelevantTextsError('Texts that are related to the query were not found. Try another query.')
        sorted_texts = sorted(close_texts, reverse=True)
        return tuple(sorted(sorted_texts, key=lambda x: close_texts[x], reverse=True))[:n_texts]

    def reply(self, query: str, n_summaries: int = 3) -> str:
        """
        Replies to the query.
        """
        if not isinstance(n_summaries, int) or len(self._knowledge_database) < n_summaries:
            raise ValueError
        if not query or not isinstance(query, str):
            raise IncorrectQueryError('Incorrect query. Use string as input.')
        keywords = self._text_preprocessor.preprocess_text(query)
        summaries = self._find_texts_close_to_keywords(keywords, n_summaries)
        return 'The answer:\n' + '\n\n'.join(self._knowledge_database[text]['summary'] for text in summaries)
