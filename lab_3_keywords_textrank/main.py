"""
Lab 3
Extract keywords based on TextRank algorithm
"""
from pathlib import Path
from typing import Optional, Union


class TextPreprocessor:
    """
    A class to preprocess raw text
    ...
    Attributes
    ----------
    _stop_words: tuple[str, ...]
        insignificant words to remove from tokens
    _punctuation: tuple[str, ...]
        punctuation symbols to remove during text cleaning
    Methods
    -------
    _clean_and_tokenize(text: str) -> tuple[str, ...]:
        Removes punctuation, casts to lowercase, splits into tokens.
    _remove_stop_words(tokens: tuple[str, ...] -> tuple[str, ...]:
        Filters tokens, removing stop words
    preprocess_text(text: str) -> tuple[str, ...]:
        Produces filtered clean lowercase tokens from raw text
    """
    # Step 1.1
    def __init__(self, stop_words: tuple[str, ...], punctuation: tuple[str, ...]) -> None:
        """
        Constructs all the necessary attributes for the text preprocessor object
        Parameters
        ----------
            stop_words : tuple[str, ...]
                insignificant words to remove from tokens
            punctuation : tuple[str, ...]
                punctuation symbols to remove during text cleaning
        """
        self._stop_words = stop_words
        self._punctuation = punctuation

    # Step 1.2
    def _clean_and_tokenize(self, text: str) -> tuple[str, ...]:
        """
        Removes punctuation, casts to lowercase, splits into tokens.
        Parameters
        ----------
            text : str
                raw text
        Returns
        -------
            tuple[str, ...]
                clean lowercase tokens
        """
        for sign in self._punctuation:
            text = text.replace(sign, '')
        return tuple(text.lower().split())

    # Step 1.3
    def _remove_stop_words(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
        """
        Filters tokens, removing stop words
        Parameters
        ----------
            tokens : tuple[str, ...]
                tokens containing stop-words
        Returns
        -------
            tuple[str, ...]
                tokens without stop-words
        """
        cleaned_text = []
        for word in tokens:
            if word in self._stop_words:
                continue
            cleaned_text.append(word)
        return tuple(cleaned_text)

    # Step 1.4
    def preprocess_text(self, text: str) -> tuple[str, ...]:
        """
        Produces filtered clean lowercase tokens from raw text
        Parameters
        ----------
            text : str
                raw text
        Returns
        -------
            tuple[str, ...]
                clean lowercase tokens with no stop-words
        """
        return self._remove_stop_words(self._clean_and_tokenize(text))


class TextEncoder:
    """
    A class to encode string sequence into matching integer sequence
    ...
    Attributes
    ----------
    _word2id: dict[str, int]
        maps words to integers
    _id2word: dict[int, str]
        maps integers to words
    Methods
    -------
     _learn_indices(self, tokens: tuple[str, ...]) -> None:
        Fills attributes mapping words and integer equivalents to each other
    encode(self, tokens: tuple[str, ...]) -> Optional[tuple[int, ...]]:
        Encodes input sequence of string tokens to sequence of integer tokens
    decode(self, encoded_tokens: tuple[int, ...]) -> Optional[tuple[str, ...]]:
        Decodes input sequence of integer tokens to sequence of string tokens
    """

    # Step 2.1
    def __init__(self) -> None:
        """
        Constructs all the necessary attributes for the text encoder object
        """
        self._word2id = {}
        self._id2word = {}

    # Step 2.2
    def _learn_indices(self, tokens: tuple[str, ...]) -> None:
        """
        Fills attributes mapping words and integer equivalents to each other
        Parameters
        ----------
            tokens : tuple[str, ...]
                sequence of string tokens
        """
        id_lst = [integer for integer in range(1000, len(tokens) + 1001)]
        self._word2id = {token: id_n for (token, id_n) in zip(tokens, id_lst)}
        self._id2word = {id_n: token for (id_n, token) in zip(id_lst, tokens)}

    # Step 2.3
    def encode(self, tokens: tuple[str, ...]) -> Optional[tuple[int, ...]]:
        """
        Encodes input sequence of string tokens to sequence of integer tokens
        Parameters
        ----------
            tokens : tuple[str, ...]
                sequence of string tokens
        Returns
        -------
            tuple[int, ...]
                sequence of integer tokens
        In case of empty tokens input data, None is returned
        """
        if not tokens:
            return None

        self._learn_indices(tokens)
        int_words = tuple((self._word2id[word] for word in tokens))
        return int_words

    # Step 2.4
    def decode(self, encoded_tokens: tuple[int, ...]) -> Optional[tuple[str, ...]]:
        """
        Decodes input sequence of integer tokens to sequence of string tokens
        Parameters
        ----------
            encoded_tokens : tuple[int, ...]
                sequence of integer tokens
        Returns
        -------
            tuple[str, ...]
                sequence of string tokens
        In case of out-of-dictionary input data, None is returned
        """
        for word in encoded_tokens:
            if word not in self._id2word.keys():
                return None

        str_words = tuple((self._id2word[id_n] for id_n in encoded_tokens))
        return str_words


# Step 3
def extract_pairs(tokens: tuple[int, ...], window_length: int) -> Optional[tuple[tuple[int, ...], ...]]:
    """
    Retrieves all pairs of co-occurring words in the token sequence
    Parameters
    ----------
        tokens : tuple[int, ...]
            sequence of tokens
        window_length: int
            maximum distance between co-occurring tokens: tokens are considered co-occurring
            if they appear in the same window of this length
    Returns
    -------
        tuple[tuple[int, ...], ...]
            pairs of co-occurring tokens
    In case of corrupt input data, None is returned:
    tokens must not be empty, window lengths must be integer, window lengths cannot be less than 2.
    """
    if not (tokens and isinstance(window_length, int) and window_length >= 2):
        return None

    window_tokens = tokens[:window_length]
    pairs = []
    start_slice = 0
    for word in tokens:
        for window_word in window_tokens:
            if word == window_word:
                continue

            pair = tuple(sorted((word, window_word)))

            if pair in pairs:
                continue

            pairs.append(pair)
        start_slice += 1
        end_slice = window_length + start_slice
        window_tokens = tokens[start_slice:end_slice]
    return tuple(pairs)


class AdjacencyMatrixGraph:
    """
    A class to represent graph as matrix of adjacency
    ...
    Attributes
    ----------
    _matrix: list[list[int]]
        stores information about vertices interrelation
    _positions: dict[int, list[int]]
        stores information about positions in text
    Methods
    -------
     get_vertices(self) -> tuple[int, ...]:
        Returns a sequence of all vertices present in the graph
     add_edge(self, vertex1: int, vertex2: int) -> int:
        Adds or overwrites an edge in the graph between the specified vertices
     is_incidental(self, vertex1: int, vertex2: int) -> int:
        Retrieves information about whether the two vertices are incidental
     calculate_inout_score(self, vertex: int) -> int:
        Retrieves a number of incidental vertices to a specified vertex
    fill_from_tokens(self, tokens: tuple[int, ...], window_length: int) -> None:
        Updates graph instance with vertices and edges extracted from tokenized text
    fill_positions(self, tokens: tuple[int, ...]) -> None:
        Saves information on all positions of each vertex in the token sequence
    calculate_position_weights(self) -> None:
        Computes position weights for all tokens in text
    get_position_weights(self) -> dict[int, float]:
        Retrieves position weights for all vertices in the graph
    """

    _matrix: list[list[int]]
    _positions: dict[int, list[int]]
    _position_weights: dict[int, float]

    # Step 4.1
    def __init__(self) -> None:
        """
        Constructs all the necessary attributes for the adjacency matrix graph object
        """
        self._matrix = []
        self._positions = {}
        self._position_weights = {}
        self._vertices = []

    # Step 4.2
    def add_edge(self, vertex1: int, vertex2: int) -> int:
        """
        Adds or overwrites an edge in the graph between the specified vertices
        Parameters
        ----------
            vertex1 : int
                the first vertex incidental to the added edge
            vertex2 : int
                the second vertex incidental to the added edge
        Returns
        -------
            int
                0 if edge was added successfully, otherwise -1
        In case of vertex1 being equal to vertex2, -1 is returned as loops are prohibited
        """
        if vertex1 == vertex2:
            return -1

        for vertex in vertex1, vertex2:
            if vertex not in self._vertices:
                self._vertices.append(vertex)
                self._matrix.append([0 for _ in range(len(self._vertices))])

        max_len_of_row = len(max(self._matrix, key=len))

        for row in self._matrix:
            if len(row) < max_len_of_row:
                row += [0] * (max_len_of_row - len(row))

        index1 = self._vertices.index(vertex1)
        index2 = self._vertices.index(vertex2)
        self._matrix[index1][index2] = 1
        self._matrix[index2][index1] = 1
        return 0

    # Step 4.3
    def is_incidental(self, vertex1: int, vertex2: int) -> int:
        """
        Retrieves information about whether the two vertices are incidental
        Parameters
        ----------
            vertex1 : int
                the first vertex incidental to the edge sought
            vertex2 : int
                the second vertex incidental to the edge sought
        Returns
        -------
            Optional[int]
                1 if vertices are incidental, otherwise 0
        If either of vertices is not present in the graph, -1 is returned
        """
        for vertex in vertex1, vertex2:
            if vertex not in self._vertices:
                return -1

        index1 = self._vertices.index(vertex1)
        index2 = self._vertices.index(vertex2)
        return self._matrix[index1][index2]

    # Step 4.4
    def get_vertices(self) -> tuple[int, ...]:
        """
        Returns a sequence of all vertices present in the graph
        Returns
        -------
            tuple[int, ...]
                a sequence of vertices present in the graph
        """
        return tuple(self._vertices)

    # Step 4.5
    def calculate_inout_score(self, vertex: int) -> int:
        """
        Retrieves a number of incidental vertices to a specified vertex
        Parameters
        ----------
            vertex : int
                a vertex to calculate inout score for
        Returns
        -------
            int
                number of incidental vertices
        If vertex is not present in the graph, -1 is returned
        """
        if vertex not in self._vertices:
            return -1

        index = self._vertices.index(vertex)
        counting = 0
        for row in self._matrix:
            if row[index] == 1:
                counting += 1

        return counting

    # Step 4.6
    def fill_from_tokens(self, tokens: tuple[int, ...], window_length: int) -> None:
        """
        Updates graph instance with vertices and edges extracted from tokenized text
        Parameters
        ----------
            tokens : tuple[int, ...]
                sequence of tokens
            window_length: int
                maximum distance between co-occurring tokens: tokens are considered co-occurring
                if they appear in the same window of this length
        """
        pairs = extract_pairs(tokens, window_length)

        for pair in pairs:
            for word in pair:
                index = pair.index(word)
                try:
                    self.add_edge(word, pair[index + 1])
                except IndexError:
                    pass

    # Step 8.2
    def fill_positions(self, tokens: tuple[int, ...]) -> None:
        """
        Saves information about all positions of each vertex in the token sequence
        ----------
            tokens : tuple[int, ...]
                sequence of tokens
        """
        for token in tokens:
            position = tokens.index(token) + 1

            if token not in self._positions.keys():
                self._positions[token] = []

            self._positions[token].append(position)

    # Step 8.3
    def calculate_position_weights(self) -> None:
        """
        Computes position weights for all tokens in text
        """
        unnormalized_positional_weight = {}

        for elements in self._positions.items():
            unnormalized = 0

            for element in elements[1]:
                summ = 1 / element
                unnormalized += summ

            unnormalized_positional_weight[elements[0]] = unnormalized

        unnormalized_values = sum(unnormalized_positional_weight.values())

        for key, weight in unnormalized_positional_weight.items():
            self._position_weights[key] = weight / unnormalized_values

    # Step 8.4
    def get_position_weights(self) -> dict[int, float]:
        """
        Retrieves position weights for all vertices in the graph
        Returns
        -------
            dict[int, float]
                position weights for all vertices in the graph
        """
        return self._position_weights


class EdgeListGraph:
    """
    A class to represent graph as a list of edges
    ...
    Attributes
    ----------
    _edges: dict[int, list[int]]
        stores information about vertices interrelation
    Methods
    -------
     get_vertices(self) -> tuple[int, ...]:
        Returns a sequence of all vertices present in the graph
     add_edge(self, vertex1: int, vertex2: int) -> int:
        Adds or overwrites an edge in the graph between the specified vertices
     is_incidental(self, vertex1: int, vertex2: int) -> int:
        Retrieves information about whether the two vertices are incidental
     calculate_inout_score(self, vertex: int) -> int:
        Retrieves a number of incidental vertices to a specified vertex
    fill_from_tokens(self, tokens: tuple[int, ...], window_length: int) -> None:
        Updates graph instance with vertices and edges extracted from tokenized text
    fill_positions(self, tokens: tuple[int, ...]) -> None:
        Saves information on all positions of each vertex in the token sequence
    calculate_position_weights(self) -> None:
        Computes position weights for all tokens in text
    get_position_weights(self) -> dict[int, float]:
        Retrieves position weights for all vertices in the graph
    """

    # Step 7.1
    def __init__(self) -> None:
        """
        Constructs all the necessary attributes for the edge list graph object
        """
        self._edges = {}
        self._positions = {}
        self._position_weights = {}

    # Step 7.2
    def get_vertices(self) -> tuple[int, ...]:
        """
        Returns a sequence of all vertices present in the graph
        Returns
        -------
            tuple[int, ...]
                a sequence of vertices present in the graph
        """
        return tuple(self._edges.keys())

    # Step 7.2
    def add_edge(self, vertex1: int, vertex2: int) -> int:
        """
        Adds or overwrites an edge in the graph between the specified vertices
        Parameters
        ----------
            vertex1 : int
                the first vertex incidental to the added edge
            vertex2 : int
                the second vertex incidental to the added edge
        Returns
        -------
            int
                0 if edge was added successfully, otherwise -1
        In case of vertex1 being equal to vertex2, -1 is returned as loops are prohibited
        """
        if vertex1 == vertex2:
            return -1

        for vertex in vertex1, vertex2:
            if vertex not in self._edges.keys():
                self._edges[vertex] = []

        self._edges[vertex1].append(vertex2)
        self._edges[vertex2].append(vertex1)
        return 0

    # Step 7.2
    def is_incidental(self, vertex1: int, vertex2: int) -> int:
        """
        Retrieves information about whether the two vertices are incidental
        Parameters
        ----------
            vertex1 : int
                the first vertex incidental to the edge sought
            vertex2 : int
                the second vertex incidental to the edge sought
        Returns
        -------
            Optional[int]
                1 if vertices are incidental, otherwise 0
        If either of vertices is not present in the graph, -1 is returned
        """
        for vertex in vertex1, vertex2:
            if vertex not in self._edges.keys():
                return -1

        if vertex2 in self._edges[vertex1]:
            return 1
        else:
            return 0

    # Step 7.2
    def calculate_inout_score(self, vertex: int) -> int:
        """
        Retrieves a number of incidental vertices to a specified vertex
        Parameters
        ----------
            vertex : int
                a vertex to calculate inout score for
        Returns
        -------
            int
                number of incidental vertices
        If vertex is not present in the graph, -1 is returned
        """
        if vertex not in self._edges:
            return -1

        return len(self._edges[vertex])

    # Step 7.2
    def fill_from_tokens(self, tokens: tuple[int, ...], window_length: int) -> None:
        """
        Updates graph instance with vertices and edges extracted from tokenized text
        Parameters
        ----------
            tokens : tuple[int, ...]
                sequence of tokens
            window_length: int
                maximum distance between co-occurring tokens: tokens are considered co-occurring
                if they appear in the same window of this length
        """
        pairs = extract_pairs(tokens, window_length)

        for pair in pairs:
            for word in pair:
                index = pair.index(word)
                try:
                    self.add_edge(word, pair[index + 1])
                except IndexError:
                    pass

    # Step 8.2
    def fill_positions(self, tokens: tuple[int, ...]) -> None:
        """
        Saves information on all positions of each vertex in the token sequence
        ----------
            tokens : tuple[int, ...]
                sequence of tokens
        """
        for token in tokens:
            position = tokens.index(token) + 1

            if token not in self._positions.keys():
                self._positions[token] = []

            self._positions[token].append(position)

    # Step 8.3
    def calculate_position_weights(self) -> None:
        """
        Computes position weights for all tokens in text
        """
        unnormalized_positional_weight = {}

        for elements in self._positions.items():
            unnormalized = 0

            for element in elements[1]:
                summ = 1 / element
                unnormalized += summ

            unnormalized_positional_weight[elements[0]] = unnormalized

        unnormalized_values = sum(unnormalized_positional_weight.values())

        for key, weight in unnormalized_positional_weight.items():
            self._position_weights[key] = weight / unnormalized_values

    # Step 8.4
    def get_position_weights(self) -> dict[int, float]:
        """
        Retrieves position weights for all vertices in the graph
        Returns
        -------
            dict[int, float]
                position weights for all vertices in the graph
        """
        return self._position_weights


class VanillaTextRank:
    """
    Basic TextRank implementation
    ...
    Attributes
    ----------
    _graph: Union[AdjacencyMatrixGraph, EdgeListGraph]
        a graph representing the text
    _damping_factor: float
         probability of jumping from a given vertex to another random vertex
         in the graph during vertices scores calculation
    _convergence_threshold: float
        maximal acceptable difference between the vertices scores in two consequent iteration
    _max_iter: int
        maximal number of iterations to perform
    _scores: dict[int, float]
        scores of significance for all vertices present in the graph
    Methods
    -------
     update_vertex_score(self, vertex: int, incidental_vertices: list[int], scores: dict[int, float]) -> None:
        Changes vertex significance score using algorithm-specific formula
     score_vertices(self) -> dict[int, float]:
        Iteratively computes significance scores for vertices
     get_scores(self) -> dict[int, float]:
        Retrieves importance scores of all tokens in the encoded text
     get_top_keywords(self, n_keywords: int) -> tuple[int, ...]:
        Retrieves top n most important tokens in the encoded text
     """

    _scores: dict[int, float]

    # Step 5.1
    def __init__(self, graph: Union[AdjacencyMatrixGraph, EdgeListGraph]) -> None:
        """
        Constructs all the necessary attributes for the text rank algorithm implementation
        Parameters
        ----------
        graph: Union[AdjacencyMatrixGraph, EdgeListGraph]
            a graph representing the text
        """
        self._graph = graph
        self._damping_factor = 0.85
        self._convergence_threshold = 0.0001
        self._max_iter = 50
        self._scores = {}

    # Step 5.2
    def update_vertex_score(self, vertex: int, incidental_vertices: list[int], scores: dict[int, float]) -> None:
        """
        Changes vertex significance score using algorithm-specific formula
        Parameters
        ----------
            vertex : int
                a vertex which significance score is updated
            incidental_vertices: list[int]
                vertices incidental to the scored one
            scores: dict[int, float]
                scores of all vertices in the graph
        """
        summ = 0

        for incidental_vertex in incidental_vertices:
            in_out_score = self._graph.calculate_inout_score(incidental_vertex)
            var = 1 / abs(in_out_score) * scores[vertex]
            summ += var

        new_weight = summ * self._damping_factor + (1 - self._damping_factor)
        self._scores[vertex] = new_weight

    # Step 5.3
    def train(self) -> None:
        """
        Iteratively computes significance scores for vertices
        Returns
        -------
            dict[int, float]:
                scores for all vertices present in the graph
        """
        vertices = self._graph.get_vertices()
        for vertex in vertices:
            self._scores[vertex] = 1.0

        for _ in range(0, self._max_iter):
            prev_score = self._scores.copy()
            for scored_vertex in vertices:
                incidental_vertices = [vertex for vertex in vertices
                                       if self._graph.is_incidental(scored_vertex, vertex) == 1]
                self.update_vertex_score(scored_vertex, incidental_vertices, prev_score)
            abs_score_diff = [abs(i - j) for i, j in zip(prev_score.values(), self._scores.values())]
            if sum(abs_score_diff) <= self._convergence_threshold:
                break

    # Step 5.4
    def get_scores(self) -> dict[int, float]:
        """
        Retrieves importance scores of all tokens in the encoded text
        Returns
        -------
            dict[int, float]
                importance scores of all tokens in the encoded text
        """
        return self._scores

    # Step 5.5
    def get_top_keywords(self, n_keywords: int) -> tuple[int, ...]:
        """
        Retrieves top n most important tokens in the encoded text
        Returns
        -------
            tuple[int, ...]
                top n most important tokens in the encoded text
        """
        return tuple(sorted(self._scores, reverse=True, key=lambda key: self._scores[key])[:n_keywords])


class PositionBiasedTextRank(VanillaTextRank):
    """
    Advanced TextRank implementation: positions of tokens in text are taken into consideration
    ...
    Attributes
    ----------
    _graph: Union[AdjacencyMatrixGraph, EdgeListGraph]
        a graph representing the text
    _damping_factor: float
         probability of jumping from a given vertex to another random vertex
         in the graph during vertices scores calculation
    _convergence_threshold: float
        maximal acceptable difference between the vertices scores in two consequent iteration
    _max_iter: int
        maximal number of iterations to perform
    _scores: dict[int, float]
        scores of significance for all vertices present in the graph
    _position_weights: dict[int, float]
        position weights for all tokens in the text
    Methods
    -------
     update_vertex_score(self, vertex: int, incidental_vertices: list[int], scores: dict[int, float]) -> None:
        Changes vertex significance score using algorithm-specific formula
     score_vertices(self) -> dict[int, float]:
        Iteratively computes significance scores for vertices
     get_scores(self) -> dict[int, float]:
        Retrieves importance scores of all tokens in the encoded text
     get_top_keywords(self, n_keywords: int) -> tuple[int, ...]:
        Retrieves top n most important tokens in the encoded text
    """

    # Step 9.1
    def __init__(self, graph: Union[AdjacencyMatrixGraph, EdgeListGraph]) -> None:
        """
        Constructs all the necessary attributes
        for the position-aware text rank algorithm implementation
        Attributes
        ----------
        graph: Union[AdjacencyMatrixGraph, EdgeListGraph]
            a graph representing the text
        """
        super().__init__(graph)
        self._position_weights = graph.get_position_weights()

    # Step 9.2
    def update_vertex_score(self, vertex: int, incidental_vertices: list[int], scores: dict[int, float]) -> None:
        """
        Changes vertex significance score using algorithm-specific formula
        Parameters
        ----------
            vertex : int
                a vertex which significance score is updated
            incidental_vertices: list[int]
                vertices incidental to the scored one
            scores: dict[int, float]
                scores of all vertices in the graph
        """
        summ = 0

        for incidental_vertex in incidental_vertices:
            in_out_score = self._graph.calculate_inout_score(incidental_vertex)
            var = 1 / abs(in_out_score) * scores[vertex]
            summ += var

        new_weight = summ * self._damping_factor + (1 - self._damping_factor) * self._position_weights[vertex]
        self._scores[vertex] = new_weight


class TFIDFAdapter:
    """
    A class to unify the interface of TF-IDF keywords extractor with TextRank algorithms
    ...
    Attributes
    ----------
    _tokens: tuple[str, ...]
        sequence of tokens from which to extract keywords
    _idf: dict[str, float]
         Inverse Document Frequency scores for tokens
    _scores: dict[str, float]
        TF-IDF scores reflecting how important each token is
    Methods
    -------
     train(self) -> int:
        Computes importance scores for all tokens
     get_top_keywords(self, n_keywords: int) -> tuple[str, ...]:
        Retrieves a requested number of the most important tokens
    """

    _scores: dict[str, float]

    # Step 10.1
    def __init__(self, tokens: tuple[str, ...], idf: dict[str, float]) -> None:
        """
        Constructs all the necessary attributes
        for the TF-IDF keywords extractor
        Parameters
        ----------
            tokens: tuple[str, ...]
                sequence of tokens from which to extract keywords
            idf: dict[str, float]
                Inverse Document Frequency scores for tokens
        """
        pass

    # Step 10.2
    def train(self) -> int:
        """
        Computes importance scores for all tokens
        Returns
        -------
            int:
                0 if importance scores were calculated successfully, otherwise -1
        """
        pass

    # Step 10.3
    def get_top_keywords(self, n_keywords: int) -> tuple[str, ...]:
        """
        Retrieves a requested number of the most important tokens
        Parameters
        ----------
            n_keywords: int
                requested number of keywords to extract
        Returns
        -------
            tuple[str, ...]:
                a requested number tokens with the highest importance scores
        """
        pass


class RAKEAdapter:
    """
    A class to unify the interface of RAKE keywords extractor with TextRank algorithms
    ...
    Attributes
    ----------
    _text: str
        a text from which to extract keywords
    _stop_words: tuple[str, ...]
         a sequence of stop-words
    _scores: dict[str, float]
        word scores reflecting how important each token is
    Methods
    -------
     train(self) -> int:
        Computes importance scores for all tokens
     get_top_keywords(self, n_keywords: int) -> tuple[str, ...]:
        Retrieves a requested number of the most important tokens
    """

    _scores: dict[str, float]

    # Step 11.1
    def __init__(self, text: str, stop_words: tuple[str, ...]) -> None:
        """
        Constructs all the necessary attributes
        for the RAKE keywords extractor
        Parameters
        ----------
            text: str
                a text from which to extract keywords
            stop_words: tuple[str, ...]
                a sequence of stop-words
        """
        pass

    # Step 11.2
    def train(self) -> int:
        """
        Computes importance scores for all tokens
        Returns
        -------
            int:
                0 if importance scores were calculated successfully, otherwise -1
        """
        pass

    # Step 11.3
    def get_top_keywords(self, n_keywords: int) -> tuple[str, ...]:
        """
        Retrieves a requested number of the most important tokens
        Parameters
        ----------
            n_keywords: int
                requested number of keywords to extract
        Returns
        -------
            tuple[str, ...]:
                a requested number tokens with the highest importance scores
        """
        pass


# Step 12.1
def calculate_recall(predicted: tuple[str, ...], target: tuple[str, ...]) -> float:
    """
    Computes recall metric
    Parameters
    ----------
        predicted: tuple[str, ...]
            keywords predictions of an algorithm to estimate
        target: tuple[str, ...]
            ground truth keywords
    Returns
    -------
        float:
            recall value
    """
    pass


class KeywordExtractionBenchmark:
    """
    A class to compare 4 different algorithms of keywords extraction
    ...
    Attributes
    ----------
    _stop_words: tuple[str, ...]
        a sequence of stop-words
    _punctuation: tuple[str, ...]
        symbols of punctuation
    _idf: dict[str, float]
        Inverse Document Frequency scores for the words in materials
    _materials_path: Path
        a path to materials to use for comparison
    themes: tuple[str, ...]
        a sequence of topics to which comparison materials relate
    report: dict[str, dict[str, float]]
        comparison report reflecting how successfully each model extracts keywords
    Methods
    -------
    run(self) -> Optional[dict[str, dict[str, float]]]:
        creates comparison report
    save_to_csv(self, path: Path) -> None:
        saves the report in the .csv format
    """

    # Step 12.2
    def __init__(self, stop_words: tuple[str, ...], punctuation: tuple[str, ...],
                 idf: dict[str, float], materials_path: Path) -> None:
        """
        Constructs all the necessary attributes for the Benchmark instance
        Parameters
        ----------
            stop_words: tuple[str, ...]
                a sequence of stop-words
            punctuation: tuple[str, ...]
                symbols of punctuation
            idf: dict[str, float]
                Inverse Document Frequency scores for the words in materials
            materials_path: Path
                a path to materials to use for comparison
        """
        pass

    # Step 12.3
    def run(self) -> Optional[dict[str, dict[str, float]]]:
        """
        Creates comparison report
        Returns
        -------
            Optional[dict[str, dict[str, float]]]:
                comparison report
        In case it is impossible to extract keywords due to corrupt inputs, None is returned
        """
        pass

    # Step 12.4
    def save_to_csv(self, path: Path) -> None:
        """
        Saves comparison report to csv
        Parameters
        ----------
            path: Path
                a path where to save the report file
        """
        pass
