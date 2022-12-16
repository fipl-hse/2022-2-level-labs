"""
TextRank keyword extraction starter
"""

from pathlib import Path

from lab_3_keywords_textrank.main import extract_pairs, TextPreprocessor, TextEncoder, AdjacencyMatrixGraph, \
    VanillaTextRank, EdgeListGraph,  PositionBiasedTextRank, KeywordExtractionBenchmark


if __name__ == "__main__":

    # finding paths to the necessary utils
    project_root = Path(__file__).parent
    asserts_path = project_root / 'assets'

    # reading the text from which keywords are going to be extracted
    target_text_path = asserts_path / 'article.txt'
    with open(target_text_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # reading list of stop words
    stop_words_path = asserts_path / 'stop_words.txt'
    with open(stop_words_path, 'r', encoding='utf-8') as file:
        stop_words = tuple(file.read().split('\n'))

    preprocessor = TextPreprocessor(stop_words, tuple('.,!?-:;()'))
    tokens = preprocessor.preprocess_text(text)

    encoder = TextEncoder()
    encoded_tokens = encoder.encode(tokens)

    # step 3
    preprocessor = TextPreprocessor(stop_words, tuple('.,!?-:;()'))
    preprocessed_text = preprocessor.preprocess_text(text)
    encoder = TextEncoder()
    tokens = encoder.encode(preprocessed_text)
    if tokens:
        print(extract_pairs(tokens, 3))

    # steps 6 and 7
    adjacency_matrix_graph = AdjacencyMatrixGraph()
    edge_list_graph = EdgeListGraph()
    if tokens:
        for graph in adjacency_matrix_graph, edge_list_graph:
            adjacency_matrix_graph.fill_from_tokens(tokens, 3)
            adjacency_matrix_graph.fill_positions(tokens)
            adjacency_matrix_graph.calculate_position_weights()
    vanilla_rank_adjacency = VanillaTextRank(adjacency_matrix_graph)
    vanilla_rank_edge = VanillaTextRank(edge_list_graph)
    for vanilla_rank in vanilla_rank_adjacency, vanilla_rank_edge:
        vanilla_rank.train()
        print(encoder.decode(vanilla_rank.get_top_keywords(10)))

    # step 9
    position_rank_adjacency = PositionBiasedTextRank(adjacency_matrix_graph)
    position_rank_edge = PositionBiasedTextRank(edge_list_graph)
    for position_rank in position_rank_adjacency, position_rank_edge:
        position_rank.train()
    get_top_keywords = encoder.decode(position_rank.get_top_keywords(10))
    print(get_top_keywords)

    RESULT = get_top_keywords

    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
