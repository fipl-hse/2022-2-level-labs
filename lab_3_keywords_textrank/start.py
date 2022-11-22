"""
TextRank keyword extraction starter
"""
import string
from pathlib import Path
from lab_3_keywords_textrank.main import TextPreprocessor, TextEncoder, extract_pairs, AdjacencyMatrixGraph, \
    VanillaTextRank, EdgeListGraph, PositionBiasedTextRank

if __name__ == "__main__":
    # finding paths to the necessary utils
    PROJECT_ROOT = Path(__file__).parent
    ASSETS_PATH = PROJECT_ROOT / 'assets'

    # reading the text from which keywords are going to be extracted
    TARGET_TEXT_PATH = ASSETS_PATH / 'article.txt'
    with open(TARGET_TEXT_PATH, 'r', encoding='utf-8') as file:
        text = file.read()

    # reading list of stop words
    STOP_WORDS_PATH = ASSETS_PATH / 'stop_words.txt'
    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
        stop_words = tuple(file.read().split('\n'))

    keywords = None
    punctuation = tuple(string.punctuation + "–—¡¿⟨⟩«»…⋯‹›“”")

    preprocessor = TextPreprocessor(stop_words, punctuation)
    tokens = preprocessor.preprocess_text(text)

    encoder = TextEncoder()
    encoded_tokens = encoder.encode(tokens)

    if encoded_tokens:
        graph_matrix = AdjacencyMatrixGraph()
        graph_edges = EdgeListGraph()

        for graph in (graph_matrix, graph_edges):
            graph.fill_from_tokens(encoded_tokens, 3)
            graph.fill_positions(encoded_tokens)
            graph.calculate_position_weights()

        vanilla_text_ranker_matrix = VanillaTextRank(graph_matrix)
        vanilla_text_ranker_edges = VanillaTextRank(graph_edges)
        position_biased_text_ranker_matrix = PositionBiasedTextRank(graph_matrix)
        position_biased_text_ranker_edges = PositionBiasedTextRank(graph_edges)

        for ranker in (vanilla_text_ranker_edges, vanilla_text_ranker_matrix,
                       position_biased_text_ranker_edges, position_biased_text_ranker_matrix):
            ranker.train()
            encoded_keywords = ranker.get_top_keywords(10)

            keywords = encoder.decode(encoded_keywords)
            print(ranker.__class__.__name__, graph.__class__.__name__, keywords)

    RESULT = keywords
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
