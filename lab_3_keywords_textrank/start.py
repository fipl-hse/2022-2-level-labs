"""
TextRank keyword extraction starter
"""

from pathlib import Path
from lab_3_keywords_textrank.main import(
    TextPreprocessor,
    TextEncoder,
    extract_pairs,
    AdjacencyMatrixGraph,
    # VanillaTextRank,
    # EdgeListGraph,
    # PositionBiasedTextRank,
    # KeywordExtractionBenchmark
)



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

    preprocessor = TextPreprocessor(stop_words, tuple('.,!?-:;()'))
    clean_tokens = preprocessor.preprocess_text(text)
    print(clean_tokens)

    dcts = TextEncoder()
    encoded_tokens = dcts.encode(clean_tokens)
    print(encoded_tokens)
    decoded_tokens = dcts.decode(encoded_tokens)
    print(decoded_tokens)

    pairs = extract_pairs(encoded_tokens, 3)
    print(pairs)

    matrix = AdjacencyMatrixGraph()
    for pair in pairs:
        matrix.add_edge(*pair)
        matrix.is_incidental(*pair)
        matrix.get_vertices()
        matrix.calculate_inout_score(pair[0])
        matrix.calculate_inout_score(pair[1])

    matrix.fill_from_tokens(encoded_tokens, 3)
    print(matrix._matrix)

    RESULT = None
    # DO NOT REMOVE NEXT LINE - KEEP IT INTENTIONALLY LAST
    assert RESULT, 'Keywords are not extracted'
