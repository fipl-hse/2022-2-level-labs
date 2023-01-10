"""
Co-occurrence-driven keyword extraction starter
"""

from pathlib import Path
from lab_2_keywords_cooccurrence.main import (process_text, get_top_n, load_stop_words)


def read_target_text(file_path: Path) -> str:
    """
    Utility functions that reads the text content from the file
    """
    with open(file_path, 'r', encoding='utf-8') as target_text_file:
        return target_text_file.read()


if __name__ == "__main__":
    root = Path(__file__).parent
    path = root / 'assets'

    stop_words_path = path / 'stop_words.txt'
    with open(stop_words_path, 'r', encoding='utf-8') as fd:
        stop_words = fd.read().split('\n')


        albatross = assets / 'albatross.txt'
        gagarin = assets / 'gagarin.txt'
        genome_engineering = assets / 'genome_engineering.txt'
        pain_detection = assets / 'pain_detection.txt'


        corpus = {
            'gagarin': read_target_text(gagarin),
            'albatross': read_target_text(albatross),
            'genome_engineering': read_target_text(genome_engineering),
            'pain_detection': read_target_text(pain_detection)
        }

        Gagarin = process_text(corpus['gagarin'], stop_words)
        if Gagarin:
            print(get_top_n(Gagarin, 10, 5))
        Albatross = process_text(corpus['albatross'], stop_words)
        if Albatross:
            print(get_top_n(Albatross, 10, 5))
        Genome = process_text(corpus['genome_engineering'], stop_words)
        if Genome:
            print(get_top_n(Genome, 10, 5))
        Pain = process_text(corpus['pain_detection'], stop_words)
        if Pain:
            print(get_top_n(Pain, 10, 5))

        stop_words = load_stop_words(assets / 'stopwords.json')

        Polish = None
        if stop_words:
            Polish = process_text(read_target_text(assets / 'polish.txt'), stop_words['pl'])
        if Polish:
            print(get_top_n(Polish, 10, 5))

        Unknown = process_text(read_target_text(assets / 'unknown.txt'), max_length=8)
        if Unknown:
            print(get_top_n(Unknown, 10, 5))

    RESULT = Unknown

    assert RESULT
