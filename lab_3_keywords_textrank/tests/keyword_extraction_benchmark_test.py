"""
Checks the third lab KeywordExtractionBenchmark
"""

import csv
import json
import unittest
from pathlib import Path
from string import punctuation
from unittest import mock

import pytest

from lab_3_keywords_textrank.main import KeywordExtractionBenchmark, TextEncoder


class KeywordExtractionBenchmarkTest(unittest.TestCase):
    """
    Tests KeywordExtractionBenchmark
    """

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark10
    def test_benchmark(self):
        stop_words = ('to', 'a', 'is', 'was')

        project_root = Path(__file__).parent.parent
        assets_path = project_root / 'assets'
        benchmark_material_path = assets_path / 'benchmark_materials'
        idf_path = benchmark_material_path / 'IDF.json'
        with open(idf_path, 'r', encoding='utf-8') as file:
            idf = json.load(file)

        benchmark = KeywordExtractionBenchmark(stop_words=stop_words,
                                               punctuation=tuple(punctuation),
                                               idf=idf,
                                               materials_path=benchmark_material_path)
        benchmark.themes = ('culture', 'business', 'crime')
        report = benchmark.run()

        expected_algorithms = tuple(map(str.lower, ('TF-IDF', 'RAKE', 'VanillaTextRank', 'PositionBiasedTextRank')))
        algorithms = tuple(map(str.lower, report.keys()))
        self.assertCountEqual(expected_algorithms, algorithms)

        expected_themes = ('culture', 'business', 'crime')
        for scores in report.values():
            self.assertCountEqual(expected_themes, tuple(scores.keys()))
            for score in scores.values():
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)

        csv_report_path = project_root.joinpath('report.csv')

        csv_report_path.unlink(missing_ok=True)

        benchmark.save_to_csv(csv_report_path)

        with open(csv_report_path, 'r', encoding='utf-8') as benchmark_report_file:
            reader = csv.DictReader(benchmark_report_file)

            expected_columns = ('name', *expected_themes)

            for row in reader:
                for column_name, column_value in row.items():
                    self.assertIn(column_name, expected_columns)
                    if column_name == 'name':
                        self.assertIn(column_value.lower(), expected_algorithms)
                    else:
                        self.assertGreaterEqual(float(column_value), 0.0)

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark10
    @mock.patch.object(TextEncoder, 'encode', return_value=None)
    def test_emtpy_output(self, mock_encoder):
        stop_words = ('to', 'a', 'is', 'was')

        project_root = Path(__file__).parent.parent
        assets_path = project_root / 'assets'
        benchmark_material_path = assets_path / 'benchmark_materials'
        idf_path = benchmark_material_path / 'IDF.json'
        with open(idf_path, 'r', encoding='utf-8') as file:
            idf = json.load(file)

        benchmark = KeywordExtractionBenchmark(stop_words=stop_words,
                                               punctuation=tuple(punctuation),
                                               idf=idf,
                                               materials_path=benchmark_material_path)
        benchmark.themes = ('culture', 'business', 'crime')
        self.assertEqual(benchmark.run(), None)
