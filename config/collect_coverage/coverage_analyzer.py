"""
Runner for collecting coverage
"""

import json
import sys
from pathlib import Path
from typing import Optional, Iterable, Mapping

from config.collect_coverage.run_coverage import run_coverage_collection, CoverageRunError, \
    CoverageCreateReportError

CoverageResults = Mapping[str, Optional[int]]


def collect_all_labs_names(start_path: Path) -> Iterable[Path]:
    labs_enumerated_path = start_path / 'config' / 'labs.txt'
    with labs_enumerated_path.open(encoding='utf-8') as f:
        return map(lambda x: start_path / x.strip(), f.readlines())


def collect_coverage(all_labs_names: Iterable[Path], artifacts_path: Path) -> CoverageResults:
    all_labs_results = {}
    for lab_path in all_labs_names:
        percentage = None
        try:
            percentage = run_coverage_collection(lab_path=lab_path, artifacts_path=artifacts_path)
        except (CoverageRunError, CoverageCreateReportError) as e:
            print(e)
        finally:
            all_labs_results[lab_path.name] = percentage
    return all_labs_results


def is_decrease_present(all_labs_results: CoverageResults, previous_coverage_results_path: Path) -> bool:
    with previous_coverage_results_path.open(encoding='utf-8') as f:
        previous_coverage_results = json.load(f)

    print('\n\n' + '------' * 3)
    print('REPORT')
    print('------' * 3)
    any_degradation = False
    for lab_name, current_lab_percentage in all_labs_results.items():
        prev_lab_percentage = previous_coverage_results.get(lab_name, 0)
        if current_lab_percentage is None:
            current_lab_percentage = 0
        diff = current_lab_percentage - prev_lab_percentage

        if diff < 0:
            any_degradation = True
        print(f'{lab_name:<30}: {current_lab_percentage}% ({"+" if diff >= 0 else ""}{diff})')

    print('\n\n' + '------' * 3)
    print('END OF REPORT')
    print('------' * 3 + '\n\n')

    return any_degradation


def main() -> None:
    project_root = Path(__file__).parent.parent.parent
    artifacts_path = project_root / 'build' / 'coverage'
    artifacts_path.mkdir(parents=True, exist_ok=True)

    previous_coverage_results_path = project_root / 'config' / 'labs_coverage_thresholds.json'

    all_labs_names = collect_all_labs_names(start_path=project_root)

    all_labs_results = collect_coverage(all_labs_names, artifacts_path)

    any_degradation = is_decrease_present(all_labs_results, previous_coverage_results_path)

    if any_degradation:
        print('Some of labs have worse coverage. We cannot accept this. Write more tests!')
        sys.exit(1)

    print('Nice coverage. Anyway, write more tests!', end='\n\n')

    print('You can copy-paste the following content to the ./config/labs_coverage_thresholds.json '
          'to update thresholds. \n\n')

    print(json.dumps(all_labs_results, indent=4, sort_keys=True))


if __name__ == '__main__':
    main()
