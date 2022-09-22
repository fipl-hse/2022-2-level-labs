"""
Runner for collecting coverage
"""

import json
import platform
import subprocess
from pathlib import Path
from typing import Any


class CoverageRunError(Exception):
    pass


class CoverageCreateReportError(Exception):
    pass


def get_target_score(lab_path: Path) -> int:
    target_score_file_path = lab_path.joinpath('target_score.txt')
    with open(target_score_file_path, 'r', encoding='utf-8') as fd:
        return int(fd.readline())


def _run_console_tool(exe: str, /, *args: str, **kwargs: Any) -> subprocess.CompletedProcess:
    kwargs_processed: list[str] = []
    for item in kwargs.items():
        if item[0] in ('env', 'debug', 'cwd'):
            continue
        kwargs_processed.extend(map(str, item))

    options = [
        str(exe),
        *args,
        *kwargs_processed
    ]

    if kwargs.get('debug', False):
        print(f'Attempting to run with the following arguments: {" ".join(options)}')

    env = kwargs.get('env')
    if env:
        # pylint:disable = subprocess-run-check
        return subprocess.run(options, capture_output=True, env=env)
    if kwargs.get('cwd'):
        # pylint:disable = subprocess-run-check
        return subprocess.run(options, capture_output=True, cwd=kwargs.get('cwd'))
    # pylint:disable = subprocess-run-check
    return subprocess.run(options, capture_output=True)


def extract_percentage_from_report(report_path: Path) -> int:
    with report_path.open(encoding='utf-8') as f:
        content = json.load(f)
    return int(content['totals']['percent_covered_display'])


def choose_python_exe() -> Path:
    lab_path = Path(__file__).parent.parent.parent
    if platform.system() == 'Windows':
        python_exe_path = lab_path / 'venv' / 'Scripts' / 'python.exe'
    else:
        python_exe_path = lab_path / 'venv' / 'bin' / 'python'
    return python_exe_path


def run_coverage_collection(lab_path: Path, artifacts_path: Path) -> int:
    print(f'Processing {lab_path} ...')

    python_exe_path = choose_python_exe()
    target_score = get_target_score(lab_path)

    res_process = _run_console_tool(str(python_exe_path), '-m', 'coverage',
                                    'run', '--include', f'{lab_path.name}/*.py', '-m', 'pytest', '-m',
                                    f'{lab_path.name} and mark{target_score}',
                                    debug=False,
                                    cwd=str(lab_path.parent))
    if res_process.returncode != 0:
        raise CoverageRunError(res_process.stderr.decode('utf-8'))

    report_path = artifacts_path / f'{lab_path.name}.json'
    res_process = _run_console_tool(str(python_exe_path), '-m', 'coverage', 'json', '-o', str(report_path),
                                    debug=False,
                                    cwd=str(lab_path.parent))
    if res_process.returncode != 0:
        raise CoverageCreateReportError(res_process.stderr.decode('utf-8'))

    return extract_percentage_from_report(report_path)
