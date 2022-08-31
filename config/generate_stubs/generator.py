"""
Generator of stubs for existing lab implementation
"""

import ast
import shutil
from pathlib import Path
from typing import Optional

from tap import Tap


class NoDocStringForAMethodError(Exception):
    pass


def remove_implementation_from_function(original_declaration: ast.stmt, parent: Optional[ast.ClassDef] = None) -> None:
    if not isinstance(original_declaration, ast.FunctionDef):
        return
    # print(ast.dump(ast.parse("raise NotImplementedError()"), annotate_fields=False, indent=4))
    expr = original_declaration.body[0]
    if not isinstance(expr, ast.Expr) and \
            (
                    not hasattr(expr, 'value') or
                    not isinstance(getattr(expr, 'value'), ast.Constant)
            ):
        raise NoDocStringForAMethodError(
            'You have to provide docstring for a method %s%s' %
            (f'{parent.name + "." if parent is not None else ""}', f'{original_declaration.name}'))
    original_declaration.body[1:] = [ast.Pass()]


def cleanup_code(source_code_path: Path) -> str:
    with source_code_path.open(encoding='utf-8') as f:
        data = ast.parse(f.read(), source_code_path.name)

    accepted_modules = ['typing']
    new_decl = []
    for decl in data.body:
        if isinstance(decl, (ast.Import, ast.ImportFrom)):
            if (module_name := getattr(decl, 'module', None)) is None:
                module_name = decl.names[0].name

            if module_name not in accepted_modules:
                continue
        if isinstance(decl, ast.ClassDef):
            for class_decl in decl.body:
                remove_implementation_from_function(class_decl, parent=decl)
        remove_implementation_from_function(decl)
        new_decl.append(decl)

    data.body = list(new_decl)
    return ast.unparse(data)


class ArgumentParser(Tap):
    source_code_path: str
    target_code_path: str


def main() -> None:
    args = ArgumentParser().parse_args()

    res_stub_path = Path(args.target_code_path)
    shutil.rmtree(res_stub_path.parent, ignore_errors=True)
    res_stub_path.parent.mkdir(parents=True)

    source_code = cleanup_code(Path(args.source_code_path))

    with res_stub_path.open(mode='w', encoding='utf-8') as f:
        f.write(source_code)


if __name__ == '__main__':
    main()
