from typing import Callable

def g() -> None:
    d = {}
    def f(a, b):
        if (a,b) not in d:
            print(f'Computing for {(a,b)} ...')
            d[(a,b)] = a + b
        print(f'Retrieving results from internal cache...')
        return d[(a,b)]
    return f


def cached(fn: Callable) -> None:
    d = dict()
    def internal(*args):
        if args not in d:
            print(f'Computing for {args} ...')
            d[args] = sum(args)
        print(f'Retrieving results from internal cache...')
        return d[args]
    return internal

@cached
def f(a: int, b: int) -> int:
    return a + b


def main() -> None:
    print(f'######### Closure-based calls')
    c = g()
    res = c(10, 20)
    res = c(10, 20)
    res = c(10, 20)
    res = c(10, 20)
    res = c(10, 20)
    res = c(10, 20)
    print(f'Result is {res}')

    print(f'\n\n\n######### Decorator-based calls')
    res = f(10, 20)
    res = f(10, 20)
    res = f(10, 20)
    res = f(10, 20)
    res = f(10, 20)
    res = f(10, 20)
    print(f'Result is {res}')

if __name__ == '__main__':
    main()
