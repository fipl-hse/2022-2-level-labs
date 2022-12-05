#                                         BaseException
#             ^                                       ^                       ^
#             |                                       |                       |
#             Exception                       KeyboardInterrupt       SystemExit
#     ^       ^                   ^
#     |       |                   |
# ValueError  ZeroDivisionError CustomError

# EAFP - easier to ask for forgiveness than for permission - try/except
# LBYL - look before you leap - if/else

#                             EAFP              LBYL
# performance                  + (-)              +
# readability                  -                  +
# race conditions              +                  -
# number of checks (few/many)  +/-                -/+

def compare_lbyl_vs_eafp():
    # LBYL style
    b = []
    if 100 < len(b) - 1:
        print(b[100])
    else:
        print('No')

    # EAFP
    try:
        a = {}
        b = []
        print(b[12])
        print(a['key'])
    except (KeyError, IndexError) as my_error:
        print('Error!!')
    except IndexError:
        print('IndexError!!')
    except:  # ~ except BaseException
        print('General error!!')
    else:
        print('iff no exception')
    finally:
        print('Always!!')


def check_exception_raise():
    # 0: __main__
    # 1: main()->__main__
    # 2: g()->main()->__main__
    # 2: f()->g()->main()->__main__
    # 3: g()->main()->__main__
    # 4: main()->__main__
    # 5: __main__

    # 0:
    # __main__
    #   -> main()
    #         -> g()
    #               -> f() -> ZeroDivisionError

    # 1:
    # __main__
    #   -> main()
    #         -> g() -> ZeroDivisionError
    #               -> f()

    # 2:
    # __main__
    #   -> main() -> ZeroDivisionError
    #         -> g()
    #               -> f()

    # 3:
    # __main__ -> ZeroDivisionError
    #   -> main()
    #         -> g()
    #               -> f()

    # 4:
    # -> ZeroDivisionError -> exit!!!
    # __main__
    #   -> main()
    #         -> g()
    #               -> f()

    # Nested functions only to separate topics from lecture.
    # Do not nest functions in your code!
    def f(a, b):
        return a / b

    def g(a, b):
        c = f(a, b)
        print(c)
        return c

    try:
        c = g(1, 0)
    except ZeroDivisionError:
        print('Error')
    else:
        print(c)


def propagate_error_without_exceptions():
    def f(a, b):
        if b == 0:
            return -1
        return a / b

    def g(a, b):
        c = f(a, b)
        if c == -1:
            return None
        print(c)
        return c

    d = g(1, 0)
    if d is not None:
        print(f'Success: {d}')


def main():
    compare_lbyl_vs_eafp()
    check_exception_raise()
    propagate_error_without_exceptions()


# Railway programming
# ------- OK scenario (return)
#   \
# ------- Error scenario (exceptions)


if __name__ == '__main__':
    main()
