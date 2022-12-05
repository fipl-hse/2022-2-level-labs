"""
Programming 2022
Seminar 8
Working with exceptions
"""


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
    dummy_b = []
    if len(dummy_b) - 1 > 100:
        print(dummy_b[100])
    else:
        print('No')

    # EAFP
    try:
        dummy_a = {}
        dummy_b = []
        print(dummy_b[12])
        print(dummy_a['key'])
    # pylint: disable=unused-variable
    except (KeyError, IndexError) as my_error:
        print('Error!!')
    # pylint: disable=duplicate-except
    except IndexError:
        print('IndexError!!')
    # pylint: disable=bare-except
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
    def dummy_f(first, second):
        return first / second

    def dummy_g(first, second):
        res = dummy_f(first, second)
        print(res)
        return res

    try:
        res = dummy_g(1, 0)
    except ZeroDivisionError:
        print('Error')
    else:
        print(res)


def propagate_error_without_exceptions():
    def dummy_f(first, second):
        if second == 0:
            return -1
        return first / second

    def dummy_g(first, second):
        res = dummy_f(first, second)
        if res == -1:
            return None
        print(res)
        return res

    res = dummy_g(1, 0)
    if res is not None:
        print(f'Success: {res}')


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
