"""
Programming 2022
Seminar 7
Introduction into exceptions
"""
from random import randint

# Define functions that can raise an error
def bad_function() -> None:
    # Any exception
    raise Exception(f'Exception from bad_function')

def div(numerator: float, denominator: float) -> float:
    # raises ZeroDivisionError
    return numerator / denominator

# TASKS
# easy level

# Task 1
# Catch an exception without specifying its type
# bad_function()

# Task 2
# Print error message for Zero Division Error
# numerator = randint(1, 10)
# for denominator in [3, 4, 0, 8]:
#     result = div(numerator, denominator)
#     print(f'{numerator} / {denominator} = {result}')

# Task 3
# MyList is a specific implementation of list - you can't usethe builtin function len.
# Implement a method of find
# Use try except aproach (with specified Exception type) to find the length
class MyList(list):
    def __len__(self) -> int:
        raise AssertionError('Do not use len!')


ARRAY = MyList((0 for _ in range(randint(0, 4))))

LENGTH = 0

print(f'The length of {ARRAY} is {LENGTH}')

# Task 4
# MyList is a specific implementation of list class - you can't usethe builtin function len.
# Implement a method of finfing length of the MyList using the check_index_of_array function
# Use try except aproach (with specified Exception type) to find the length
