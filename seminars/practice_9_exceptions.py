"""
Programming 2022
Seminar 9
Working with exceptions
"""
class Cartridge:
    """
    Class describes a printer cartridge
    """

    def __init__(self, value: int, capacity: int = 1000) -> None:
        self._value = value
        self._capacity = capacity

    def get_paint(self, num_portion: int) -> int:
        """
        Method to return a paint to print `num_portion` characters
        if paint is lower returns -1
        """
        if num_portion > self._value:
            return -1

        self._value -= num_portion
        return self._value


class Printer:
    """
    Class describes a printer that can do print
    """
    def __init__(self, name: str, cartridge: Cartridge) -> None:
        self._name = name
        self._cartridge = cartridge

    def print(self, string_to_print: str) -> bool:
        """
        The method to print characters
        """

        print(f'===== Priner {self._name} starts print =====')

        str_len = len(string_to_print)
        paint = self._cartridge.get_paint(str_len)
        if paint < 0:
            print(f'===== Printer {self._name}: There is not enough paint in the printer =====')
            return False

        print(string_to_print)
        print(f'===== Printer {self._name} ended print =====')

        return True


# Create a cartridge
CARTRIDGE = Cartridge(100)
# Create a printer
PRINTER = Printer('BestPrinterEver', CARTRIDGE)


# Print "Hello World!"
PRINTER.print('Hello World!')

# Task 1:
# Easy level
# Align error handling in the code (use only bools or only int values)


# Task 2:
# Medium level
# Replace return aproach to rasing errors using default AssertionError


# Task 3:
# Medium level
# Create a method of Сartridge to refill the cartridge


# Task 4:
# Medium level
# Create your own errors


# Task 5:
# Medium level
# Handle exception in a client code
