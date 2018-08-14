import os

__tests_location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))
)

print(__tests_location__)
