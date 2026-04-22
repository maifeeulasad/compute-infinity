from abc import ABC, abstractmethod
from collections.abc import Sequence

Number = int | float
Vector = Sequence[Number]
Matrix = Sequence[Sequence[Number]]
Operand = Number | Vector | Matrix


def is_number(value: object) -> bool:
    return isinstance(value, (int, float))


def is_vector(value: object) -> bool:
    if is_number(value) or isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, Sequence):
        return False
    return all(is_number(item) for item in value)


def is_matrix(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if len(value) == 0:
        return False
    first_row = value[0]
    if not is_vector(first_row):
        return False
    width = len(first_row)
    return all(is_vector(row) and len(row) == width for row in value)


def ensure_operand(value: Operand) -> None:
    if is_number(value) or is_vector(value) or is_matrix(value):
        return
    raise TypeError("Operand must be Number, Vector, or Matrix")


def same_shape(left: Operand, right: Operand) -> bool:
    if is_vector(left) and is_vector(right):
        return len(left) == len(right)
    if is_matrix(left) and is_matrix(right):
        return len(left) == len(right) and len(left[0]) == len(right[0])
    return False


class ArithmeticBackend(ABC):
    @abstractmethod
    def plus(self, left: Operand, right: Operand) -> Operand:
        """Add two operands."""

    @abstractmethod
    def minus(self, left: Operand, right: Operand) -> Operand:
        """Subtract right operand from left operand."""

    @abstractmethod
    def mul(self, left: Operand, right: Operand) -> Operand:
        """Multiply two operands."""

    @abstractmethod
    def divide(self, left: Operand, right: Operand) -> Operand:
        """Divide left operand by right operand."""


def hello_world() -> str:
    return "hello, world"
