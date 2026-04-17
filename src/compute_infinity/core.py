from abc import ABC, abstractmethod
from collections.abc import Sequence

Number = int | float
Vector = Sequence[Number]
Matrix = Sequence[Sequence[Number]]
Operand = Number | Vector | Matrix


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
