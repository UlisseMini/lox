import string
from dataclasses import dataclass
from enum import Enum
from typing import List, Callable, Any, Optional, Generator

KEYWORDS = {'if', 'while'}

class TokenType(Enum):
    EOF = 0
    IDENT = 1
    LITERAL = 2
    KEYWORD = 3
    OP = 4
    SYM = 5

    def __str__(self):
        return self.name

@dataclass
class Token:
    lexeme: str
    type_: TokenType
    literal: Any = None

    def __bool__(self):
        return self.type_ != TokenType.EOF

    def __repr__(self):
        if self.literal:
            return f'T({self.lexeme}, {self.type_}, {self.literal})'
        else:
            return f'T({self.lexeme}, {self.type_})'

is_digit = lambda c: c in string.digits
is_alpha = lambda c: c in string.ascii_letters+'_'
is_alphanumeric = lambda c: is_digit(c) or is_alpha(c)

class Scanner:
    def __init__(self, source: str):
        self.start = 0
        self.current = 0
        self.source = source

    def scan(self):
        while not self.at_end():
            char = self.advance()
            if is_digit(char): yield self.scan_number()
            if is_alpha(char): yield self.scan_ident()
            if char in '+-*/=!><': yield self.scan_op(char)
            self.start = self.current

        return Token("", TokenType.EOF)

    def scan_op(self, char: str):
        two = char + self.peek()
        if two in ['==', '!=', '>=', '<=']:
            self.advance()
            return self.emit(TokenType.OP)
        else:
            return self.emit(TokenType.OP)

    def scan_number(self) -> Token:
        self.advance_if(lambda x: x == '.')
        self.advance_while(is_digit)
        return self.emit(TokenType.LITERAL, float(self.lexeme()))

    def scan_ident(self) -> Token:
        self.advance_while(is_alphanumeric)
        type_ = TokenType.KEYWORD if self.lexeme() in KEYWORDS else TokenType.IDENT
        return self.emit(type_)

    def advance_while(self, pred: Callable):
        while pred(self.peek()): self.advance()

    def advance_if(self, pred: Callable) -> Optional[str]:
        if pred(self.peek()):
            return self.advance()

    def advance(self):
        char = self.source[self.current]
        self.current += 1
        return char

    def eat(self, want: str):
        got = self.advance()
        assert got == want, f'want {want} got {got}'

    def peek(self):
        return '\0' if self.at_end() else self.source[self.current]

    def at_end(self):
        return self.current >= len(self.source)

    def lexeme(self):
        return self.source[self.start : self.current]

    def emit(self, type_: TokenType, literal=None) -> Token:
        tok = Token(lexeme=self.lexeme(), type_=type_, literal=literal)
        self.start = self.current
        return tok


tests = [
    ("3*foo+5.2", "[T(3, LITERAL, 3.0), T(*, OP), T(foo, IDENT), T(+, OP), T(5.2, LITERAL, 5.2)]"),
    ("foo", "[T(foo, IDENT)]"),
    ("foo != !bar", "[T(foo, IDENT), T(!=, OP), T(!, OP), T(bar, IDENT)]"),
    ("if(foo != bar)", "[T(if, KEYWORD), T(foo, IDENT), T(!=, OP), T(bar, IDENT)]"),
]

for source, want_tokens in tests:
    got_tokens = str(list(Scanner(source).scan()))
    assert got_tokens == want_tokens, f'got {got_tokens} want {want_tokens}'

