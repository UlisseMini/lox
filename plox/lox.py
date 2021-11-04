import string
from dataclasses import dataclass
from enum import Enum
from typing import List, Callable, Any, Optional, Generator

KEYWORDS = {'if', 'while'}

@dataclass
class Token:
    lexeme: str
    literal: Any = None

    def __str__(self):
        return self.lexeme

    def __bool__(self):
        # eof is represented as an empty lexeme
        return self.lexeme != ''


# some predicates for parsing
is_digit = lambda c: c in string.digits
is_alpha = lambda c: c in string.ascii_letters+'_'
is_alphanumeric = lambda c: is_digit(c) or is_alpha(c)

class Scanner:
    def __init__(self, source: str):
        self.start = 0
        self.current = 0
        self.source = source

    # ======== Helpers ========

    def emit(self, literal=None) -> Token:
        tok = Token(lexeme=self.lexeme(), literal=literal)
        self.start = self.current
        return tok

    def at_end(self):
        return self.current >= len(self.source)

    def peek(self):
        return '' if self.at_end() else self.source[self.current]

    def advance(self):
        char = self.peek()
        self.current += 1
        return char

    def lexeme(self):
        return self.source[self.start : self.current]

    def advance_if(self, pred: Callable) -> Optional[str]:
        if not self.at_end() and pred(self.peek()):
            return self.advance()

    def advance_while(self, pred: Callable):
        while self.advance_if(pred): pass

    # ======== Scanning ========

    def scan(self):
        while char := self.advance():
            if char == '"': yield self.scan_string()
            if is_digit(char): yield self.scan_number()
            if is_alpha(char): yield self.scan_ident()
            if char in '+-*/=!><()}{;': yield self.scan_sym(char)
            self.start = self.current

        return Token('')

    def scan_string(self):
        self.advance_while(lambda c: c != '"')
        assert not self.at_end(), 'eof before end of string'
        self.advance()
        return self.emit()

    def scan_sym(self, char: str):
        two = char + self.peek()
        if two in {'==', '!=', '>=', '<='}:
            self.advance()

        return self.emit()

    def scan_number(self) -> Token:
        self.advance_if(lambda x: x == '.')
        self.advance_while(is_digit)
        return self.emit(literal=float(self.lexeme()))

    def scan_ident(self) -> Token:
        self.advance_while(is_alphanumeric)
        return self.emit()


tests = [
    ("3*foo+5.2", '3 * foo + 5.2'),
    ("foo != !bar", 'foo != ! bar'),
    ("if(foo != bar)", 'if ( foo != bar )'),
    ('if(foo != bar) { print "cats"; }', 'if ( foo != bar ) { print "cats" ; }'),
    ('while(foo!=bar){print "dogs"; foo=bar+1;}',
     'while ( foo != bar ) { print "dogs" ; foo = bar + 1 ; }'),
]

for source, want in tests:
    got_tokens = list(Scanner(source).scan())
    got = ' '.join([str(tok) for tok in got_tokens])
    assert want == got, f'got "{got}" want "{want}"'

print(f'Tests passed')
