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
        return self.emit(literal=self.lexeme()[1:-1])

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


def test_scanner():
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


# ==================== Parser ==================== 

class Parser:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.current = 0

    # ======== Helpers ========

    def at_end(self):
        return self.current >= len(self.tokens)

    def peek(self):
        return self.tokens[self.current]

    def advance(self):
        tok = self.peek()
        self.current += 1
        return tok

    def error(self, msg: str):
        raise ValueError(msg)

    def eat(self, want: str):
        if self.at_end(): self.error(f'want {want} got EOF')
        got = self.advance()
        if got.lexeme != want: self.error(f'want {want} got {got}')

    def until(self, cond, method):
        res = []
        while not self.at_end() and not cond(): res.append(method())
        return res

    # ======== Parse ========

    def parse(self):
        statements = []
        while not self.at_end():
            statements.append(self.parse_statement())

        return statements

    def parse_statement(self):
        token = self.advance()

        if token.lexeme == 'if':
            self.eat('(')
            cond = self.parse_expr()
            self.eat(')')
            stmt = self.parse_statement()
            return ('if', cond, stmt)
        elif token.lexeme == '{':
            return self.parse_block()
        elif token.lexeme == 'print':
            operand = self.parse_expr()
            self.eat(';')
            return ('print', operand)
        else:
            print(f'{token.lexeme} not handled')

    def parse_block(self):
        stmts = self.until(lambda: self.peek().lexeme == '}', self.parse_statement)
        self.eat('}')
        return stmts

    def parse_expr(self):
        expr = self.advance().literal

        while self.peek().lexeme in '+*/-':
            op = self.advance()
            rhs = self.parse_expr()
            expr = (op.lexeme, expr, rhs)

        return expr


def sexpr(ast) -> str:
    if isinstance(ast, list):
        return '\n'.join(map(sexpr, ast))
    elif isinstance(ast, tuple):
        return '(' + ' '.join(map(sexpr, ast)) + ')'
    else:
        return str(ast)


def test_parser():
    source = 'if(2 + 3) { print "foo"; }'
    tokens = Scanner(source).scan()
    ast = Parser(tokens).parse()
    print('AST:')
    print(sexpr(ast))


if __name__ == '__main__':
    test_scanner()
    test_parser()
