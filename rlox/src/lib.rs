use std::fmt;

#[allow(non_camel_case_types)]
#[rustfmt::skip]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TokenType {
    // Single character
    LEFT_PAREN, RIGHT_PAREN, LEFT_BRACE, RIGHT_BRACE,
    COMMA, DOT, MINUS, PLUS, SEMICOLON, SLASH, STAR,

    // One or two
    BANG, BANG_EQUAL,
    EQUAL, EQUAL_EQUAL,
    GREATER, GREATER_EQUAL,
    LESS, LESS_EQUAL,

    // Literals
    IDENTIFIER, STRING, NUMBER,

    // Keywords
    AND, CLASS, ELSE, FALSE, FUN, FOR, IF, NIL, OR,
    PRINT, RETURN, SUPER, THIS, TRUE, VAR, WHILE,

    // Misc
    COMMENT, EOF
}

use phf::phf_map;

static KEYWORDS: phf::Map<&'static str, TokenType> = phf_map! {
    "and" =>    TokenType::AND,
    "class" =>  TokenType::CLASS,
    "else" =>   TokenType::ELSE,
    "false" =>  TokenType::FALSE,
    "for" =>    TokenType::FOR,
    "fun" =>    TokenType::FUN,
    "if" =>     TokenType::IF,
    "nil" =>    TokenType::NIL,
    "or" =>     TokenType::OR,
    "print" =>  TokenType::PRINT,
    "return" => TokenType::RETURN,
    "super" =>  TokenType::SUPER,
    "this" =>   TokenType::THIS,
    "true" =>   TokenType::TRUE,
    "var" =>    TokenType::VAR,
    "while" =>  TokenType::WHILE,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Object {
    Nil,
    Number(f64),
    String(String),
    Bool(bool),
}

impl Object {
    fn number(&self) -> Option<f64> {
        match self {
            Object::Number(n) => Some(*n),
            _ => None,
        }
    }

    fn bool(&self) -> Option<bool> {
        match self {
            Object::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

// NOTE: all of these operators panic on invalid types, you must check types before calling them.
use std::ops::{Add, Div, Mul, Neg, Not};

impl Add for Object {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::Number(self.number().unwrap() + other.number().unwrap())
    }
}

impl Mul for Object {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self::Number(self.number().unwrap() * other.number().unwrap())
    }
}

impl Div for Object {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        Self::Number(self.number().unwrap() / other.number().unwrap())
    }
}

impl Neg for Object {
    type Output = Self;
    fn neg(self) -> Self {
        Self::Number(-self.number().unwrap())
    }
}

impl Not for Object {
    type Output = Self;
    fn not(self) -> Self {
        Self::Bool(!self.bool().unwrap())
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Object::*;
        match self {
            Nil => write!(f, "nil"),
            Number(n) => write!(f, "{}", n),
            String(s) => write!(f, "\"{}\"", s),
            Bool(b) => write!(f, "{}", b),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    type_: TokenType,
    lexeme: String,
    literal: Object,
    line: usize,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.literal)
    }
}

pub struct Scanner {
    chars: Vec<char>,
    tokens: Vec<Token>,

    start: usize,
    current: usize,
    line: usize,
}

impl Scanner {
    fn new(source: String) -> Scanner {
        Scanner {
            start: 0,
            current: 0,
            line: 1,
            chars: source.chars().collect(),
            tokens: Vec::new(),
        }
    }

    fn scan_tokens(&mut self) -> Result<(), Error> {
        while !self.at_end() {
            if let Some(token) = self.scan_token()? {
                self.tokens.push(token);
            }
            self.start = self.current;
        }
        Ok(())
    }

    fn scan_token(&mut self) -> Result<Option<Token>, Error> {
        use TokenType::*;

        let c = self.advance();
        let type_ = match c {
            '(' => LEFT_PAREN,
            ')' => RIGHT_PAREN,
            '{' => LEFT_BRACE,
            '}' => RIGHT_BRACE,
            ',' => COMMA,
            '.' => DOT,
            '-' => MINUS,
            '+' => PLUS,
            ';' => SEMICOLON,
            '*' => STAR,

            '!' if self.advance_if('=') => BANG_EQUAL,
            '!' => BANG,
            '=' if self.advance_if('=') => EQUAL_EQUAL,
            '=' => EQUAL,
            '<' if self.advance_if('=') => LESS_EQUAL,
            '<' => LESS,
            '>' if self.advance_if('=') => GREATER_EQUAL,
            '>' => GREATER,

            '/' if self.advance_if('/') => {
                while self.peek() != '\n' && !self.at_end() {
                    self.advance();
                }
                self.start += 2;
                COMMENT
            }
            '/' => SLASH,

            '\n' | ' ' | '\t' | '\r' => {
                return Ok(None);
            }

            c if Self::is_digit(c) => return self.number(),
            c if Self::is_alpha(c) => return self.identifier(),
            '"' => return self.string(),

            _ => {
                // TODO: make a macro for this
                return Err(self.error(format!("Invalid character {:?}", c)));
            }
        };
        return Ok(Some(self.emit_token(type_)));
    }

    fn identifier(&mut self) -> Result<Option<Token>, Error> {
        use TokenType::*;

        while Self::is_alphanumeric(self.peek()) {
            self.advance();
        }

        let lexeme = self.lexeme();
        let type_ = match KEYWORDS.get(&lexeme) {
            Some(type_) => *type_,
            None => IDENTIFIER,
        };
        let token = match type_ {
            TRUE => self.emit_literal(type_, Object::Bool(true)),
            FALSE => self.emit_literal(type_, Object::Bool(false)),
            _ => self.emit_token(type_),
        };
        Ok(Some(token))
    }

    fn number(&mut self) -> Result<Option<Token>, Error> {
        use std::str::FromStr;

        // TODO: This pattern comes up a lot, abstract with higher order function
        while Self::is_digit(self.peek()) {
            self.advance();
        }

        // Look for fractional bit
        if self.peek() == '.' && Self::is_digit(self.peek_next()) {
            self.advance(); // consume the "."
            while Self::is_digit(self.peek()) {
                self.advance();
            }
        }

        // unwrap here is fine since if the number is invalid it is a bug in our lexer.
        let lexeme = self.lexeme();
        let num = f64::from_str(&lexeme).unwrap();
        let token = self.emit_literal(TokenType::NUMBER, Object::Number(num));
        Ok(Some(token))
    }

    // this shoulden't return an option, I just did this to line the types up with scan_token.
    fn string(&mut self) -> Result<Option<Token>, Error> {
        while self.peek() != '"' && !self.at_end() {
            self.advance();
        }
        if self.at_end() {
            return Err(self.error(format!("Missing end of string")));
        }

        // this is a mildly cursed way to ignore the quotes, whatever

        self.start += 1; // ignore the first '"'
        let token = self.emit_literal(TokenType::STRING, Object::String(self.lexeme()));

        self.advance(); // consume the closing '"'
        return Ok(Some(token));
    }

    fn lexeme(&self) -> String {
        self.chars[self.start..self.current].iter().collect()
    }

    fn emit_literal(&mut self, type_: TokenType, literal: Object) -> Token {
        Token {
            type_: type_,
            lexeme: self.lexeme(),
            literal: literal,
            line: self.line,
        }
    }

    fn emit_token(&mut self, type_: TokenType) -> Token {
        self.emit_literal(type_, Object::Nil)
    }

    fn error(&self, message: String) -> Error {
        Error::new(self.line, "".to_string(), message)
    }

    fn is_digit(c: char) -> bool {
        c.is_digit(10)
    }

    fn is_alpha(c: char) -> bool {
        c.is_alphabetic() || c == '_'
    }

    fn is_alphanumeric(c: char) -> bool {
        Self::is_alpha(c) || c.is_numeric()
    }

    fn at_end(&self) -> bool {
        return self.current >= self.chars.len();
    }

    fn advance(&mut self) -> char {
        let c = self.chars[self.current];
        // kinda hackish, done since I want to only change line in a single place.
        if self.current > 0 && self.chars[self.current - 1] == '\n' {
            self.line += 1;
        }
        self.current += 1;
        return c;
    }

    fn advance_if(&mut self, want: char) -> bool {
        let advance = !self.at_end() && self.chars[self.current] == want;
        if advance {
            self.advance();
        }
        return advance;
    }

    fn peek(&self) -> char {
        if self.at_end() {
            '\0'
        } else {
            self.chars[self.current]
        }
    }

    fn peek_next(&self) -> char {
        if self.current + 1 >= self.chars.len() {
            '\0'
        } else {
            self.chars[self.current + 1]
        }
    }
}

#[derive(Debug)]
pub struct Error {
    line: usize,
    where_: String,
    message: String,
}

impl Error {
    fn new(line: usize, where_: String, message: String) -> Error {
        return Error {
            line,
            where_,
            message,
        };
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[line {}] Error{}: {}",
            self.line, self.where_, self.message
        )
    }
}

type ExprB = Box<Expr>;

#[derive(Debug)]
pub struct Binary {
    left: ExprB,
    right: ExprB,
    operator: Token,
}

impl Binary {
    fn eval(&self) -> Result<Object, Error> {
        use TokenType::*;

        let left = self.left.eval()?;
        let right = self.right.eval()?;

        // FIXME: Error handling
        Ok(match self.operator.type_ {
            PLUS => left + right,
            STAR => left * right,
            SLASH => left / right,

            _ => panic!("invalid binary operator '{}'", self.operator.lexeme),
        })
    }
}

impl fmt::Display for Binary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.operator.lexeme, self.left, self.right)
    }
}

#[derive(Debug)]
pub struct Unary {
    right: ExprB,
    operator: Token,
}

impl Unary {
    fn eval(&self) -> Result<Object, Error> {
        use TokenType::*;

        let right = self.right.eval()?;
        Ok(match self.operator.type_ {
            MINUS => -right,
            PLUS => right, // TODO: assert number
            BANG => !right,
            _ => panic!("invalid unary operator '{}'", self.operator.lexeme),
        })
    }
}

impl fmt::Display for Unary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.operator.lexeme, self.right)
    }
}

#[derive(Debug)]
pub enum Expr {
    Literal(Token),
    Unary(Unary),
    Binary(Binary),
    Grouping(ExprB),
}

impl Expr {
    fn eval(&self) -> Result<Object, Error> {
        use Expr::*;
        Ok(match self {
            Literal(t) => t.literal.clone(),
            Unary(u) => u.eval()?,
            Binary(b) => b.eval()?,
            Grouping(g) => g.eval()?,
        })
    }

    fn binary(left: Expr, operator: Token, right: Expr) -> Expr {
        Expr::Binary(Binary {
            left: Box::new(left),
            operator: operator,
            right: Box::new(right),
        })
    }

    fn unary(operator: Token, right: Expr) -> Expr {
        Expr::Unary(Unary {
            operator: operator,
            right: Box::new(right),
        })
    }

    fn grouping(expr: Expr) -> Expr {
        Expr::Grouping(Box::new(expr))
    }

    fn literal(tok: Token) -> Expr {
        use TokenType::*;
        match tok.type_ {
            NUMBER | STRING | TRUE | FALSE | NIL => Expr::Literal(tok),
            _ => panic!("{:?} is not a literal", tok),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Expr::*;

        match self {
            Literal(lit) => write!(f, "{}", lit),
            Unary(un) => write!(f, "({})", un),
            Binary(bin) => write!(f, "({})", bin),
            Grouping(expr) => write!(f, "(group {})", expr),
        }
    }
}

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

// Parsing is essentially the same as scanning, except every character is now a token.
impl Parser {
    fn new(tokens: Vec<Token>) -> Parser {
        let current = 0;
        Parser { tokens, current }
    }

    fn parse_tokens(&mut self) -> Result<Expr, Error> {
        self.expression()
    }

    // This is a literal translation of the BNF style grammer described in
    // https://craftinginterpreters.com/parsing-expressions.html
    // In order of increasing precedence
    /*
      expression     → equality ;
      equality       → comparison ( ( "!=" | "==" ) comparison )* ;
      comparison     → term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
      term           → factor ( ( "-" | "+" ) factor )* ;
      factor         → unary ( ( "/" | "*" ) unary )* ;
      unary          → ( "!" | "-" ) unary | primary ;
      primary        → NUMBER | STRING | "true" | "false" | "nil" | "(" expression ")" ;
    */
    // Parser combinators would make this very nice

    fn expression(&mut self) -> Result<Expr, Error> {
        self.equality()
    }

    fn equality(&mut self) -> Result<Expr, Error> {
        use TokenType::*;

        let mut expr = self.comparison()?;
        while self.advance_if(&[EQUAL_EQUAL, BANG_EQUAL]) {
            let operator = self.previous();
            let right = self.comparison()?;
            expr = Expr::binary(expr, operator, right);
        }
        Ok(expr)
    }

    fn comparison(&mut self) -> Result<Expr, Error> {
        use TokenType::*;

        let mut expr = self.term()?;
        while self.advance_if(&[GREATER, GREATER_EQUAL, LESS, LESS_EQUAL]) {
            let operator = self.previous();
            let right = self.term()?;
            expr = Expr::binary(expr, operator, right);
        }
        Ok(expr)
    }

    fn term(&mut self) -> Result<Expr, Error> {
        use TokenType::*;
        let mut expr = self.factor()?;
        while self.advance_if(&[PLUS, MINUS]) {
            let operator = self.previous();
            let right = self.factor()?;
            expr = Expr::binary(expr, operator, right);
        }
        Ok(expr)
    }

    fn factor(&mut self) -> Result<Expr, Error> {
        use TokenType::*;
        let mut expr = self.unary()?;
        while self.advance_if(&[SLASH, STAR]) {
            let operator = self.previous();
            let right = self.unary()?;
            expr = Expr::binary(expr, operator, right);
        }
        Ok(expr)
    }

    fn unary(&mut self) -> Result<Expr, Error> {
        use TokenType::*;

        if self.advance_if(&[BANG, MINUS]) {
            let operator = self.previous();
            let right = self.unary()?;
            Ok(Expr::unary(operator, right))
        } else {
            self.primary()
        }
    }

    // NUMBER | STRING | "true" | "false" | "nil" | "(" expression ")" ;
    fn primary(&mut self) -> Result<Expr, Error> {
        use TokenType::*;
        if self.advance_if(&[NUMBER, STRING, TRUE, FALSE, NIL]) {
            Ok(Expr::literal(self.previous()))
        } else if self.advance_if(&[LEFT_PAREN]) {
            let expr = self.expression()?;
            self.consume(RIGHT_PAREN)?;
            Ok(Expr::grouping(expr))
        } else {
            let tok = self.current();
            Err(Error::new(
                tok.line,
                "".to_string(),
                format!("Expected expression, got {:?}", tok.type_),
            ))
        }
    }

    fn current(&self) -> Token {
        if self.at_end() {
            return Token {
                type_: TokenType::EOF,
                lexeme: "".to_string(),
                literal: Object::Nil,
                line: self.previous().line,
            };
        }

        self.tokens[self.current].clone()
    }

    fn previous(&self) -> Token {
        self.tokens[self.current - 1].clone()
    }

    fn advance(&mut self) {
        self.current += 1;
    }

    fn at_end(&self) -> bool {
        return self.current >= self.tokens.len();
    }

    fn advance_if(&mut self, types: &[TokenType]) -> bool {
        let tok = self.current();
        for type_ in types {
            if type_ == &tok.type_ {
                self.advance();
                return true;
            }
        }
        return false;
    }

    fn consume(&mut self, type_: TokenType) -> Result<(), Error> {
        let tok = self.current();
        if tok.type_ == type_ {
            self.advance();
            Ok(())
        } else {
            Err(Error::new(
                tok.line,
                "".to_string(),
                format!("expected {:?} got {:?}", type_, tok.type_),
            ))
        }
    }
}

pub struct Lox {}

impl Lox {
    pub fn new() -> Lox {
        Lox {}
    }

    pub fn run(&mut self, source: &str) -> Result<(), Error> {
        let mut scanner = Scanner::new(source.to_string());

        scanner.scan_tokens()?;
        // for token in &scanner.tokens {
        //     println!("{:?}", token);
        // }
        let tokens = scanner.tokens;
        let mut parser = Parser::new(tokens);
        let expr = parser.parse_tokens()?;
        println!("{}", expr);
        let result = expr.eval().unwrap();
        println!("=> {}", result);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser() {
        fn s(source: &str) -> String {
            let mut scanner = Scanner::new(source.to_string());
            scanner.scan_tokens().unwrap();
            let mut parser = Parser::new(scanner.tokens);
            format!("{}", parser.parse_tokens().unwrap())
        }

        assert_eq!(s("2 + 3"), "(+ 2 3)");
        assert_eq!(s("2 + 3*4"), "(+ 2 (* 3 4))");
        assert_eq!(s("2 + 3 + 4"), "(+ (+ 2 3) 4)");
        assert_eq!(s("2 >= 3 + 4*2"), "(>= 2 (+ 3 (* 4 2)))");
        assert_eq!(s("2 >= (3 + 4)*2"), "(>= 2 (* (group (+ 3 4)) 2))");
    }

    #[test]
    fn test_evaluator() {
        fn e(source: &str) -> Object {
            // TODO: fix this garbage (put in lox struct)
            let mut scanner = Scanner::new(source.to_string());
            scanner.scan_tokens().unwrap();
            let mut parser = Parser::new(scanner.tokens);
            let expr = parser.parse_tokens().unwrap();
            let mut l = Lox::new();
            expr.eval().unwrap()
        }

        assert_eq!(e("2"), Object::Number(2.));
        assert_eq!(e("-2"), Object::Number(-2.));
        assert_eq!(e("!false"), Object::Bool(true));
        assert_eq!(e("2 + 2"), Object::Number(4.));
        assert_eq!(e("(2 + 3) * 4"), Object::Number(20.));
        assert_eq!(e("(2 + 6) / 4"), Object::Number(2.));
        // assert_eq!(e("(2 + 6) == 8"), Object::Bool(true));
    }
}
