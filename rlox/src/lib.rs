use std::fmt;

#[allow(non_camel_case_types)]
#[rustfmt::skip]
#[derive(Debug, Copy, Clone)]
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

#[derive(Debug)]
pub enum Object {
    Nil,
    Number(f64),
}

#[derive(Debug)]
pub struct Token {
    type_: TokenType,
    lexeme: String,
    literal: Object,
    line: usize,
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
        while Self::is_alphanumeric(self.peek()) {
            self.advance();
        }

        let lexeme = self.lexeme();
        let type_ = match KEYWORDS.get(&lexeme) {
            Some(type_) => *type_,
            None => TokenType::IDENTIFIER,
        };
        let token = self.emit_token(type_);
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
        let token = self.emit_token(TokenType::STRING);

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

#[derive(Debug)]
pub enum Literal {
    Number(f64),
    Bool(bool),
    Nil,
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Literal::*;
        match self {
            Number(n) => write!(f, "{}", n),
            Bool(b) => write!(f, "{}", b),
            Nil => write!(f, "nil"),
        }
    }
}

type ExprB = Box<Expr>;

#[derive(Debug)]
#[rustfmt::skip]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Eq
}

impl fmt::Display for BinaryOp {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use BinaryOp::*;
        write!(f, "{}", match self {
            Add => "+", Sub => "-", Mul => "*", Div => "/", Eq => "=="
        })
    }
}

#[derive(Debug)]
pub struct Binary {
    left: ExprB,
    right: ExprB,
    operator: BinaryOp,
}

impl fmt::Display for Binary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.operator, self.left, self.right)
    }
}

#[derive(Debug)]
#[rustfmt::skip]
pub enum UnaryOp {
    Minus, LogicalNot
}

impl fmt::Display for UnaryOp {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use UnaryOp::*;
        write!(f, "{}", match self {
            Minus => "-", LogicalNot => "!",
        })
    }
}

#[derive(Debug)]
pub struct Unary {
    right: ExprB,
    operator: UnaryOp,
}

impl fmt::Display for Unary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.operator, self.right)
    }
}

#[derive(Debug)]
pub enum Expr {
    Literal(Literal),
    Unary(Unary),
    Binary(Binary),
    Grouping(ExprB),
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
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Parser {
        Parser { tokens }
    }

    fn parse_tokens(&mut self) -> Result<Expr, Error> {
        Ok(Expr::Literal(Literal::Number(5.)))
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
        for token in &scanner.tokens {
            println!("{:?}", token);
        }
        let tokens = scanner.tokens;
        let mut parser = Parser::new(tokens);
        let expr = parser.parse_tokens()?;
        println!("{}", expr);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_pretty_print() {
        let expr = Expr::Binary(Binary {
            left: Box::new(Expr::Unary(Unary {
                operator: UnaryOp::Minus,
                right: Box::new(Expr::Literal(Literal::Number(123.))),
            })),
            operator: BinaryOp::Mul,
            right: Box::new(Expr::Grouping(Box::new(Expr::Literal(Literal::Number(
                45.67,
            ))))),
        });
        assert_eq!(format!("{}", expr), "(* (- 123) (group 45.67))");
    }
}
