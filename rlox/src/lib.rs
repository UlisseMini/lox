#[allow(non_camel_case_types)]
#[rustfmt::skip]
#[derive(Debug)]
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

    EOF
}

#[derive(Debug)]
pub enum Object {
    Nil,
}

#[derive(Debug)]
pub struct Token {
    type_: TokenType,
    lexeme: String,
    literal: Object,
    line: usize,
}

pub struct Scanner {
    source: String,
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
            source: source,
            tokens: Vec::new(),
        }
    }

    fn at_end(&self) -> bool {
        return self.current >= self.source.len();
    }

    fn scan_tokens(&mut self) {
        while !self.at_end() {
            let token = self.scan_token();
            self.tokens.push(token);
        }
    }

    fn scan_token(&self) -> Token {
        Token {
            type_: TokenType::IDENTIFIER,
            lexeme: "foo".to_string(),
            literal: Object::Nil,
            line: self.line,
        }
    }
}

pub struct Lox {
    had_error: bool,
}

impl Lox {
    pub fn new() -> Lox {
        Lox { had_error: false }
    }

    pub fn run(&mut self, source: &str) -> i32 {
        let mut scanner = Scanner::new(source.to_string());

        scanner.scan_tokens();
        for token in scanner.tokens {
            println!("{:?}", token);
        }

        let exitcode = if self.had_error { 65 } else { 0 };
        return exitcode;
    }

    pub fn error(&mut self, line: usize, message: String) {
        self.report(line, "".to_string(), message)
    }

    pub fn report(&mut self, line: usize, where_: String, message: String) {
        eprintln!("[line {}] Error{}: {}", line, where_, message);
        self.had_error = true;
    }
}
