use std::fmt;
use std::rc::Rc;

#[allow(non_camel_case_types)]
#[rustfmt::skip]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TokenType {
    // Single character
    LEFT_PAREN, RIGHT_PAREN, LEFT_BRACE, RIGHT_BRACE,
    COMMA, DOT, MINUS, PLUS, SEMICOLON, SLASH, STAR, PERCENT,

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
    EOF
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
    Callable(Rc<Box<dyn Callable>>),
}

#[rustfmt::skip]
impl Object {
    fn err(&self, want: &str) -> Error {
        Error::new(
            0, // error will be caught in Interpreter and the line number will be corrected.
            "".to_string(),
            format!("want {} got {:?}", want, self),
        )
    }

    fn number(&self) -> Result<f64, Error> {
        if let Object::Number(n) = self { Ok(*n) } else { Err(self.err("number")) }
    }

    fn bool(&self) -> Result<bool, Error> {
        if let Object::Bool(b) = self { Ok(*b) } else { Err(self.err("bool")) }
    }

    fn add(self, other: Object) -> Result<Object, Error> {
        use Object::*;
        match (&self, &other) {
            (String(s), String(o)) => Ok(Object::String(s.to_owned() + o)),
            (Number(s), Number(o)) => Ok(Object::Number(s + o)),
            _ => Err(self.err("number or string")),
        }
    }

    fn sub(self, other: Object) -> Result<Object, Error> {
        Ok(Self::Number(self.number()? - other.number()?))
    }

    fn mul(self, other: Self) -> Result<Self, Error> {
        Ok(Self::Number(self.number()? * other.number()?))
    }

    fn div(self, other: Self) -> Result<Self, Error> {
        Ok(Self::Number(self.number()? / other.number()?))
    }

    fn modulo(self, other: Self) -> Result<Self, Error> {
        Ok(Self::Number(self.number()? % other.number()?))
    }

    fn neg(self) -> Result<Self, Error> {
        Ok(Self::Number(-self.number()?))
    }

    fn not(self) -> Result<Self, Error> {
        Ok(Self::Bool(!self.bool()?))
    }

    fn less_equal(self, other: Object) -> Result<Self, Error> {
        Ok(Self::Bool(self.number()? <= other.number()?))
    }

    fn greater_equal(self, other: Object) -> Result<Self, Error> {
        Ok(Self::Bool(self.number()? >= other.number()?))
    }

    fn greater(self, other: Object) -> Result<Self, Error> {
        Ok(Self::Bool(self.number()? > other.number()?))
    }

    fn less(self, other: Object) -> Result<Self, Error> {
        Ok(Self::Bool(self.number()? < other.number()?))
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
            Callable(_) => todo!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
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
    fn new() -> Scanner {
        Scanner { start: 0, current: 0, line: 1, chars: Vec::new(), tokens: Vec::new() }
    }

    fn add_chars(&mut self, source: String) {
        let mut v: Vec<char> = source.chars().collect();
        self.chars.append(&mut v);
    }

    fn scan_tokens(mut self, source: String) -> Result<Vec<Token>, Error> {
        self.add_chars(source);

        while !self.at_end() {
            if let Some(token) = self.scan_token()? {
                self.tokens.push(token);
            }
            self.start = self.current;
        }
        Ok(self.tokens)
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
            '%' => PERCENT,

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
                return Ok(None);
            }
            '/' => SLASH,

            '\n' | ' ' | '\t' | '\r' => {
                return Ok(None);
            }

            c if Self::is_digit(c) => return self.number(),
            c if Self::is_alpha(c) => return self.identifier(),
            '"' => return self.string(),

            _ => {
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
        let mut string = String::new();
        let mut prev = '\0';
        while self.peek() != '"' && !self.at_end() {
            let c = self.advance();
            // if c is \n, and prev is not \, ie. the \ is not escaped
            if c == '\\' && prev != '\\' {
                if self.advance_if('n') {
                    string.push('\n');
                }
                // we don't want to push an escaped backslash, since we already
                // did previously. we want "\\" -> \
            } else {
                string.push(c);
            }
            prev = c;
        }
        if self.at_end() {
            return Err(self.error(format!("Missing end of string")));
        }

        // this is a mildly cursed way to ignore the quotes, whatever

        self.start += 1; // ignore the first '"'
        let token = self.emit_literal(TokenType::STRING, Object::String(string));

        self.advance(); // consume the closing '"'
        return Ok(Some(token));
    }

    fn lexeme(&self) -> String {
        self.chars[self.start..self.current].iter().collect()
    }

    fn emit_literal(&mut self, type_: TokenType, literal: Object) -> Token {
        Token { type_: type_, lexeme: self.lexeme(), literal: literal, line: self.line }
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

#[derive(Debug, PartialEq)]
pub struct Error {
    line: usize,
    where_: String,
    message: String,
}

impl Error {
    fn new(line: usize, where_: String, message: String) -> Error {
        return Error { line: line, where_: where_.into(), message: message.into() };
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[line {}] Error{}: {}", self.line, self.where_, self.message)
    }
}

type ExprB = Box<Expr>;

#[derive(Debug)]
pub struct Binary {
    left: ExprB,
    right: ExprB,
    operator: Token,
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

impl fmt::Display for Unary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.operator.lexeme, self.right)
    }
}

#[derive(Debug)]
pub struct Logical {
    left: ExprB,
    operator: Token,
    right: ExprB,
}

impl fmt::Display for Logical {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.operator, self.left, self.right)
    }
}

pub struct DisplayVec<'a, T>(&'a Vec<T>)
where
    T: fmt::Display;

impl<T: fmt::Display> fmt::Display for DisplayVec<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (k, v) in self.0.iter().enumerate() {
            if k > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", v)?;
        }
        write!(f, "]")
    }
}

#[derive(Debug)]
pub struct Call {
    callee: ExprB,
    paren: Token,
    args: Vec<Expr>,
}

impl fmt::Display for Call {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(call {} {})", self.callee, DisplayVec(&self.args))
    }
}

#[derive(Debug)]
pub enum Expr {
    Literal(Object),
    Unary(Unary),
    Binary(Binary),
    Grouping(ExprB),
    Identifier(Identifier),
    Assign(Identifier, ExprB),
    Logical(Logical),
    Call(Call),
}

impl Expr {
    fn binary(left: Expr, operator: Token, right: Expr) -> Expr {
        Expr::Binary(Binary { left: Box::new(left), operator: operator, right: Box::new(right) })
    }

    fn unary(operator: Token, right: Expr) -> Expr {
        Expr::Unary(Unary { operator: operator, right: Box::new(right) })
    }

    fn grouping(expr: Expr) -> Expr {
        Expr::Grouping(Box::new(expr))
    }

    fn literal(tok: Token) -> Expr {
        use TokenType::*;
        match tok.type_ {
            NUMBER | STRING | TRUE | FALSE | NIL => Expr::Literal(tok.literal),
            _ => panic!("{:?} is not a literal", tok),
        }
    }

    fn logical(left: Expr, operator: Token, right: Expr) -> Expr {
        use TokenType::*;
        Expr::Logical(match operator.type_ {
            OR | AND => {
                Logical { left: Box::new(left), operator: operator, right: Box::new(right) }
            }
            _ => panic!("{:?} is not a logical operator", operator),
        })
    }

    fn call(callee: Expr, paren: Token, args: Vec<Expr>) -> Expr {
        let callee = Box::new(callee);
        Expr::Call(Call { callee, paren, args })
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
            Identifier(ident) => write!(f, "{}", ident),
            Assign(ident, expr) => write!(f, "(set {} {})", ident, expr),
            Logical(l) => write!(f, "({})", l),
            Call(c) => write!(f, "{}", c),
        }
    }
}

type StatementB = Box<Statement>;

#[derive(Debug)]
pub enum Statement {
    Expr(Expr),
    Print(Expr),
    Block(Vec<Declaration>),
    If(Expr, StatementB, Option<StatementB>),
    While(Expr, StatementB),
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::Expr(expr) => write!(f, "{}", expr),
            Statement::Print(operand) => write!(f, "(print {})", operand),
            Statement::Block(decls) => {
                write!(f, "(block\n")?;
                for decl in decls {
                    writeln!(f, "{}", decl)?;
                }
                write!(f, ")")
            }
            Statement::If(cond, if_branch, else_branch) => {
                write!(f, "(if {} {}", cond, if_branch)?;
                if let Some(else_branch) = else_branch {
                    write!(f, " {}", else_branch)?;
                }
                write!(f, ")")
            }
            Statement::While(cond, stmt) => {
                write!(f, "(while {} {}", cond, stmt)
            }
        }
    }
}

#[derive(Debug)]
pub struct Identifier(Token);
impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.lexeme)
    }
}

#[derive(Debug)]
pub enum Declaration {
    VarDecl(Identifier, Expr),
    Statement(Statement),
    FunDecl(Identifier, Vec<Identifier>, Vec<Declaration>)
}

impl fmt::Display for Declaration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Declaration::VarDecl(ident, expr) => write!(f, "(var {} {})", ident, expr),
            Declaration::Statement(stmt) => write!(f, "{}", stmt),
            Declaration::FunDecl(name, params, body) => write!(f, "(fun {} {} {})", name, DisplayVec(params), DisplayVec(body)),
        }
    }
}

#[derive(Debug)]
pub struct AST {
    declarations: Vec<Declaration>,
}

impl fmt::Display for AST {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for statement in &self.declarations {
            write!(f, "{}\n", statement)?;
        }
        Ok(())
    }
}

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    line: usize,
}

type ParseMethod<T> = fn(&mut Parser) -> Result<T, Error>;

// Parsing is essentially the same as scanning, except every character is now a token.
impl Parser {
    fn new() -> Parser {
        let current = 0;
        let line = 1;
        let tokens = Vec::new();
        Parser { tokens, current, line }
    }

    fn current(&mut self) -> Token {
        if self.at_end() {
            return Token {
                type_: TokenType::EOF,
                lexeme: "".to_string(),
                literal: Object::Nil,
                line: self.line,
            };
        }

        self.tokens[self.current].clone()
    }

    fn previous(&self) -> Token {
        self.tokens[self.current - 1].clone()
    }

    fn advance(&mut self) -> Token {
        self.line = self.tokens[self.current].line;
        self.current += 1;
        self.previous()
    }

    fn backup(&mut self) {
        self.current -= 1;
        self.line = self.tokens[self.current].line;
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

    fn consume(&mut self, type_: TokenType) -> Result<Token, Error> {
        let tok = self.current();
        if tok.type_ == type_ {
            Ok(self.advance())
        } else {
            Err(Error::new(
                tok.line,
                "".to_string(),
                format!("expected {:?} got {:?}", type_, tok.type_),
            ))
        }
    }

    fn parse_binop_expr(
        &mut self,
        types: &[TokenType],
        f: ParseMethod<Expr>,
        g: fn(Expr, Token, Expr) -> Expr,
    ) -> Result<Expr, Error> {
        let mut expr = f(self)?;
        while self.advance_if(types) {
            let operator = self.previous();
            let right = f(self)?;
            expr = g(expr, operator, right);
        }
        Ok(expr)
    }

    // ------------------------ Begin parsing ------------------------

    fn parse_tokens(&mut self, mut tokens: Vec<Token>) -> Result<AST, Error> {
        self.tokens.append(&mut tokens);
        self.parse_program()
    }

    fn parse_program(&mut self) -> Result<AST, Error> {
        let mut declarations = Vec::new();
        while !self.at_end() {
            declarations.push(self.declaration()?);
        }

        Ok(AST { declarations })
    }

    fn declaration(&mut self) -> Result<Declaration, Error> {
        if self.advance_if(&[TokenType::VAR]) {
            self.parse_var()
        } else if self.advance_if(&[TokenType::FUN]) {
            self.parse_fun()
        } else {
            Ok(Declaration::Statement(self.statement()?))
        }
    }

    fn parse_var(&mut self) -> Result<Declaration, Error> {
        use TokenType::*;
        self.consume(IDENTIFIER)?;
        let ident = self.previous();
        let expr =
            if self.advance_if(&[EQUAL]) { self.expression()? } else { Expr::Literal(Object::Nil) };
        self.consume(SEMICOLON)?;

        Ok(Declaration::VarDecl(Identifier(ident), expr))
    }

    fn identifier(&mut self) -> Result<Identifier, Error> {
        Ok(Identifier(self.consume(TokenType::IDENTIFIER)?))
    }

    fn parse_fun(&mut self) -> Result<Declaration, Error> {
        use TokenType::*;

        let name = self.identifier()?;
        self.consume(LEFT_PAREN)?;
        let mut parameters = Vec::new();
        if self.current().type_ != RIGHT_PAREN {
            parameters.push(self.identifier()?);
            while self.advance_if(&[COMMA]) {
                parameters.push(self.identifier()?);
                if parameters.len() >= 255 {
                    todo!();
                }
            }
        }

        self.consume(RIGHT_PAREN)?;

        self.consume(LEFT_BRACE)?;
        let mut body = Vec::new();
        while !self.advance_if(&[RIGHT_BRACE]) && !self.at_end() {
            body.push(self.declaration()?);
        }


        Ok(Declaration::FunDecl(name, parameters, body))
    }

    fn statement(&mut self) -> Result<Statement, Error> {
        use TokenType::*;
        if self.advance_if(&[PRINT]) {
            let operand = self.expression()?;
            self.consume(SEMICOLON)?;

            Ok(Statement::Print(operand))
        } else if self.advance_if(&[LEFT_BRACE]) {
            let mut statements = Vec::new();

            while !self.advance_if(&[RIGHT_BRACE]) && !self.at_end() {
                statements.push(self.declaration()?);
            }

            Ok(Statement::Block(statements))
        } else if self.advance_if(&[IF]) {
            self.consume(LEFT_PAREN)?;
            let cond = self.expression()?;
            self.consume(RIGHT_PAREN)?;
            let if_branch = Box::new(self.statement()?);

            let else_branch =
                if self.advance_if(&[ELSE]) { Some(Box::new(self.statement()?)) } else { None };

            Ok(Statement::If(cond, if_branch, else_branch))
        } else if self.advance_if(&[WHILE]) {
            self.consume(LEFT_PAREN)?;
            let cond = self.expression()?;
            self.consume(RIGHT_PAREN)?;
            let stmt = self.statement()?;
            Ok(Statement::While(cond, Box::new(stmt)))
        } else {
            let expr = self.expression()?;
            self.consume(SEMICOLON)?;
            Ok(Statement::Expr(expr))
        }
    }

    fn expression(&mut self) -> Result<Expr, Error> {
        self.assignment()
    }

    fn assignment(&mut self) -> Result<Expr, Error> {
        use TokenType::*;

        if self.advance_if(&[IDENTIFIER]) {
            let ident = self.previous();
            if self.advance_if(&[EQUAL]) {
                let expr = self.assignment()?;
                return Ok(Expr::Assign(Identifier(ident), Box::new(expr)));
            }
            self.backup();
        }
        self.logic_or()
    }

    // logic_or = logic_and ("or" logic_and)*
    fn logic_or(&mut self) -> Result<Expr, Error> {
        self.parse_binop_expr(&[TokenType::OR], Self::logic_and, Expr::logical)
    }

    fn logic_and(&mut self) -> Result<Expr, Error> {
        self.parse_binop_expr(&[TokenType::AND], Self::equality, Expr::logical)
    }

    fn equality(&mut self) -> Result<Expr, Error> {
        use TokenType::*;
        self.parse_binop_expr(&[EQUAL_EQUAL, BANG_EQUAL], Self::comparison, Expr::binary)
    }

    fn comparison(&mut self) -> Result<Expr, Error> {
        use TokenType::*;
        self.parse_binop_expr(&[GREATER, GREATER_EQUAL, LESS, LESS_EQUAL], Self::term, Expr::binary)
    }

    fn term(&mut self) -> Result<Expr, Error> {
        use TokenType::*;
        self.parse_binop_expr(&[PLUS, MINUS], Self::factor, Expr::binary)
    }

    fn factor(&mut self) -> Result<Expr, Error> {
        use TokenType::*;
        self.parse_binop_expr(&[SLASH, STAR, PERCENT], Self::unary, Expr::binary)
    }

    fn unary(&mut self) -> Result<Expr, Error> {
        use TokenType::*;

        if self.advance_if(&[BANG, MINUS]) {
            let operator = self.previous();
            let right = self.unary()?;
            Ok(Expr::unary(operator, right))
        } else {
            self.parse_call()
        }
    }

    fn parse_call(&mut self) -> Result<Expr, Error> {
        use TokenType::*;
        let mut expr = self.primary()?;
        loop {
            if self.advance_if(&[LEFT_PAREN]) {
                let paren = self.previous();
                let mut args = Vec::new();
                if self.advance_if(&[RIGHT_PAREN]) {
                    expr = Expr::call(expr, self.previous(), args);
                } else {
                    args.push(self.expression()?);
                    while self.advance_if(&[COMMA]) {
                        args.push(self.expression()?);
                        if args.len() >= 255 {
                            return Err(Error::new(
                                paren.line,
                                "".to_string(),
                                "cannot have more then 255 arguments".to_string(),
                            ));
                        }
                    }
                    self.consume(RIGHT_PAREN)?;
                    expr = Expr::call(expr, paren, args);
                }
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn primary(&mut self) -> Result<Expr, Error> {
        use TokenType::*;
        // TODO: Rewrite with match

        if self.advance_if(&[NUMBER, STRING, TRUE, FALSE, NIL]) {
            Ok(Expr::literal(self.previous()))
        } else if self.advance_if(&[LEFT_PAREN]) {
            let expr = self.expression()?;
            self.consume(RIGHT_PAREN)?;
            Ok(Expr::grouping(expr))
        } else if self.advance_if(&[IDENTIFIER]) {
            Ok(Expr::Identifier(Identifier(self.previous())))
        } else {
            let tok = self.current();
            Err(Error::new(
                tok.line,
                "".to_string(),
                format!("Expected expression, got {:?}", tok.type_),
            ))
        }
    }

}

use std::collections::HashMap;

#[derive(Debug)]
struct Environment {
    scopes: Vec<HashMap<String, Object>>,
}

impl Environment {
    fn new() -> Environment {
        let values = HashMap::new();
        let scopes = vec![values];
        Environment { scopes }
    }

    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop(&mut self) {
        self.scopes.pop();
    }

    fn define(&mut self, key: String, value: Object) {
        // var a = 1; var a = 2; is fine, the second declaration shadows the first.
        self.scopes.last_mut().unwrap().insert(key, value);
    }

    fn set(&mut self, key: String, value: Object) -> bool {
        for scope in self.scopes.iter_mut().rev() {
            if scope.get(&key).is_some() {
                scope.insert(key, value);
                return true;
            }
        }
        return false;
    }

    fn get(&self, key: &str) -> Option<&Object> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(key) {
                return Some(v);
            }
        }
        return None;
    }
}

pub trait Callable {
    fn arity(&self) -> usize;
    fn call(&self, args: Vec<Object>) -> Result<Object, Error>;
}

impl fmt::Debug for dyn Callable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<Callable>")
    }
}

impl PartialEq for dyn Callable {
    fn eq(&self, _other: &Self) -> bool {
        unreachable!()
    }
}

struct Clock();
impl Callable for Clock {
    fn arity(&self) -> usize {
        0
    }
    fn call(&self, _args: Vec<Object>) -> Result<Object, Error> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let start = SystemTime::now();
        let since_epoch = start.duration_since(UNIX_EPOCH).unwrap();
        let ms = since_epoch.as_millis();
        Ok(Object::Number(ms as f64))
    }
}

struct Interpreter {
    env: Box<Environment>,
}

impl Interpreter {
    fn new() -> Interpreter {
        let mut env = Box::new(Environment::new());
        env.define("clock".to_string(), Object::Callable(Rc::new(Box::new(Clock()))));
        Interpreter { env }
    }

    fn interpret(&mut self, ast: AST) -> Result<Object, Error> {
        let mut last = Object::Nil;
        for decl in ast.declarations {
            last = self.interpret_declaration(&decl)?;
        }
        Ok(last)
    }

    fn interpret_declaration(&mut self, decl: &Declaration) -> Result<Object, Error> {
        match decl {
            Declaration::Statement(stmt) => self.interpret_statement(stmt),
            Declaration::VarDecl(ident, expr) => {
                let name = &ident.0.lexeme.clone();
                let value = self.eval_expr(expr)?;
                self.env.define(name.into(), value);
                Ok(Object::Nil)
            }
            Declaration::FunDecl(name, params, body) => todo!(),
        }
    }

    fn interpret_statement(&mut self, statement: &Statement) -> Result<Object, Error> {
        match statement {
            Statement::Expr(expr) => self.eval_expr(expr),
            Statement::Print(operand) => self.print_statement(operand),
            Statement::Block(block) => self.block_statement(block),
            Statement::If(cond, if_branch, else_branch) => {
                if self.eval_expr(cond)?.bool()? {
                    self.interpret_statement(&if_branch)
                } else {
                    if let Some(else_branch) = else_branch {
                        self.interpret_statement(&else_branch)
                    } else {
                        Ok(Object::Nil)
                    }
                }
            }
            Statement::While(cond, stmt) => {
                while self.eval_expr(cond)?.bool()? {
                    self.interpret_statement(&stmt)?;
                }
                Ok(Object::Nil)
            }
        }
    }

    fn block_statement(&mut self, statements: &Vec<Declaration>) -> Result<Object, Error> {
        let mut last = Object::Nil;

        self.env.push();
        for stmt in statements {
            last = self.interpret_declaration(stmt)?;
        }
        self.env.pop();

        Ok(last)
    }

    fn print_statement(&mut self, operand: &Expr) -> Result<Object, Error> {
        use std::io::{stdout, Write};

        let obj = self.eval_expr(operand)?;
        if let Object::String(s) = obj {
            print!("{}", s);
        } else {
            print!("{}", obj);
        }
        stdout().flush().expect("failed to flush stdout");

        Ok(Object::Nil)
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<Object, Error> {
        use Expr::*;

        Ok(match expr {
            Literal(l) => l.clone(),
            Unary(u) => self.eval_unary(u)?,
            Binary(b) => self.eval_binary(b)?,
            Grouping(g) => self.eval_expr(&g)?,
            Identifier(i) => self.eval_identifier(i)?,
            Assign(i, e) => self.eval_assign(i, e)?,
            Logical(l) => self.eval_logical(l)?,
            Call(c) => self.eval_call(c)?,
        })
    }

    fn eval_call(&mut self, c: &Call) -> Result<Object, Error> {
        let callee = self.eval_expr(&c.callee)?;
        if let Object::Callable(f) = callee {
            if f.arity() != c.args.len() {
                Err(Error::new(c.paren.line, "".to_string(), format!("want {} arguments got {}", f.arity(), c.args.len())))
            } else {
                let mut args = Vec::with_capacity(c.args.len());
                for arg in &c.args {
                    args.push(self.eval_expr(arg)?);
                }

                f.call(args)
            }
        } else {
            Err(Error::new(c.paren.line, "".to_string(), format!("want Callable got {:?}", callee)))
        }
    }

    fn eval_binary(&mut self, b: &Binary) -> Result<Object, Error> {
        use TokenType::*;

        let left = self.eval_expr(&b.left)?;
        let right = self.eval_expr(&b.right)?;

        let res = match b.operator.type_ {
            PLUS => left.add(right),
            STAR => left.mul(right),
            SLASH => left.div(right),
            PERCENT => left.modulo(right),
            MINUS => left.sub(right),
            EQUAL_EQUAL => Ok(Object::Bool(left == right)),
            BANG_EQUAL => Ok(Object::Bool(left != right)),
            LESS_EQUAL => left.less_equal(right),
            GREATER_EQUAL => left.greater_equal(right),
            GREATER => left.greater(right),
            LESS => left.less(right),

            _ => panic!("invalid binary operator '{}'", b.operator.lexeme),
        };

        self.fix_res(&b.operator, res)
    }

    fn eval_unary(&mut self, u: &Unary) -> Result<Object, Error> {
        use TokenType::*;

        let right = self.eval_expr(&u.right)?;
        let res = match u.operator.type_ {
            MINUS => right.neg(),
            BANG => right.not(),
            // this is a compiler bug, so we panic
            _ => panic!("invalid unary operator '{}'", u.operator.lexeme),
        };

        self.fix_res(&u.operator, res)
    }

    fn eval_identifier(&self, ident: &Identifier) -> Result<Object, Error> {
        self.getenv(ident)
    }

    fn eval_assign(&mut self, ident: &Identifier, expr: &Expr) -> Result<Object, Error> {
        let value = self.eval_expr(&expr)?;
        self.setenv(ident, value)
    }

    fn eval_logical(&mut self, l: &Logical) -> Result<Object, Error> {
        match l.operator.type_ {
            TokenType::OR => self.eval_or(l),
            TokenType::AND => self.eval_and(&l),
            _ => panic!("invalid logical operator {:?}", l),
        }
    }

    fn eval_or(&mut self, l: &Logical) -> Result<Object, Error> {
        let left = self.eval_expr(&l.left)?;

        match left {
            Object::Bool(b) => {
                if b {
                    Ok(Object::Bool(true))
                } else {
                    self.eval_expr(&l.right)
                }
            }
            _ => Err(Error::new(
                l.operator.line,
                "".to_string(),
                format!("want bool got {:?}", left),
            )),
        }
    }

    fn eval_and(&mut self, l: &Logical) -> Result<Object, Error> {
        let left = self.eval_expr(&l.left)?;

        match left {
            Object::Bool(b) => {
                if b {
                    self.eval_expr(&l.right)
                } else {
                    Ok(Object::Bool(false))
                }
            }
            _ => Err(Error::new(
                l.operator.line,
                "".to_string(),
                format!("want bool got {:?}", left),
            )),
        }
    }

    fn setenv(&mut self, ident: &Identifier, value: Object) -> Result<Object, Error> {
        let name = &ident.0.lexeme;
        if !self.env.set(name.clone(), value.clone()) {
            Err(Error::new(
                ident.0.line,
                "".to_string(),
                format!("Attempt to assign undefined variable '{}'", name),
            ))
        } else {
            Ok(value)
        }
    }

    fn getenv(&self, ident: &Identifier) -> Result<Object, Error> {
        match self.env.get(&ident.0.lexeme) {
            Some(v) => Ok(v.clone()),
            None => Err(Error::new(
                ident.0.line,
                "".to_string(),
                format!("Undefined variable '{}'", ident.0.lexeme),
            )),
        }
    }

    fn fix_res(&self, op: &Token, res: Result<Object, Error>) -> Result<Object, Error> {
        match res {
            Ok(obj) => Ok(obj),
            Err(mut err) => {
                err.line = op.line;
                Err(err)
            }
        }
    }
}

pub struct Lox {
    interpreter: Interpreter,
    pub repl: bool,
}

impl Lox {
    pub fn new() -> Lox {
        Lox { interpreter: Interpreter::new(), repl: false }
    }

    pub fn run<S: Into<String>>(&mut self, source: S) -> Result<Object, Error> {
        let tokens = self.scan(source)?;
        let ast = self.parse(tokens)?;
        if self.repl {
            eprintln!("{}", ast);
        }
        let result = self.interpreter.interpret(ast)?;
        if self.repl {
            eprintln!("=> {}", result);
        }

        Ok(result)
    }

    pub fn scan<S: Into<String>>(&mut self, source: S) -> Result<Vec<Token>, Error> {
        Scanner::new().scan_tokens(source.into())
    }

    pub fn parse(&mut self, tokens: Vec<Token>) -> Result<AST, Error> {
        Parser::new().parse_tokens(tokens)
    }

    // Parse and evaluate a single expression
    pub fn eval(&mut self, source: &str) -> Result<Object, Error> {
        let mut s = source.to_string();
        s.push(';');
        let tokens = self.scan(s)?;
        let ast = self.parse(tokens)?;
        assert!(ast.declarations.len() == 1);
        self.interpreter.interpret(ast)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_expr() {
        fn s(source: &str) -> String {
            let mut lox = Lox::new();
            let tokens = lox.scan(source).unwrap();
            let ast = lox.parse(tokens).unwrap();
            format!("{}", ast).trim_end().to_string()
        }

        assert_eq!(s("2 + 3;"), "(+ 2 3)");
        assert_eq!(s("2 + 3*4;"), "(+ 2 (* 3 4))");
        assert_eq!(s("2 + 3 + 4;"), "(+ (+ 2 3) 4)");
        assert_eq!(s("2 >= 3 + 4*2;"), "(>= 2 (+ 3 (* 4 2)))");
        assert_eq!(s("2 >= (3 + 4)*2;"), "(>= 2 (* (group (+ 3 4)) 2))");
        assert_eq!(s("print 3 >= 2;"), "(print (>= 3 2))");
        assert_eq!(s("print 3 >= 2; print 5+3;"), "(print (>= 3 2))\n(print (+ 5 3))");
    }

    #[test]
    fn test_interpreter_eval() {
        fn e(source: &str) -> Object {
            Lox::new().eval(source).unwrap()
        }

        assert_eq!(e("2"), Object::Number(2.));
        assert_eq!(e("-2"), Object::Number(-2.));
        assert_eq!(e("!false"), Object::Bool(true));
        assert_eq!(e("2 + 2"), Object::Number(4.));
        assert_eq!(e("(2 + 3) * 4"), Object::Number(20.));
        assert_eq!(e("(2 + 6) / 4"), Object::Number(2.));
        assert_eq!(e("(2 + 6) == 8"), Object::Bool(true));
        assert_eq!(e("!((2 + 6) == 8)"), Object::Bool(false));
        assert_eq!(e("\"foo\" + \"bar\""), Object::String("foobar".to_string()));
        assert_eq!(e("3 < 5"), Object::Bool(true));
        assert_eq!(e("3 >= 3"), Object::Bool(true));
        assert_eq!(e("3 > 3"), Object::Bool(false));
        assert_eq!(e("3 != 3"), Object::Bool(false));
        assert_eq!(e("(3 + 2) == 5"), Object::Bool(true));
    }

    #[test]
    fn test_interpreter() {
        fn r(source: &str) -> Object {
            let mut lox = Lox::new();
            lox.run(source).unwrap()
        }

        assert_eq!(r("var x = 5; var y = 3; x + y;"), Object::Number(8.));
        assert_eq!(
            r("var x = \"foo\"; var y = \"bar\"; x + y;"),
            Object::String("foobar".to_string())
        );
        assert_eq!(r("var x; x;"), Object::Nil);
        assert_eq!(r("var x = 5; x = 2; x;"), Object::Number(2.));
        assert_eq!(r("var x = 5; var y; x = y = 2; y+x;"), Object::Number(4.));
        r("clock();");
    }

    #[test]
    fn test_scope() {
        let mut lox = Lox::new();
        assert_eq!(lox.run("var x = 5; { var x = 3; x; }"), Ok(Object::Number(3.)));
        assert_eq!(lox.run("x;"), Ok(Object::Number(5.)));
    }

    #[test]
    fn test_scope_assign() {
        let mut lox = Lox::new();
        assert_eq!(lox.run("var x = 5; { x = 3; x; }"), Ok(Object::Number(3.)));
        assert_eq!(lox.run("x;"), Ok(Object::Number(3.)));
    }

    #[test]
    fn test_scope_wierd() {
        let mut lox = Lox::new();
        assert_eq!(lox.run("var a = 1; { var a = a + 2; a; }"), Ok(Object::Number(3.)));
    }

    #[test]
    fn test_conditionals() {
        let mut lox = Lox::new();
        assert_eq!(lox.run("var x = 5; if (x == 5) { 3; }"), Ok(Object::Number(3.)));

        assert_eq!(lox.run("var x = 5; if (x == 4) { 3; } else { 4; }"), Ok(Object::Number(4.)));

        // else is bound to closest if that precedes it. if you write multiple if-else's on
        // a single line god help you
        assert_eq!(
            lox.run(
                "
                var x = 5;
                if (x == 5)
                    if (x-1 == 6)
                        6;
                    else
                        3;
                "
            ),
            Ok(Object::Number(3.))
        );
    }

    #[test]
    fn test_logical_or() {
        let mut lox = Lox::new();
        assert_eq!(lox.run("true or false;"), Ok(Object::Bool(true)));
        assert_eq!(lox.run("true or true;"), Ok(Object::Bool(true)));
        assert_eq!(lox.run("false or true;"), Ok(Object::Bool(true)));
        assert_eq!(lox.run("false or false;"), Ok(Object::Bool(false)));
    }

    #[test]
    fn test_logical_and() {
        let mut lox = Lox::new();
        assert_eq!(lox.run("true and false;"), Ok(Object::Bool(false)));
        assert_eq!(lox.run("false and true;"), Ok(Object::Bool(false)));
        assert_eq!(lox.run("false and false;"), Ok(Object::Bool(false)));
        assert_eq!(lox.run("true and true;"), Ok(Object::Bool(true)));
    }

    #[test]
    fn test_looping() {
        let mut lox = Lox::new();
        assert_eq!(lox.run("var x = 5;"), Ok(Object::Nil));
        assert_eq!(lox.run("while (x < 10) { x = x + 1; }"), Ok(Object::Nil));
        assert_eq!(lox.run("x;"), Ok(Object::Number(10.)));
    }

    #[test]
    fn test_modulo() {
        let mut lox = Lox::new();
        assert_eq!(lox.run("4 % 2;"), Ok(Object::Number(0.)));
        assert_eq!(lox.run("5 % 2;"), Ok(Object::Number(1.)));
        assert_eq!(lox.run("7 % 4;"), Ok(Object::Number(3.)));
    }

    #[test]
    fn test_lex_string() {
        let tokens = Scanner::new().scan_tokens("\"foo\\nbar\"".into()).unwrap();
        assert_eq!(tokens[0].literal, Object::String("foo\nbar".into()));

        let tokens = Scanner::new().scan_tokens("\"foo\\\\nbar\"".into()).unwrap();
        assert_eq!(tokens[0].literal, Object::String("foo\\nbar".into()));
    }

    #[test]
    fn test_parse_functions() {
        fn s(source: &str) -> String {
            let mut lox = Lox::new();
            let tokens = lox.scan(source).unwrap();
            let ast = lox.parse(tokens).unwrap();
            format!("{}", ast).trim_end().to_string()
        }

        assert_eq!(s("foo();"), "(call foo [])");
        assert_eq!(s("foo(1, 2);"), "(call foo [1, 2])");
        assert_eq!(s("foo(1)(2);"), "(call (call foo [1]) [2])");
        assert_eq!(s("fun foo() {}"), "(fun foo [] [])");
        assert_eq!(s("fun foo(x, y) { print x + y; }"), "(fun foo [x, y] [(print (+ x y))])");
    }

}
