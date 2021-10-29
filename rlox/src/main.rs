use rlox::Lox;
use std::io::BufRead;

fn main() {
    let mut lox = Lox::new();

    let stdin = std::io::stdin();
    eprint!("> ");
    for line in stdin.lock().lines() {
        let source = line.unwrap();
        let source = source.to_string() + "\n";
        if let Err(e) = lox.run(&source) {
            eprintln!("{}", e);
        };
        eprint!("> ");
    }
}
