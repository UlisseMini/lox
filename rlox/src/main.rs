use rlox::Lox;
use std::io::BufRead;

fn main() {
    let mut lox = Lox::new();

    let stdin = std::io::stdin();
    eprint!("> ");
    for line in stdin.lock().lines() {
        let source = line.unwrap();
        if let Err(e) = lox.run(&source) {
            eprintln!("Error: {}", e);
        };
        eprint!("> ");
    }
}
