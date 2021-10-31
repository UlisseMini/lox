use rlox::Lox;
use std::io::BufRead;

fn main() {
    let args = std::env::args();
    if let Some(file) = args.skip(1).next() {
        run_file(file)
    } else {
        repl()
    }
}

fn run_file(path: String) {
    let source = String::from_utf8(std::fs::read(path).unwrap()).unwrap();
    let mut lox = Lox::new();
    lox.run(source).unwrap();
}

fn repl() {
    let mut lox = Lox::new();
    lox.repl = true;

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
