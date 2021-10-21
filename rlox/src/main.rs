use rlox::Lox;

fn main() {
    let mut lox = Lox::new();

    match lox.run("if a != b { 2 + 2 }") {
        Ok(_) => {
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(65);
        }
    };
}
