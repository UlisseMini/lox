use rlox::Lox;

fn main() {
    let mut lox = Lox::new();
    let returncode = lox.run("print 'hello'");
    std::process::exit(returncode);
}
