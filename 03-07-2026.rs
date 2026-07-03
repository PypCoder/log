fn parse_log_line(line: &str) -> Result<String, String> {
    match line.find('=') {
        Some(index) => {
            let value = &line[index + 1..];
            Ok(String::from(value))
        }
        None => Err(String::from("Corrupted log line: missing '=' delimiter")),
    }
}

fn main() {
    let good_line = "DB_HOST=127.0.0.1";
    let bad_line = "INVALID_LOG_ENTRY_WITHOUT_SIGN";

    println!("Parsing good line: {:?}", parse_log_line(good_line));
    println!("Parsing bad line: {:?}", parse_log_line(bad_line));
}
