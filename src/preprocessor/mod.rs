pub fn _preprocess(src: &str) -> String {
    // currently, just define main.
    let mut source = "extern main()\n".to_string();
    source.push_str(src);
    source
}
