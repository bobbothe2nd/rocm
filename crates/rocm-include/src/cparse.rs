pub fn parse_define_const(contents: &str, name: &str) -> Option<u32> {
    contents.lines().find_map(|line| {
        let mut parts = line.split_whitespace();

        if parts.next()? == "#define" && parts.next()? == name {
            parts.next()?.parse().ok()
        } else {
            None
        }
    })
}
