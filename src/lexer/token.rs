#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Eof,
    Fn,
    Extern,
    Ident(String),
    Num(f64),
    Comment,
    Single(char), // currently only single char operator supported.
                  // LParen,   // '('
                  // RParen,   // ')'
                  // LCur,     // '{'
                  // RCur,     // '}'
                  // LSq,      // '['
                  // RSq,      // ']'
                  // Comma,
                  // SemiCoron,
}

// impl From<char> for Token {
//     fn from(c: char) -> Self {
//         match c {
//             '(' => Self::LParen,
//             ')' => Self::RParen,
//             '{' => Self::LCur,
//             '}' => Self::RCur,
//             '[' => Self::LSq,
//             ']' => Self::RSq,
//             c => Self::Op(c),
//         }
//     }
// }

pub fn available_in_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

pub fn can_be_head_of_ident(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

/// Only supports 0..=9 and `.`. Hex or exp is not currently supported.
pub fn available_in_num(c: char) -> bool {
    ('0'..='9').contains(&c) || c == '.'
}
