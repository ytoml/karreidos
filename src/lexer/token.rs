use crate::error::info::SrcInfo;

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Eof,
    Fn,
    Extern,
    If,
    Else,
    For,
    Let,
    Mut,
    Ident(String),
    Num(f64),
    Comment,
    // Currently only single char operator supported.
    // Also, Single works for other tokens like braces or brackets.
    Single(char),
    Double(&'static str),
}
info_impl!(Token, token);

impl<'a> TryFrom<&'a str> for Token {
    type Error = &'a str;
    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        match value {
            "fn" => Ok(Self::Fn),
            "extern" => Ok(Self::Extern),
            "if" => Ok(Self::If),
            "else" => Ok(Self::Else),
            "for" => Ok(Self::For),
            "let" => Ok(Self::Let),
            "mut" => Ok(Self::Mut),
            s if valid_as_ident(s) => Ok(Self::Ident(s.to_string())),
            s => Err(s),
        }
    }
}

fn valid_as_ident(s: &str) -> bool {
    matches!(
        s.chars().peekable().peek().copied(),
        Some(c) if can_be_head_of_ident(c)
    ) && s.chars().all(available_in_ident)
}

pub fn available_in_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

pub fn can_be_head_of_ident(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

/// Only supports 0..=9 and `.`. Hex or exp is not currently supported.
pub fn available_in_num(c: char) -> bool {
    c.is_ascii_digit() || c == '.'
}
