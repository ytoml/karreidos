use std::{iter::Peekable, ops::DerefMut, str::Chars};

pub mod token;

use self::token::{available_in_ident, available_in_num, can_be_head_of_ident, Token};

#[derive(Debug)]
pub enum LexError {
    BlockCommentNotTerminated,
}

pub type Result<T> = std::result::Result<T, LexError>;
type LexResult = Result<Token>;

#[derive(Debug)]
pub struct Lexer<'a> {
    src: &'a str,
    chars: Box<Peekable<Chars<'a>>>,
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Self {
            src,
            chars: Box::new(src.chars().peekable()),
            pos: 0,
        }
    }

    fn get_token(&mut self) -> LexResult {
        let chars = self.chars.deref_mut();
        let mut pos = self.pos;

        // read until non-whitespace or EOF
        loop {
            if let Some(c) = chars.peek() {
                if !c.is_whitespace() {
                    break;
                }
                let _ = chars.next().unwrap();
            } else {
                self.pos = pos;
                return Ok(Token::Eof);
            }
            pos += 1;
        }

        // know that next non-whitespace exists here.
        let start = pos;
        pos += 1;
        let result = match chars.next().unwrap() {
            c if can_be_head_of_ident(c) => {
                loop {
                    match chars.peek() {
                        Some(&c) if available_in_ident(c) => {
                            // valid for identifier
                            let _ = chars.next().unwrap();
                            pos += 1;
                        }
                        Some(_) | None => break,
                    }
                }
                let tok = match &self.src[start..pos] {
                    "fn" => Token::Fn,
                    "extern" => Token::Extern,
                    tok => Token::Ident(tok.to_string()),
                };
                Ok(tok)
            }
            c if available_in_num(c) => {
                while matches!(chars.peek(), Some(&c) if available_in_num(c)) {
                    let _ = chars.next().unwrap();
                    pos += 1;
                }
                Ok(Token::Num(self.src[start..pos].parse().unwrap()))
            }
            // Can be div/comment/block comment (or div-assign in future)
            '/' => match chars.peek() {
                Some('/') => {
                    loop {
                        match chars.peek() {
                            Some(&c) => {
                                let _ = chars.next().unwrap();
                                pos += 1;
                                if c == '\n' {
                                    break;
                                }
                            }
                            None => {
                                // Eof must be on next call.
                                return Ok(Token::Comment);
                            }
                        }
                    }
                    Ok(Token::Comment)
                }
                Some('*') => {
                    let _ = chars.next().unwrap();
                    pos += 1;
                    loop {
                        match chars.next() {
                            Some('*') => {
                                pos += 1;
                                if let Some(&'/') = chars.peek() {
                                    let _ = chars.next().unwrap();
                                    pos += 1;
                                    break;
                                }
                            }
                            Some(_) => pos += 1,
                            None => return Err(LexError::BlockCommentNotTerminated),
                        }
                    }
                    Ok(Token::Comment)
                }
                Some(_) | None => Ok(Token::Single('/')),
            },
            c => Ok(Token::Single(c)),
        };
        self.pos = pos;
        result
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = LexResult;
    fn next(&mut self) -> Option<Self::Item> {
        match self.get_token() {
            Ok(Token::Eof) => None,
            res => Some(res),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn expect_success(src: &'static str, expected_tokens: Vec<Token>) {
        let result: Result<Vec<Token>> = Lexer::new(src).into_iter().collect();
        assert_eq!(result.expect("lex failed."), expected_tokens,);
    }

    #[test]
    fn test() {
        const SRC: &str = r"
            fn main(x, y)
            x = 0.02320840394834;
            /*
            hello, this is comment.
            */
            y = (1 + 2) - 1 * 3;
        ";
        let expected_tokens = vec![
            Token::Fn,
            Token::Ident("main".to_string()),
            Token::Single('('),
            Token::Ident('x'.to_string()),
            Token::Single(','),
            Token::Ident('y'.to_string()),
            Token::Single(')'),
            Token::Ident('x'.to_string()),
            Token::Single('='),
            Token::Num(0.02320840394834),
            Token::Single(';'),
            Token::Comment,
            Token::Ident('y'.to_string()),
            Token::Single('='),
            Token::Single('('),
            Token::Num(1.0),
            Token::Single('+'),
            Token::Num(2.0),
            Token::Single(')'),
            Token::Single('-'),
            Token::Num(1.0),
            Token::Single('*'),
            Token::Num(3.0),
            Token::Single(';'),
        ];
        expect_success(SRC, expected_tokens)
    }
}
