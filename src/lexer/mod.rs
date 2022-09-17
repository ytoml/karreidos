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
                let tok = &self.src[start..pos];
                // We know it's alreadly confirmed tok is non-empty and valid thus can be unwrapped.
                Ok(tok.try_into().unwrap())
            }
            c if available_in_num(c) => {
                if c == '.' && Some(&'.') == chars.peek() {
                    let _ = chars.next().unwrap();
                    pos += 1;
                    Ok(Token::Double(".."))
                } else {
                    loop {
                        match chars.peek() {
                            Some('.')
                                if pos + 1 < self.src.len() && &self.src[pos..pos + 2] == ".." =>
                            {
                                // ".." will be parsed next time.
                            }
                            Some(&c) if available_in_num(c) => {
                                let _ = chars.next().unwrap();
                                pos += 1;
                                continue;
                            }
                            _ => {}
                        }
                        break Ok(Token::Num(self.src[start..pos].parse().unwrap()));
                    }
                }
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
            '<' if matches!(chars.peek(), Some(&'-')) => {
                let _ = chars.next().unwrap();
                pos += 1;
                Ok(Token::Double("<-"))
            }
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
    fn comment() {
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

    #[test]
    fn forloop() {
        const SRC: &str = r"
            fn main()
                for mut i <- 1.0..10.2, 0.1 {
                    foo();
                };
        ";
        let expected_tokens = vec![
            Token::Fn,
            Token::Ident("main".to_string()),
            Token::Single('('),
            Token::Single(')'),
            Token::For,
            Token::Mut,
            Token::Ident("i".to_string()),
            Token::Double("<-"),
            Token::Num(1.0),
            Token::Double(".."),
            Token::Num(10.2),
            Token::Single(','),
            Token::Num(0.1),
            Token::Single('{'),
            Token::Ident("foo".to_string()),
            Token::Single('('),
            Token::Single(')'),
            Token::Single(';'),
            Token::Single('}'),
            Token::Single(';'),
        ];
        expect_success(SRC, expected_tokens)
    }
}
