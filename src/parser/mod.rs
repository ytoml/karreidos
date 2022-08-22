pub mod ast;

use std::collections::HashMap;

use self::ast::Expr;
use crate::lexer::token::Token;
use crate::ANONYMOUS_FN_NAME;

pub type Result<T> = std::result::Result<T, ParseError>;

#[derive(Debug)]
pub enum ParseError {
    Unexpected(Token),
    TokenShortage,
    Unimplemented,
    Reserved(&'static str),
}

#[derive(Debug)]
pub struct ProtoType {
    name: String,
    args: Vec<String>,
}
impl ProtoType {
    pub fn num_args(&self) -> usize {
        self.args.len()
    }

    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    pub fn arg_names_slice(&self) -> &[String] {
        self.args.as_slice()
    }
}

#[derive(Debug)]
pub struct Function {
    proto: ProtoType,
    body: Option<Vec<Expr>>,
    is_anonymous: bool,
}
impl Function {
    pub fn proto(&self) -> &ProtoType {
        &self.proto
    }

    pub fn body(&self) -> Option<&[Expr]> {
        self.body.as_deref()
    }

    pub fn is_anonymous(&self) -> bool {
        self.is_anonymous
    }
}

#[derive(Debug)]
pub enum GlobalVar {
    Function(Function),
}

#[derive(Debug)]
pub struct Scope {
    _global_vars: Vec<GlobalVar>,
    _level: usize,
}

#[derive(Debug)]
pub struct Parser {
    tokens: Vec<Token>,
    // for binop precedence, but key is char
    // because [`BinOp`] is created through parsing
    prec: HashMap<char, i32>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>, prec: HashMap<char, i32>) -> Self {
        Self {
            tokens,
            prec,
            pos: 0,
        }
    }

    pub fn parse(&mut self) -> Result<Option<GlobalVar>> {
        if let Some(tok) = self.current() {
            log::debug!("{tok:?}");
            let func = match tok {
                Token::Extern => self.ext()?,
                Token::Fn => self.func()?,
                _ => self.top_level_expr()?,
            };
            if !func.is_anonymous() && func.proto().name() == ANONYMOUS_FN_NAME {
                Err(ParseError::Reserved(ANONYMOUS_FN_NAME))
            } else {
                Ok(Some(GlobalVar::Function(func)))
            }
        } else {
            Ok(None)
        }
    }

    /// ```no_run
    /// top_level_expr := stmt
    /// ```
    fn top_level_expr(&mut self) -> Result<Function> {
        let mut exprs = vec![];
        if let Some(expr) = self.stmt()? {
            exprs.push(expr);
        }
        Ok(Function {
            proto: ProtoType {
                name: ANONYMOUS_FN_NAME.to_string(),
                args: vec![],
            },
            body: Some(exprs),
            is_anonymous: true,
        })
    }

    /// ```no_run
    /// expr := primary bin_rhs
    /// ```
    fn expr(&mut self) -> Result<Expr> {
        let lhs = self.primary()?;
        self.bin_rhs(0, lhs)
    }

    /// ```no_run
    /// stmt := expr ';'
    /// ```
    fn stmt(&mut self) -> Result<Option<Expr>> {
        // TODO: refine BNF
        let _ = self._consume_comment();
        if self.is_single(';')? {
            self.consume_single(';');
            return Ok(None);
        }
        let expr = self.expr()?;
        log::debug!("{expr:?}");
        self.try_consume_single(';')?;
        Ok(Some(expr))
    }

    /// ```no_run
    /// primary := ident_expr
    ///         | num_expr
    ///         | paren_expr
    /// ```
    fn primary(&mut self) -> Result<Expr> {
        match self.expect()? {
            Token::Ident(_) => self.ident_expr(),
            Token::Num(_) => self.num_expr(),
            Token::Single('(') => self.paren_expr(),
            tok => Err(ParseError::Unexpected(tok)),
        }
    }

    fn current_precedence(&self) -> Option<i32> {
        if let Some(Token::Single(op)) = self.current() {
            self.prec.get(&op).copied()
        } else {
            None
        }
    }

    /// ```no_run
    /// bin_rhs := (bin_op primary)*
    /// bin_op := /* refer to BinOp */
    /// ```
    fn bin_rhs(&mut self, precedence: i32, mut lhs: Expr) -> Result<Expr> {
        loop {
            log::debug!("{:?}", self.current());
            let tok_prec = self.current_precedence();
            if matches!(tok_prec, Some(p) if p >= precedence) {
                let tok_prec = tok_prec.unwrap();
                if let Token::Single(op) = self.consume() {
                    let mut rhs = self.primary()?;
                    // in case form of:
                    //     bin_op1 primary bin_op2 primary ..
                    // If prec(bin_op1) < prec(bin_op2), "primary bin_op2 primary .. " must be
                    // merged first, thus regarding "primary bin_op2 primary" as bin_rhs.
                    // Otherwise, we can greedily merge "primary bin_op1 primary" first.
                    if matches!(self.current_precedence(), Some(p) if p > tok_prec) {
                        rhs = self.bin_rhs(tok_prec + 1, rhs)?;
                    }
                    // XXX: currently skipping check of operator, thus might cause panic.
                    lhs = Expr::Binary {
                        op: op.try_into().unwrap(),
                        left: Box::new(lhs),
                        right: Box::new(rhs),
                    };
                } else {
                    unreachable!()
                }
            } else {
                return Ok(lhs);
            }
        }
    }

    /// ```no_run
    /// num_expr := number
    ///
    /// ```
    fn num_expr(&mut self) -> Result<Expr> {
        let expr = match self.tokens[self.pos] {
            Token::Num(val) => Expr::Number(val),
            _ => return Err(ParseError::Unimplemented),
        };
        self.pos += 1;
        Ok(expr)
    }

    /// ```no_run
    /// paren_expr := '(' expr ')'
    ///
    /// ```
    /// # Panics
    /// If current head token is not '('.
    fn paren_expr(&mut self) -> Result<Expr> {
        self.consume_single('(');
        let expr = self.expr()?;
        self.try_consume_single(')')?;
        Ok(expr)
    }

    /// ```no_run
    /// ident_expr := ident
    ///             | ident '(' (expr ',')* (expr)? (',')?  ')'
    /// ```
    fn ident_expr(&mut self) -> Result<Expr> {
        let name = self.try_consume_ident()?;
        if self.is_single('(')? {
            self.pos += 1;
            let mut args = Vec::new();
            loop {
                if self.is_single(')')? {
                    break;
                }
                let expr = self.expr()?;
                args.push(expr);
                if !self.is_single(',')? {
                    break;
                }
                self.consume_single(',');
            }
            self.try_consume_single(')')?;
            Ok(Expr::Call { callee: name, args })
        } else {
            Ok(Expr::Variable(name))
        }
    }

    /// ```no_run
    /// proto := ident '(' (ident ',')* ident ')'
    /// ```
    fn proto(&mut self) -> Result<ProtoType> {
        let name = self.try_consume_ident()?;
        let mut args = vec![];
        self.try_consume_single('(')?;
        loop {
            log::debug!("{:?}", self.current());
            match self.expect()? {
                Token::Ident(name) => {
                    self.pos += 1;
                    args.push(name);
                    if !self.is_single(',')? {
                        self.try_consume_single(')')?;
                        break;
                    }
                    self.consume_single(',')
                }
                Token::Single(')') => {
                    self.pos += 1;
                    break;
                }
                tok => return Err(ParseError::Unexpected(tok)),
            }
        }
        Ok(ProtoType { name, args })
    }

    /// ```no_run
    /// func := "fn" proto stmt*
    /// ```
    /// # Panics
    /// If current head token is not 'fn'.
    fn func(&mut self) -> Result<Function> {
        // TODO: escaping scope of 'fn'
        self.try_consume_fn()?;
        let proto = self.proto()?;
        let mut exprs = vec![];
        loop {
            if self.current().is_none() {
                break;
            }
            match self.stmt() {
                Ok(Some(expr)) => exprs.push(expr),
                res => {
                    log::debug!("{res:?}");
                    res?;
                }
            }
        }
        Ok(Function {
            proto,
            body: Some(exprs),
            is_anonymous: false,
        })
    }

    /// ```no_run
    /// ext := "extern" proto
    /// ```
    fn ext(&mut self) -> Result<Function> {
        self.try_consume_extern().unwrap();
        let proto = self.proto()?;
        Ok(Function {
            proto,
            body: None,
            is_anonymous: false,
        })
    }
}

// Token eating utility functions
impl Parser {
    #[inline]
    fn current(&self) -> Option<Token> {
        (self.pos < self.tokens.len()).then(|| self.tokens[self.pos].clone())
    }

    /// Note that this take [`Token::Eof`] as an error.
    #[inline]
    fn expect(&self) -> Result<Token> {
        self.current()
            .ok_or(ParseError::TokenShortage)
            .and_then(|tok| match tok {
                Token::Eof => Err(ParseError::Unexpected(Token::Eof)),
                tok => Ok(tok),
            })
    }

    // implicitly ignoring comments
    #[inline]
    fn _consume_comment(&mut self) -> Option<Token> {
        match self.current() {
            Some(Token::Comment) => {
                self.pos += 1;
                self._consume_comment()
            }
            option => option,
        }
    }

    #[inline]
    fn forward(&mut self) -> Option<Token> {
        self._consume_comment().map(|t| {
            self.pos += 1;
            t
        })
    }

    #[inline]
    fn try_consume(&mut self) -> Result<Token> {
        self.forward().ok_or(ParseError::TokenShortage)
    }

    /// # Panics
    /// If current token is [Token::Eof].
    #[inline]
    fn consume(&mut self) -> Token {
        self.try_consume().unwrap()
    }

    #[inline]
    fn _try_consume_specified(&mut self, expected_token: Token) -> Result<Token> {
        let tok = self.try_consume()?;
        if tok == expected_token {
            Ok(tok)
        } else {
            Err(ParseError::Unexpected(tok))
        }
    }

    #[inline]
    fn try_consume_single(&mut self, c: char) -> Result<()> {
        self._try_consume_specified(Token::Single(c)).map(|_| ())
    }

    fn consume_single(&mut self, c: char) {
        self.try_consume_single(c).unwrap();
    }

    fn try_consume_fn(&mut self) -> Result<()> {
        self._try_consume_specified(Token::Fn).map(|_| ())
    }

    fn try_consume_extern(&mut self) -> Result<()> {
        self._try_consume_specified(Token::Extern).map(|_| ())
    }

    fn try_consume_ident(&mut self) -> Result<String> {
        let tok = self.try_consume()?;
        if let Token::Ident(name) = tok {
            Ok(name)
        } else {
            Err(ParseError::Unexpected(tok))
        }
    }

    fn is_single(&self, c: char) -> Result<bool> {
        Ok(self.expect()? == Token::Single(c))
    }
}
