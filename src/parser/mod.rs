pub mod ast;
mod desugar;

use std::collections::{HashMap, VecDeque};
use std::convert::TryInto;

use once_cell::sync::Lazy;

use self::ast::{BinOp, Expr, ExprInfo, VarDecl, VarDeclInfo};
use crate::error::info::SrcInfo;
use crate::lexer::token::{Token, TokenInfo};
use crate::ANONYMOUS_FN_NAME;

pub type Result<T> = std::result::Result<T, ParseError>;

// for binop precedence
static PREC: Lazy<HashMap<BinOp, i32>> = Lazy::new(|| {
    let mut prec = HashMap::with_capacity(7);
    prec.insert(BinOp::Assign, 0);
    prec.insert(BinOp::Pipe, 10);
    prec.insert(BinOp::Lt, 20);
    prec.insert(BinOp::Gt, 20);
    prec.insert(BinOp::Add, 30);
    prec.insert(BinOp::Sub, 30);
    prec.insert(BinOp::Mul, 50);
    prec.insert(BinOp::Div, 50);
    prec
});

#[derive(Debug)]
pub enum ParseError {
    Unexpected(Unexpected),
    TokenShortage,
    Unimplemented,
    NonCallableAtPipeRhs(ExprInfo),
    Reserved(&'static str),
}
impl ParseError {
    #[inline]
    const fn eof() -> Self {
        Self::Unexpected(Unexpected::Eof)
    }

    #[inline]
    const fn unexpected_token(token: Token, info: SrcInfo) -> Self {
        Self::Unexpected(Unexpected::Token(token.with_info(info)))
    }
}

#[derive(Debug)]
pub enum Unexpected {
    Eof,
    Token(TokenInfo),
}

#[derive(Debug, Clone)]
pub struct ProtoType {
    name: String,
    args: Vec<VarDeclInfo>,
}
impl ProtoType {
    #[inline]
    pub fn num_args(&self) -> usize {
        self.args.len()
    }

    #[inline]
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    #[inline]
    pub fn args_slice(&self) -> &[VarDeclInfo] {
        self.args.as_slice()
    }

    #[inline]
    pub fn arg_names_iter(&self) -> impl Iterator<Item = &str> {
        self.args.iter().map(|value| value.var.name())
    }
}
info_impl!(ProtoType, proto);

#[derive(Debug)]
pub struct Function {
    proto: ProtoTypeInfo,
    body: Option<Vec<ExprInfo>>,
    is_anonymous: bool,
}
impl Function {
    #[inline]
    pub fn proto(&self) -> &ProtoTypeInfo {
        &self.proto
    }

    #[inline]
    pub fn body(&self) -> Option<&[ExprInfo]> {
        self.body.as_deref()
    }

    #[inline]
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
    tokens: Vec<TokenInfo>,
    pos: usize,
}

impl Parser {
    #[inline]
    pub fn new(tokens: Vec<TokenInfo>) -> Self {
        Self { tokens, pos: 0 }
    }

    /// ```no_run
    /// program ::= top_level_expr
    ///          | ext
    ///          | func
    /// ```
    pub fn parse(&mut self) -> Result<Option<GlobalVar>> {
        if let Some(tok) = self.current() {
            log::debug!("{tok:?}");
            let func = match tok {
                Token::Extern => self.ext()?,
                Token::Fn => self.func()?,
                _ => self.top_level_expr()?,
            };
            if !func.is_anonymous() && func.proto().proto.name() == ANONYMOUS_FN_NAME {
                Err(ParseError::Reserved(ANONYMOUS_FN_NAME))
            } else {
                Ok(Some(GlobalVar::Function(func)))
            }
        } else {
            Ok(None)
        }
    }

    /// ```no_run
    /// top_level_expr ::= stmt
    /// ```
    fn top_level_expr(&mut self) -> Result<Function> {
        let mut exprs = vec![];
        let info = self.expect_src_info()?;
        if let Some(expr) = self.stmt()? {
            exprs.push(expr);
        }
        Ok(Function {
            proto: ProtoType {
                name: ANONYMOUS_FN_NAME.to_string(),
                args: vec![],
            }
            .with_info(info),
            body: Some(exprs),
            is_anonymous: true,
        })
    }

    /// ```no_run
    /// expr ::= primary bin_rhs
    /// ```
    fn expr(&mut self) -> Result<ExprInfo> {
        let lhs = self.primary()?;
        self.bin_rhs(0, lhs)
    }

    /// ```no_run
    /// stmt ::= (decl | expr | block)? ';'
    /// ```
    fn stmt(&mut self) -> Result<Option<ExprInfo>> {
        // TODO: refine BNF
        let _ = self._consume_comment();
        if self.consume_in_case_single(';')? {
            return Ok(None);
        }
        let info = self.expect_src_info()?;
        let expr = if self.current_is_let()? {
            self.decl()?
        } else if self.current_is_single('{')? {
            let stmts = self.block()?;
            Expr::Block(stmts).with_info(info)
        } else {
            self.expr()?
        };
        log::debug!("{expr:?}");
        self.try_consume_single(';')?;
        Ok(Some(expr))
    }

    /// ```no_run
    /// decl ::= "let" var "=" expr
    /// ```
    /// # Panics
    /// If current head token is not 'let'.
    fn decl(&mut self) -> Result<ExprInfo> {
        self.consume_let();
        let var = self.var()?;
        let info = self.expect_src_info()?;
        self.try_consume_single('=')?;
        let left = self.expr()?.boxed();
        Ok(Expr::Decl { var, left }.with_info(info))
    }

    /// ```no_run
    /// var ::= ("mut")? ident
    /// ```
    fn var(&mut self) -> Result<VarDeclInfo> {
        let is_mutable = self.consume_in_case_mut()?;
        let info = self.expect_src_info()?;
        let name = self.try_consume_ident()?;
        Ok(VarDecl::new(name, is_mutable).with_info(info))
    }

    /// ```no_run
    /// primary ::= ident_expr
    ///          | num_expr
    ///          | paren_expr
    ///          | contional
    ///          | for_expr
    /// ```
    fn primary(&mut self) -> Result<ExprInfo> {
        let info = self.expect_src_info()?;
        match self.expect()? {
            Token::Ident(_) => self.ident_expr(),
            Token::Num(_) => self.num_expr(),
            Token::Single('(') => self.paren_expr(),
            Token::If => self.conditional(),
            Token::For => self.for_expr(),
            tok => Err(ParseError::unexpected_token(tok, info)),
        }
    }

    /// ```no_run
    /// block ::= '{' stmt* '}'
    /// ```
    /// # Panics
    /// If current head token is not '{'.
    fn block(&mut self) -> Result<Vec<ExprInfo>> {
        self.try_consume_single('{')?;
        let mut stmts = vec![];
        loop {
            if self.consume_in_case_single('}')? {
                break;
            }
            if let Some(stmt) = self.stmt()? {
                stmts.push(stmt);
            }
        }
        Ok(stmts)
    }

    /// ```no_run
    /// conditional ::= "if" expr block
    ///                ("else" (conditional | block)*
    /// ```
    /// # Panics
    /// If current head token is not 'if'.
    fn conditional(&mut self) -> Result<ExprInfo> {
        let info = self.current_src_info().unwrap();
        self.consume_if();
        let cond = self.expr()?.boxed();
        let stmts = self.block()?;
        let mut else_stmts = vec![];
        if let Some(Token::Else) = self.current() {
            let _ = self.consume();
            if let Some(Token::If) = self.current() {
                let expr = self.conditional()?;
                else_stmts.push(expr);
            } else {
                else_stmts = self.block()?
            }
        }
        Ok(Expr::If {
            cond,
            stmts,
            else_stmts,
        }
        .with_info(info))
    }

    /// ```no_run
    /// for_expr ::= "for" var "<-" expr ".." expr ',' expr block
    /// ```
    /// # Panics
    /// If current head token is not 'for'.
    fn for_expr(&mut self) -> Result<ExprInfo> {
        let info = self.current_src_info().unwrap();
        self.consume_for();
        let generatee = self.var()?;
        self.try_consume_double("<-")?;
        let start = self.expr()?.boxed();
        self.try_consume_double("..")?;
        let end = self.expr()?.boxed();
        self.try_consume_single(',')?;
        let step = self.expr()?.boxed();
        let stmts = self.block()?;
        Ok(Expr::For {
            start,
            end,
            step,
            generatee,
            stmts,
        }
        .with_info(info))
    }

    fn current_precedence(&self) -> Option<i32> {
        if let Ok(op) = self.current()?.try_into() {
            PREC.get(&op).copied()
        } else {
            None
        }
    }

    /// ```no_run
    /// bin_rhs ::= (bin_op primary)*
    /// bin_op ::= /* refer to BinOp */
    /// ```
    fn bin_rhs(&mut self, precedence: i32, mut lhs: ExprInfo) -> Result<ExprInfo> {
        loop {
            log::debug!("{:?}", self.current());
            let tok_prec = self.current_precedence();
            if matches!(tok_prec, Some(p) if p >= precedence) {
                let tok_prec = tok_prec.unwrap();
                let info = self.current_src_info().unwrap();
                let op = self.consume().try_into().unwrap();
                let mut rhs = self.primary()?;
                // in case form of:
                //     bin_op1 primary bin_op2 primary ..
                // If prec(bin_op1) < prec(bin_op2), "primary bin_op2 primary .. " must be
                // merged first, thus regarding "primary bin_op2 primary" as bin_rhs.
                // Otherwise, we can greedily merge "primary bin_op1 primary" first.
                if matches!(self.current_precedence(), Some(p) if p > tok_prec) {
                    rhs = self.bin_rhs(tok_prec + 1, rhs)?;
                }

                lhs = if op == BinOp::Pipe {
                    desugar::desugar_pipe(lhs, rhs)?
                } else {
                    Expr::Binary {
                        op,
                        left: Box::new(lhs),
                        right: Box::new(rhs),
                    }
                    .with_info(info)
                };
            } else {
                return Ok(lhs);
            }
        }
    }

    /// ```no_run
    /// num_expr ::= number
    ///
    /// ```
    fn num_expr(&mut self) -> Result<ExprInfo> {
        let info = self.expect_src_info()?;
        let expr = match self.consume() {
            Token::Num(val) => Expr::Number(val),
            _ => return Err(ParseError::Unimplemented),
        };
        Ok(expr.with_info(info))
    }

    /// ```no_run
    /// paren_expr ::= '(' expr ')'
    ///
    /// ```
    /// # Panics
    /// If current head token is not '('.
    fn paren_expr(&mut self) -> Result<ExprInfo> {
        self.consume_single('(');
        let expr = self.expr()?;
        self.try_consume_single(')')?;
        Ok(expr)
    }

    /// ```no_run
    /// ident_expr ::= ident
    ///              | ident '(' (expr ',')* (expr)? (',')?  ')'
    /// ```
    fn ident_expr(&mut self) -> Result<ExprInfo> {
        let info = self.expect_src_info()?;
        let name = self.try_consume_ident()?;
        if self.consume_in_case_single('(')? {
            let mut args = VecDeque::new();
            loop {
                if self.current_is_single(')')? {
                    break;
                }
                let expr = self.expr()?;
                args.push_back(expr);
                if !self.current_is_single(',')? {
                    break;
                }
                self.consume_single(',');
            }
            self.try_consume_single(')')?;
            Ok(Expr::Call { callee: name, args }.with_info(info))
        } else {
            Ok(Expr::Variable(name).with_info(info))
        }
    }

    /// ```no_run
    /// proto ::= ident '(' (var ',')* ident ')'
    /// ```
    fn proto(&mut self) -> Result<ProtoTypeInfo> {
        let info = self.expect_src_info()?;
        let name = self.try_consume_ident()?;
        let mut args = vec![];
        self.try_consume_single('(')?;
        loop {
            log::debug!("{:?}", self.current());
            if self.consume_in_case_single(')')? {
                break;
            }
            let value = self.var()?;
            args.push(value);
            if !self.current_is_single(',')? {
                self.try_consume_single(')')?;
                break;
            }
            self.consume_single(',');
        }
        Ok(ProtoType { name, args }.with_info(info))
    }

    /// ```no_run
    /// func ::= "fn" proto block
    /// ```
    /// # Panics
    /// If current head token is not 'fn'.
    fn func(&mut self) -> Result<Function> {
        self.try_consume_fn()?;
        let proto = self.proto()?;
        let stmts = self.block()?;
        Ok(Function {
            proto,
            body: Some(stmts),
            is_anonymous: false,
        })
    }

    // Extern declaration must be done with no contents.
    /// ```no_run
    /// ext ::= "extern" "fn" proto ";"
    /// ```
    /// # Panics
    /// If current head token is not 'extern'.
    fn ext(&mut self) -> Result<Function> {
        self.consume_extern();
        self.try_consume_fn()?;
        let proto = self.proto()?;
        self.try_consume_single(';')?;
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
        (self.pos < self.tokens.len()).then(|| self.tokens[self.pos].token.clone())
    }

    /// Get the location (i.e. line and column) of the current token.
    #[inline]
    fn current_src_info(&self) -> Option<SrcInfo> {
        (self.pos < self.tokens.len()).then(|| {
            let info = &self.tokens[self.pos];
            info.info
        })
    }

    /// Get the current token. Note that this take [`Token::Eof`] as an error.
    #[inline]
    fn expect(&self) -> Result<Token> {
        self.current()
            .ok_or(ParseError::TokenShortage)
            .and_then(|tok| match tok {
                Token::Eof => Err(ParseError::eof()),
                tok => Ok(tok),
            })
    }

    /// Get the location (i.e. line and column) of the current token. Note that this take [`Token::Eof`] as an error.
    #[inline]
    fn expect_src_info(&self) -> Result<SrcInfo> {
        self.current_src_info().ok_or(ParseError::TokenShortage)
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
        let info = self.expect_src_info()?;
        let tok = self.try_consume()?;
        if tok == expected_token {
            Ok(tok)
        } else {
            Err(ParseError::unexpected_token(tok, info))
        }
    }
}

macro_rules! impl_consume {
    ($(($fn_suffix:ident, $TokenVariant:ident, $doc_ident:literal $(,)?)),+ $(,)?) => {
        paste::paste!{
                $(
                #[allow(unused)]
                #[inline]
                fn [<consume_ $fn_suffix>](&mut self) {
                    self.[<try_consume_ $fn_suffix>]().unwrap();
                }

                #[allow(unused)]
                #[inline]
                fn [<try_consume_ $fn_suffix>](&mut self) -> Result<()> {
                    self._try_consume_specified(Token::$TokenVariant).map(|_| ())
                }

                #[allow(unused)]
                #[inline]
                fn [<current_is_ $fn_suffix>](&self) -> Result<bool> {
                    Ok(self.expect()? == Token::$TokenVariant)
                }

                #[allow(unused)]
                #[inline]
                #[doc = "Consume current token only if it's `"]
                #[doc = $doc_ident]
                #[doc = "`. Note that this function returns `Err` only if no token is left."]
                fn [<consume_in_case_ $fn_suffix>](&mut self) -> Result<bool> {
                    self.[<current_is_ $fn_suffix>]()
                        .map(|in_case| in_case.then(|| self.[<consume_ $fn_suffix>]()).is_some())
                }
            )+
        }
    };
}

impl Parser {
    #[inline]
    fn try_consume_single(&mut self, c: char) -> Result<()> {
        self._try_consume_specified(Token::Single(c)).map(|_| ())
    }

    #[inline]
    fn consume_single(&mut self, c: char) {
        self.try_consume_single(c).unwrap();
    }

    #[inline]
    fn current_is_single(&self, c: char) -> Result<bool> {
        Ok(self.expect()? == Token::Single(c))
    }

    #[inline]
    /// Consume current token only if it's [`Token::Single`] corresponds to [`c`].
    /// "Note that this function returns `Err` only if no token is left."]
    fn consume_in_case_single(&mut self, c: char) -> Result<bool> {
        // Err only in case no token left.
        self.current_is_single(c)
            .map(|is_single| is_single.then(|| self.consume_single(c)).is_some())
    }

    #[inline]
    fn try_consume_double(&mut self, s: &'static str) -> Result<()> {
        debug_assert!(
            s.chars().count() == 2,
            "Internal Error: try_consume_double called with string not length of 2."
        );
        self._try_consume_specified(Token::Double(s)).map(|_| ())
    }

    #[inline]
    fn _consume_double(&mut self, s: &'static str) {
        self.try_consume_double(s).unwrap();
    }

    impl_consume! {
        (extern, Extern, "extern"),
        (fn, Fn, "fn"),
        (for, For, "for"),
        (if, If, "if"),
        (let, Let, "let"),
        (mut, Mut, "mut")
    }

    #[inline]
    fn try_consume_ident(&mut self) -> Result<String> {
        let info = self.expect_src_info()?;
        let tok = self.try_consume()?;
        if let Token::Ident(name) = tok {
            Ok(name)
        } else {
            Err(ParseError::unexpected_token(tok, info))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl PartialEq for Expr {
        fn eq(&self, other: &Self) -> bool {
            use Expr::*;
            match self {
                Binary { op, left, right } => {
                    matches!(other, Binary { op: o, left: l, right: r } if op == o && left == l && right == r)
                }
                Block(exprs) => matches!(other, Block(b) if exprs == b),
                Call { callee, args } => {
                    matches!(other, Call { callee: c, args:a } if callee==c && args == a)
                }
                Number(f) => matches!(other, Number(g) if f == g),
                Variable(name) => matches!(other, Variable(n) if name == n),
                _ => panic!("No use for this test!"),
            }
        }
    }

    impl PartialEq for ExprInfo {
        fn eq(&self, other: &Self) -> bool {
            self.expr == other.expr
        }
    }

    fn compare_ast(expected_equivalent: &[(&str, &str)]) {
        fn compile(src: &str) -> Vec<ExprInfo> {
            use crate::lexer::{Lexer, Result as LexResult};
            let lexer = Lexer::new(src).collect::<LexResult<Vec<_>>>().unwrap();
            let mut parser = Parser::new(lexer);
            match parser.parse().unwrap().unwrap() {
                GlobalVar::Function(f) => f.body.unwrap(),
            }
        }
        expected_equivalent
            .iter()
            .map(|(l, r)| {
                // Add semicolon as Parser takes expressions without ';' invalid.
                let l = format!("{l};");
                let r = format!("{r};");
                assert_eq!(compile(&l), compile(&r));
            })
            .collect()
    }

    #[test]
    fn precedence() {
        compare_ast(&[
            ("1 + 2 + 3", "((1 + 2) + 3)"),
            ("1 + 2 * 3", "(1 + (2 * 3))"),
            ("1 + 2 * 3 / 4 ;", "(1 + ((2 * 3) / 4))"),
            ("1 + 2 * 3 / 4 - 5;", "((1 + ((2 * 3) / 4)) - 5)"),
            ("1 + 1 > 2", "(1 + 1) > 2"),
            (
                "1+1+1+1+1+1+1+1+1+1+1",
                "((((((((((1+1)+1)+1)+1)+1)+1)+1)+1)+1)+1)",
            ),
            ("1 + 2 /> add()", "(1 + 2) /> add()"),
        ]);
    }
}
