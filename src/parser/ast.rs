use std::{
    collections::VecDeque,
    convert::{TryFrom, TryInto},
};

use crate::lexer::token::Token;

#[derive(Debug, Clone)]
pub enum Expr {
    Binary {
        op: BinOp,
        left: Box<ExprInfo>,
        right: Box<ExprInfo>,
    },
    Call {
        callee: String,
        args: VecDeque<ExprInfo>,
    },
    If {
        cond: Box<ExprInfo>,
        stmts: Vec<ExprInfo>,
        // Note that it has single Expr::If for "else if ..."
        else_stmts: Vec<ExprInfo>,
    },
    For {
        start: Box<ExprInfo>,
        end: Box<ExprInfo>,
        step: Box<ExprInfo>,
        generatee: VarDeclInfo, // currently only 1 variable can be generated
        stmts: Vec<ExprInfo>,
    },
    Decl {
        var: VarDeclInfo,
        left: Box<ExprInfo>,
    },
    Block(Vec<ExprInfo>), // needed to manage scope
    Variable(String),
    Number(f64),
}
info_impl!(Expr, expr);
impl ExprInfo {
    #[inline]
    pub fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
}

#[derive(Debug, Clone)]
pub struct VarDecl {
    name: String,
    is_mutable: bool,
}
info_impl!(VarDecl, var);
impl VarDecl {
    #[inline]
    pub(super) const fn new(name: String, is_mutable: bool) -> Self {
        Self { name, is_mutable }
    }

    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[inline]
    pub const fn is_mutable(&self) -> bool {
        self.is_mutable
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Lt,
    Gt,
    Add,
    Sub,
    Div,
    Mul,
    Assign,
    Pipe, // `/>`
    LShift,
    RShift,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
}

impl TryFrom<Token> for BinOp {
    type Error = Token;
    fn try_from(token: Token) -> Result<Self, Self::Error> {
        let op = match token {
            Token::Single(c) => c.try_into().map_err(Token::Single)?,
            Token::Double(s) => s.try_into().map_err(Token::Double)?,
            _ => return Err(token),
        };
        Ok(op)
    }
}

impl<'a> TryFrom<&'a str> for BinOp {
    type Error = &'a str;
    fn try_from(s: &'a str) -> Result<Self, Self::Error> {
        let op = match s {
            "/>" => Self::Pipe,
            "<<" => Self::LShift,
            ">>" => Self::RShift,
            "+=" => Self::AddAssign,
            "-=" => Self::SubAssign,
            "*=" => Self::MulAssign,
            "/=" => Self::DivAssign,
            _ => return Err(s),
        };
        Ok(op)
    }
}

impl TryFrom<char> for BinOp {
    type Error = char;
    fn try_from(c: char) -> Result<Self, Self::Error> {
        let op = match c {
            '<' => Self::Lt,
            '>' => Self::Gt,
            '+' => Self::Add,
            '-' => Self::Sub,
            '*' => Self::Mul,
            '/' => Self::Div,
            '=' => Self::Assign,
            _ => return Err(c),
        };
        Ok(op)
    }
}
