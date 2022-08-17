#[derive(Debug, Clone)]
pub enum Expr {
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Call {
        callee: String,
        args: Vec<Expr>,
    },
    Variable(String),
    Number(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Lt,
    Gt,
    Add,
    Sub,
    Div,
    Mul,
    Assign,
    _LShift,
    _RShift,
    _AddAssign,
    _SubAssign,
    _MulAssign,
    _DivAssign,
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
