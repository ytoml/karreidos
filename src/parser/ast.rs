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
    If {
        cond: Box<Expr>,
        stmts: Vec<Expr>,
        // Note that it has single Expr::If for "else if ..."
        else_stmts: Vec<Expr>,
    },
    For {
        start: Box<Expr>,
        end: Box<Expr>,
        step: Box<Expr>,
        generatee: String, // currently only 1 variable can be generated
        stmts: Vec<Expr>,
    },
    Decl {
        name: String,
        left: Box<Expr>,
    },
    Block(Vec<Expr>), // needed to manage scope
    Variable(String),
    Number(f64),
}
impl Expr {
    pub fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
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
