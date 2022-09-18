use super::ast::{BinOp, Expr};
use super::{ParseError, Result};

pub(super) fn desugar_pipe(lhs: Expr, rhs: Expr) -> Result<Expr> {
    match rhs {
        Expr::Call { callee, mut args } => {
            args.push_front(lhs);
            Ok(Expr::Call { callee, args })
        }
        Expr::Binary {
            op: BinOp::Pipe, ..
        } => unreachable!(), // unreachable if operator precedence is properly handled
        _ => Err(ParseError::NonCallableAtPipeRhs(rhs)),
    }
}
