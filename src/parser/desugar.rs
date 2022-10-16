use super::ast::{BinOp, Expr, ExprInfo};
use super::{ParseError, Result};

pub(super) fn desugar_pipe(lhs: ExprInfo, rhs: ExprInfo) -> Result<ExprInfo> {
    match rhs.expr {
        Expr::Call { callee, mut args } => {
            args.push_front(lhs);
            Ok(Expr::Call { callee, args }.with_info(rhs.info))
        }
        Expr::Binary {
            op: BinOp::Pipe, ..
        } => unreachable!(), // unreachable if operator precedence is properly handled
        _ => Err(ParseError::NonCallableAtPipeRhs(rhs)),
    }
}
