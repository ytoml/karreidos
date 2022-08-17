use derive_builder::UninitializedFieldError;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::BasicMetadataTypeEnum;
use inkwell::values::{
    BasicMetadataValueEnum, BasicValue, FloatValue, FunctionValue, PointerValue,
};
use inkwell::FloatPredicate;
use std::borrow::Borrow;
use std::collections::HashMap;

use crate::parser::{Function, ProtoType};
// use derive_builder::Builder;
use crate::parser::ast::{BinOp, Expr};

pub type Result<T> = std::result::Result<T, CompileError>;

#[derive(Debug)]
pub enum CompileError {
    Undefined(String, Type),
    Redefined(String, Type),
    InvalidAssignment(Expr),
    Fatal(Fatal),
    Other(&'static str),
}

#[derive(Debug)]
pub enum Fatal {
    CompilerBuild(UninitializedFieldError),
    Inconsistent(String, Type),
}

#[derive(Debug)]
pub enum Type {
    Funtion,
    Variable,
    Argument,
}

/// Compile per function.
#[derive(Builder, Debug)]
#[builder(name = "Compiler", build_fn(private, error = "CompileError"))]
pub struct IrGenerator<'a, 'ctx> {
    #[builder(setter(name = "context"))]
    ctx: &'ctx Context,
    builder: &'a Builder<'ctx>,
    module: &'a Module<'ctx>,
    /// Manages variables in current scope.
    #[builder(default)]
    variables: HashMap<String, PointerValue<'ctx>>, // Note that querying functions are done throuth self.module rather than self.variables
}

impl<'a, 'ctx> IrGenerator<'a, 'ctx> {
    fn expr_gen(&mut self, expr: &Expr) -> Result<FloatValue<'ctx>> {
        match expr {
            &Expr::Number(value) => Ok(self.ctx.f64_type().const_float(value)),
            Expr::Variable(name) => self
                .variables
                .get(name)
                .map(|pval| self.builder.build_load(*pval, name).into_float_value())
                .ok_or_else(|| CompileError::Undefined(name.clone(), Type::Variable)),
            Expr::Binary { op, left, right } => {
                let lhs = self.expr_gen(left.as_ref())?;
                let rhs = self.expr_gen(right.as_ref())?;
                match *op {
                    BinOp::Add => Ok(self.builder.build_float_add(lhs, rhs, "addtmp")),
                    BinOp::Sub => Ok(self.builder.build_float_sub(lhs, rhs, "subtmp")),
                    BinOp::Mul => Ok(self.builder.build_float_mul(lhs, rhs, "multmp")),
                    BinOp::Div => Ok(self.builder.build_float_div(lhs, rhs, "divtmp")),
                    BinOp::Lt => {
                        let cmp_int = self.builder.build_float_compare(
                            FloatPredicate::ULT,
                            lhs,
                            rhs,
                            "ltftmp",
                        );
                        Ok(self.builder.build_unsigned_int_to_float(
                            cmp_int,
                            self.ctx.f64_type(),
                            "booltmp",
                        ))
                    }
                    BinOp::Gt => {
                        let cmp_int = self.builder.build_float_compare(
                            FloatPredicate::UGT,
                            lhs,
                            rhs,
                            "gtftmp",
                        );
                        Ok(self.builder.build_unsigned_int_to_float(
                            cmp_int,
                            self.ctx.f64_type(),
                            "booltmp",
                        ))
                    }
                    BinOp::Assign => {
                        let var_name = match left.borrow() {
                            Expr::Variable(name) => name.clone(),
                            expr => {
                                return Err(CompileError::InvalidAssignment(expr.clone()));
                            }
                        };
                        let expr_value = self.expr_gen(right.as_ref())?;
                        log::debug!("assignment: right value {expr_value}");
                        let var_ptr = self
                            .variables
                            .get(&var_name)
                            .ok_or(CompileError::Undefined(var_name, Type::Variable))?;
                        let _ = self.builder.build_store(*var_ptr, expr_value);
                        Ok(expr_value)
                    }
                    binop => unimplemented!("Unsupported bin op ({binop:?})"),
                }
            }
            Expr::Call { callee, args } => {
                let function = self
                    .get_function(callee)
                    .ok_or_else(|| CompileError::Undefined(callee.clone(), Type::Funtion))?;
                let mut args_gen = Vec::with_capacity(args.len());
                for arg in args {
                    let arg_gen = self.expr_gen(arg)?;
                    args_gen.push(arg_gen);
                }
                let args: Vec<BasicMetadataValueEnum> =
                    args_gen.iter().by_ref().map(|&val| val.into()).collect();
                self.builder
                    .build_call(function, args.as_slice(), "calltmp")
                    .try_as_basic_value()
                    .left()
                    .map(|value| value.into_float_value())
                    .ok_or(CompileError::Other(
                        "Void return value found but currently not supported in karreidos.",
                    ))
            }
        }
    }
    fn proto_gen(&self, proto: &ProtoType) -> Result<FunctionValue<'ctx>> {
        let mut arg_types = Vec::with_capacity(proto.num_args());
        for _ in 0..proto.num_args() {
            // Currently, Karreidos only supports float values.
            arg_types.push(self.ctx.f64_type());
        }
        let arg_types: Vec<BasicMetadataTypeEnum> =
            arg_types.into_iter().map(|ty| ty.into()).collect();
        let fn_ty = self.ctx.f64_type().fn_type(arg_types.as_slice(), false);
        let fn_val = self.module.add_function(proto.name(), fn_ty, None);

        // Not necessary but make IR more readable.
        for (name, arg) in proto.arg_names_slice().iter().zip(fn_val.get_param_iter()) {
            arg.into_float_value().set_name(name.as_str());
        }
        Ok(fn_val)
    }

    fn function_gen(&mut self, function: &Function) -> Result<FunctionValue<'ctx>> {
        let proto = function.proto();
        let fn_val = self.proto_gen(proto)?;
        if function.body().is_none() {
            // Only prototype declaration
            return Ok(fn_val);
        }

        let name = proto.name();
        if matches!(self.get_function(name), Some(f) if f.get_first_basic_block().is_some()) {
            // Function declared with same name twice (with procedure expressions)
            return Err(CompileError::Redefined(name.to_string(), Type::Funtion));
        }

        let _ = self
            .module
            .get_function(name)
            .ok_or_else(|| CompileError::Undefined(name.to_string(), Type::Funtion))?;

        // Only single basic block is needed until we implement control flow.
        let block = self.ctx.append_basic_block(fn_val, "entry");
        self.builder.position_at_end(block);

        // Make arguments name accesible in [`expr_gen()`].
        self.variables.reserve(proto.num_args());

        for (arg_name, arg) in proto.arg_names_slice().iter().zip(fn_val.get_param_iter()) {
            log::debug!("{name}: add arg {arg_name}");
            let stack_alloca = self.entry_block_stack_alloca(name, &fn_val);
            let _ = self.builder.build_store(stack_alloca, arg);
            if let Some(_prev_alloca) = self.variables.insert(arg_name.clone(), stack_alloca) {
                return Err(CompileError::Redefined(arg_name.clone(), Type::Argument));
            }
        }
        log::debug!("{:#?}", block);

        // XXX: is expr really needed to be array?
        if let Some(exprs) = function.body() {
            let mut body = None;
            for expr in exprs {
                self.builder.position_at_end(block);
                let value = self.expr_gen(expr)?;
                log::debug!("{:#?}", block);
                let _ = body.insert(value); // preserve the last value
            }
            log::debug!("{name}: building return... {body:?}");
            self.builder.position_at_end(block);
            let ret_inst = self
                .builder
                .build_return(body.as_ref().map(|v| v as &dyn BasicValue));
            log::debug!("{ret_inst}");
        }
        log::debug!("{:#?}", block);
        fn_val.verify(true).then_some(fn_val).ok_or_else(|| {
            log::error!("Function validation failed.");
            unsafe { fn_val.delete() };
            CompileError::Fatal(Fatal::Inconsistent(name.to_string(), Type::Funtion))
        })
    }
}
impl<'a, 'ctx> IrGenerator<'a, 'ctx> {
    fn get_function(&self, name: impl AsRef<str>) -> Option<FunctionValue<'ctx>> {
        self.module.get_function(name.as_ref())
    }

    /// Creates a new stack allocation instruction in the entry block of the function.
    fn entry_block_stack_alloca(
        &self,
        var_name: &str,
        fn_val: &FunctionValue<'ctx>,
    ) -> PointerValue<'ctx> {
        let entry = fn_val.get_first_basic_block().unwrap_or_else(|| {
            panic!(
                "Internal(entry_block_stack_alloca): basic block not allocated for {:?}",
                fn_val.get_name()
            )
        });

        // Insert stack allocation at beggining of the entry.
        match entry.get_first_instruction() {
            Some(instruction) => self.builder.position_before(&instruction),
            None => self.builder.position_at_end(entry),
        };

        // only f64 supported now
        self.builder.build_alloca(self.ctx.f64_type(), var_name)
    }
}

impl<'a, 'ctx> Compiler<'a, 'ctx> {
    pub fn compile(&'a self, function: &Function) -> Result<FunctionValue<'ctx>> {
        match self.build()?.function_gen(function) {
            Err(CompileError::Fatal(fata)) => {
                panic!("Fatal: {fata:?}");
            }
            res => res,
        }
    }
}
impl From<UninitializedFieldError> for CompileError {
    fn from(err: UninitializedFieldError) -> Self {
        Self::Fatal(Fatal::CompilerBuild(err))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use inkwell::context::Context;

    #[test]
    fn build() {
        let ctx = Context::create();
        let builder = ctx.create_builder();
        let module = ctx.create_module("test");
        let _compiler = Compiler::default()
            .builder(&builder)
            .context(&ctx)
            .module(&module)
            .build()
            .unwrap();
    }
}
