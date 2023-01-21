use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::rc::Rc;

use derive_builder::UninitializedFieldError;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::debug_info::{
    AsDIScope, DICompileUnit, DIFile, DIFlagsConstants, DILocation, DIScope, DISubprogram,
    DISubroutineType, DIType, DWARFEmissionKind, DWARFSourceLanguage, DebugInfoBuilder,
};
use inkwell::module::Module;
use inkwell::passes::{PassManager, PassManagerSubType};
use inkwell::targets::TargetMachine;
use inkwell::types::BasicMetadataTypeEnum;
use inkwell::values::{BasicMetadataValueEnum, FloatValue, FunctionValue, PointerValue};
use inkwell::FloatPredicate;

use crate::parser::ast::{BinOp, Expr, ExprInfo, VarDeclInfo};
use crate::parser::{Function, ProtoType, ProtoTypeInfo};

pub type Result<T> = std::result::Result<T, CompileError>;

const PRODUCER: &str = concat!(clap::crate_name!(), "-", clap::crate_version!());

#[derive(Debug)]
pub enum CompileError {
    Undefined(String, Type),
    Redefined(String, Type),
    InvalidAssignment(InvalidAssignment),
    InvalidArity { expected: usize, passed: usize },
    Fatal(Fatal),
    Other(String),
}

#[derive(Debug)]
pub enum InvalidAssignment {
    CannotBeLeft(Expr),
    Immutable(String),
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

#[derive(Debug, Clone, Copy)]
struct Variable<'ctx> {
    pointer_value: PointerValue<'ctx>,
    is_mutable: bool,
}
impl<'ctx> Variable<'ctx> {
    const fn new(pointer_value: PointerValue<'ctx>, is_mutable: bool) -> Self {
        Self {
            pointer_value,
            is_mutable,
        }
    }
}

#[derive(Debug, Clone)]
struct DebugInfoFactory<'ctx> {
    ctx: &'ctx Context,
    builder: Rc<DebugInfoBuilder<'ctx>>,
    compile_unit: DICompileUnit<'ctx>,
    ty: DIType<'ctx>, // hold because Karreidos currently support f64 only.
    lexical_blocks: Vec<DIScope<'ctx>>,
}
impl<'ctx> DebugInfoFactory<'ctx> {
    fn new(ctx: &'ctx Context, module: &Module<'ctx>, file_name: &str) -> Self {
        // TODO: flexible settings
        let path = Path::new(file_name)
            .canonicalize()
            .expect("Fatal: Source file does not exists.");
        let filename = path
            .file_name()
            .unwrap()
            .to_str()
            .expect("Fatal: invalid string for directory.");
        let directory = path
            .parent()
            .expect("Root directory specified.")
            .as_os_str()
            .to_str()
            .expect("Fatal: invalid string for directory.");
        let debug_output_path = filename.to_string() + "_debug";

        let (builder, compile_unit) = module.create_debug_info_builder(
            true,
            DWARFSourceLanguage::C,
            filename,
            directory,
            PRODUCER,
            false,
            "",
            0,
            &debug_output_path,
            DWARFEmissionKind::Full,
            0,
            false,
            false,
            "",
            "",
        );
        // encoding 0 i.e. no modification.
        let ty = builder
            .create_basic_type("double", 64, 0, DIFlagsConstants::PUBLIC)
            .unwrap()
            .as_type();

        Self {
            ctx,
            builder: Rc::new(builder),
            compile_unit,
            ty,
            lexical_blocks: vec![],
        }
    }

    fn create_function_type(&self, num_args: usize) -> DISubroutineType<'ctx> {
        let mut args_ty = Vec::with_capacity(num_args);
        for _ in 0..num_args {
            args_ty.push(self.ty)
        }
        self.builder.create_subroutine_type(
            self.get_file(),
            Some(self.ty),
            &args_ty,
            DIFlagsConstants::PUBLIC,
        )
    }

    fn create_function(&self, name: &str, num_args: usize, line: u32) -> DISubprogram<'ctx> {
        let ditype = self.create_function_type(num_args);
        let file = self.get_file();
        // line_no and scope_line are set to be same because currently scopes are managed only per function.
        self.builder.create_function(
            file.as_debug_info_scope(),
            name,
            None,
            file,
            line,
            ditype,
            false,
            true,
            line,
            DIFlagsConstants::PROTOTYPED,
            false,
        )
    }

    #[inline]
    fn get_file(&self) -> DIFile<'ctx> {
        self.compile_unit.get_file()
    }

    fn emit_location(&self, expr: &ExprInfo) -> DILocation<'ctx> {
        let scope = self
            .lexical_blocks
            .last()
            .copied()
            .unwrap_or_else(|| self.compile_unit.as_debug_info_scope());
        self.builder
            .create_debug_location(self.ctx, expr.info.line, expr.info.col, scope, None)
    }

    #[inline]
    fn scope_up(&mut self, scope: DIScope<'ctx>) {
        self.lexical_blocks.push(scope);
    }

    #[inline]
    fn scope_down(&mut self) {
        self.lexical_blocks
            .pop()
            .expect("Fatal: Too much scope down called.");
    }
}

/// Compile per function.
#[derive(Builder, Debug)]
#[builder(name = "Compiler", build_fn(private, error = "CompileError"))]
pub struct IrGenerator<'a, 'ctx> {
    #[builder(setter(name = "context"))]
    ctx: &'ctx Context,
    builder: &'a Builder<'ctx>,
    module: &'a Module<'ctx>,
    #[builder(setter(name = "function_pass_manager"))]
    fpm: &'a PassManager<FunctionValue<'ctx>>,
    /// Manages variables.
    /// HashMap on back is current scope.
    #[builder(default, private)]
    variables: VecDeque<HashMap<String, Variable<'ctx>>>,
    // There is no way to set function in building process and always None at first.
    #[builder(setter(skip))]
    this_func: Option<FunctionValue<'ctx>>,
    #[builder(default = "false")]
    emit_obj: bool,
    #[builder(setter(strip_option, custom), default, private)]
    debug: Option<DebugInfoFactory<'ctx>>,
}

impl<'a, 'ctx> IrGenerator<'a, 'ctx> {
    #[inline]
    fn if_debug(&self, callback: impl FnOnce(&DebugInfoFactory<'ctx>)) {
        if let Some(debug) = self.debug.as_ref() {
            callback(debug);
        }
    }

    #[inline]
    fn if_debug_mut(&mut self, callback: impl FnOnce(&mut DebugInfoFactory<'ctx>)) {
        if let Some(debug) = self.debug.as_mut() {
            callback(debug);
        }
    }

    fn expr_gen(&mut self, expr: &ExprInfo) -> Result<FloatValue<'ctx>> {
        self.if_debug(|factory| {
            let location = factory.emit_location(expr);
            self.builder.set_current_debug_location(location);
        });
        match &expr.expr {
            &Expr::Number(value) => Ok(self.ctx.f64_type().const_float(value)),
            Expr::Variable(name) => self
                .get_variable(name)
                .map(|var| {
                    self.builder
                        .build_load(self.ctx.f64_type(), var.pointer_value, name)
                        .into_float_value()
                })
                .ok_or_else(|| CompileError::Undefined(name.clone(), Type::Variable)),
            Expr::Decl {
                var: VarDeclInfo { var, .. },
                left,
            } => {
                let current_block = self.get_current_block();
                let alloca = self.stack_alloca_in_block(var.name(), current_block);
                self.register_variable_at_current_scope(
                    var.name().to_string(),
                    Variable::new(alloca, var.is_mutable()),
                )?;
                let value = self.expr_gen(left.as_ref())?;
                self.builder.build_store(alloca, value);
                // Declaration statement itself has no value (like Rust).
                Ok(self.ctx.f64_type().const_zero())
            }
            Expr::Block(stmts) => {
                self.scope_up();
                let current_block = self.get_current_block();
                let block_value = self.block_gen(stmts, current_block)?;
                self.scope_down();
                Ok(block_value)
            }
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
                        let var_name = match &left.expr {
                            Expr::Variable(name) => name.clone(),
                            expr => {
                                return Err(CompileError::InvalidAssignment(
                                    InvalidAssignment::CannotBeLeft(expr.clone()),
                                ));
                            }
                        };
                        let expr_value = self.expr_gen(right.as_ref())?;
                        log::debug!("assignment: right value {expr_value}");
                        let ptr = match self.get_variable(&var_name) {
                            Some(var) if var.is_mutable => var.pointer_value,
                            Some(_) => {
                                return Err(CompileError::InvalidAssignment(
                                    InvalidAssignment::Immutable(var_name),
                                ))
                            }
                            None => return Err(CompileError::Undefined(var_name, Type::Variable)),
                        };
                        let _ = self.builder.build_store(ptr, expr_value);
                        Ok(expr_value)
                    }
                    binop => unimplemented!("Unsupported bin op ({binop:?})"),
                }
            }
            Expr::Call { callee, args } => {
                let function = self
                    .get_function(callee)
                    .ok_or_else(|| CompileError::Undefined(callee.clone(), Type::Funtion))?;
                let expected = function.count_params() as usize;
                let passed = args.len();
                if expected != passed {
                    return Err(CompileError::InvalidArity { expected, passed });
                }
                let mut args_gen = Vec::with_capacity(args.len());
                for arg in args.iter() {
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
                    .ok_or_else(|| {
                        CompileError::Other(
                            "Void return value found but currently not supported in Karreidos."
                                .to_string(),
                        )
                    })
            }
            Expr::If {
                cond,
                stmts,
                else_stmts,
            } => {
                let cond = self.expr_gen(cond.as_ref())?;
                let zero = self.ctx.f64_type().const_zero();
                let comparison =
                    self.builder
                        .build_float_compare(FloatPredicate::ONE, cond, zero, "condtmp");

                let function = self.get_this_function()?;
                let origin_block = self.get_current_block();

                let then_block = self.ctx.append_basic_block(function, "then");
                let then_value = self.block_gen(stmts, then_block)?;
                // Restoration is needed in case of nested branches (i.e. current block is not "then" anymore).
                let then_end_block = self.get_current_block();

                let else_block = self.ctx.append_basic_block(function, "else");
                let else_value = self.block_gen(else_stmts, else_block)?;
                let else_end_block = self.get_current_block();

                self.goto_end_of_block(origin_block);
                let _ = self
                    .builder
                    .build_conditional_branch(comparison, then_block, else_block);

                let merge_block = self.ctx.append_basic_block(function, "merge");
                self.insert_terminating_jump(then_end_block, merge_block);
                self.insert_terminating_jump(else_end_block, merge_block);
                self.goto_start_of_block(merge_block);
                let phi_value = self.builder.build_phi(self.ctx.f64_type(), "iftmp");
                phi_value
                    .add_incoming(&[(&then_value, then_end_block), (&else_value, else_end_block)]);
                Ok(phi_value.as_basic_value().into_float_value())
            }
            Expr::For {
                start,
                end,
                step,
                generatee: VarDeclInfo { var: generatee, .. },
                stmts,
            } => {
                let function = self.get_this_function()?;
                let current_block = self.get_current_block();
                let start_value = self.expr_gen(start.as_ref())?;
                let end_value = self.expr_gen(end.as_ref())?;
                let step_value = self.expr_gen(step.as_ref())?;
                let acc_alloca = self.stack_alloca_in_block(generatee.name(), current_block);

                // Must ensure this jump is terminator for current block
                self.goto_end_of_current_block();
                self.builder.build_store(acc_alloca, start_value);
                let for_block = self.ctx.append_basic_block(function, "for");
                self.insert_terminating_jump(current_block, for_block);

                self.scope_up();
                self.goto_start_of_block(for_block);
                let generatee_alloca = if generatee.is_mutable() {
                    // Prevent number of loops from changing even when generatee is mutable.
                    // Like Rust, e.g.
                    // ```for mut i <- 0..10, 1 { i = i + 1; };```
                    // yields 10 repetitions.
                    let alloca = self.stack_alloca_in_block(generatee.name(), for_block);
                    let value = self
                        .builder
                        .build_load(self.ctx.f64_type(), acc_alloca, "muttmp");
                    self.builder.build_store(alloca, value);
                    alloca
                } else {
                    acc_alloca
                };

                self.register_variable_at_current_scope(
                    generatee.name().to_string(),
                    Variable::new(generatee_alloca, generatee.is_mutable()),
                )?;
                let for_value = self.block_gen(stmts, for_block)?;
                self.builder.position_at_end(for_block);
                let acc_value = self
                    .builder
                    .build_load(self.ctx.f64_type(), acc_alloca, "acctmp")
                    .into_float_value();
                let next_acc_value =
                    self.builder
                        .build_float_add(acc_value, step_value, "stepaddtmp");
                self.builder.build_store(acc_alloca, next_acc_value);
                let comparison = self.builder.build_float_compare(
                    FloatPredicate::OLT,
                    next_acc_value,
                    end_value,
                    "forcondtmp",
                );
                self.scope_down();

                let break_block = self.ctx.append_basic_block(function, "break");
                self.builder
                    .build_conditional_branch(comparison, for_block, break_block);
                self.builder.position_at_end(break_block);
                Ok(for_value)
            }
        }
    }

    // For flexibility, scope_(up|down) are not in this function.
    fn block_gen(
        &mut self,
        stmts: &[ExprInfo],
        basic_block: BasicBlock<'ctx>,
    ) -> Result<FloatValue<'ctx>> {
        self.goto_end_of_block(basic_block);
        let mut values = stmts
            .iter()
            .map(|expr| self.expr_gen(expr))
            .collect::<Result<Vec<_>>>()?;
        Ok(values
            .pop()
            .unwrap_or_else(|| self.ctx.f64_type().const_zero()))
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
        for (name, arg) in proto.arg_names_iter().zip(fn_val.get_param_iter()) {
            arg.into_float_value().set_name(name);
        }
        Ok(fn_val)
    }

    fn function_gen(&mut self, function: &Function) -> Result<FunctionValue<'ctx>> {
        let ProtoTypeInfo { proto, info, .. } = function.proto();
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

        // Check if current defining function is not added to module
        // (it's possible when `name` is not valid, thus useful for debugging).
        #[cfg(debug_assertions)]
        let _ = self
            .module
            .get_function(name)
            .ok_or_else(|| CompileError::Undefined(name.to_string(), Type::Funtion))?;
        self.register_this_function(fn_val)?;

        let block = self.ctx.append_basic_block(fn_val, "entry");
        self.builder.position_at_end(block);

        self.if_debug_mut(|factory| {
            let subprogram = factory.create_function(proto.name(), proto.num_args(), info.line);
            fn_val.set_subprogram(subprogram);
            let scope = subprogram.as_debug_info_scope();
            factory.scope_up(scope);
        });

        self.scope_up();
        self.variables.reserve(proto.num_args());
        for (VarDeclInfo { var: arg, .. }, arg_llvm) in
            proto.args_slice().iter().zip(fn_val.get_param_iter())
        {
            log::debug!("{name}: add arg {}", arg.name());
            let stack_alloca = self.entry_block_stack_alloca(name, &fn_val);
            let _ = self.builder.build_store(stack_alloca, arg_llvm);
            self.register_argument(
                arg.name().to_string(),
                Variable::new(stack_alloca, arg.is_mutable()),
            )?;
        }
        if let Some(stmts) = function.body() {
            let return_value = self.block_gen(stmts, block)?;
            log::debug!("{name}: building return... {return_value:?}");
            let block = self.builder.get_insert_block().unwrap();
            self.builder.position_at_end(block);
            let return_inst = self.builder.build_return(Some(&return_value));
            log::debug!("{return_inst}");
        }
        self.scope_down();
        self.if_debug_mut(|factory| factory.scope_down());

        log::debug!("Before optimization: {fn_val}");
        fn_val
            .verify(true)
            .then(|| {
                log::debug!("Runnning pass manager...");
                unsafe {
                    fn_val.run_in_pass_manager(self.fpm);
                }
                log::debug!("After optimization: {fn_val}");
                fn_val
            })
            .ok_or_else(|| {
                log::error!("Function validation failed.");
                unsafe { fn_val.delete() };
                CompileError::Fatal(Fatal::Inconsistent(name.to_string(), Type::Funtion))
            })
    }
}
impl<'a, 'ctx> IrGenerator<'a, 'ctx> {
    #[inline]
    fn get_this_function(&self) -> Result<FunctionValue<'ctx>> {
        self.this_func.ok_or_else(|| {
            CompileError::Other("Function required while it has not been set yet".to_string())
        })
    }

    #[inline]
    fn register_this_function(&mut self, function: FunctionValue<'ctx>) -> Result<()> {
        match self.this_func {
            None => {
                let _ = self.this_func.insert(function);
                Ok(())
            }
            Some(func) => Err(CompileError::Other(format!(
                "Tried to register multiple functions: {:?}",
                func.get_name()
            ))),
        }
    }

    #[inline]
    fn get_function(&self, name: impl AsRef<str>) -> Option<FunctionValue<'ctx>> {
        // Look into `self.module` rather than `self.variables`
        // Because (currently) `IeGenerator` provides per function generation and thus
        // other function in same module may not be found in `self.variable` for most case
        // (because `self.variable` only manages function-local variables).
        self.module.get_function(name.as_ref())
    }

    #[inline]
    fn get_current_block(&self) -> BasicBlock<'ctx> {
        self.builder
            .get_insert_block()
            .expect("Try to get basic block while no one is set.")
    }

    fn _register_at_current_scope(
        &mut self,
        name: String,
        value: Variable<'ctx>,
        ty: Type,
    ) -> Result<()> {
        let current_scope = self.variables.back_mut().expect("No scope is available.");
        match current_scope.entry(name) {
            Entry::Occupied(entry) => Err(CompileError::Redefined(entry.remove_entry().0, ty)),
            Entry::Vacant(entry) => {
                entry.insert(value);
                Ok(())
            }
        }
    }

    #[inline]
    fn register_variable_at_current_scope(
        &mut self,
        name: String,
        value: Variable<'ctx>,
    ) -> Result<()> {
        self._register_at_current_scope(name, value, Type::Variable)
    }

    #[inline]
    fn register_argument(&mut self, name: String, value: Variable<'ctx>) -> Result<()> {
        self._register_at_current_scope(name, value, Type::Argument)
    }

    #[inline]
    fn get_variable(&self, name: &str) -> Option<&Variable<'ctx>> {
        self.variables
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
    }

    #[inline]
    fn goto_start_of_block(&self, basic_block: BasicBlock<'ctx>) {
        match basic_block.get_first_instruction() {
            Some(instruction) => self.builder.position_before(&instruction),
            None => self.builder.position_at_end(basic_block),
        }
    }

    #[inline]
    fn goto_end_of_block(&self, basic_block: BasicBlock<'ctx>) {
        self.builder.position_at_end(basic_block);
    }

    #[inline]
    fn _goto_start_of_current_block(&self) {
        self.goto_start_of_block(self.get_current_block());
    }

    #[inline]
    fn goto_end_of_current_block(&self) {
        self.goto_end_of_block(self.get_current_block());
    }

    #[inline]
    fn insert_terminating_jump(&self, from_block: BasicBlock<'ctx>, to_block: BasicBlock<'ctx>) {
        self.goto_end_of_block(from_block);
        self.builder.build_unconditional_branch(to_block);
    }

    // Insert stack allocation at beggining of the basic block
    #[inline]
    fn stack_alloca_in_block(
        &self,
        var_name: &str,
        basic_block: BasicBlock<'ctx>,
    ) -> PointerValue<'ctx> {
        self.goto_start_of_block(basic_block);
        // only f64 supported now
        self.builder.build_alloca(self.ctx.f64_type(), var_name)
    }

    // Creates a new stack allocation instruction in the entry block of the function.
    #[inline]
    fn entry_block_stack_alloca(
        &self,
        var_name: &str,
        fn_val: &FunctionValue<'ctx>,
    ) -> PointerValue<'ctx> {
        let entry_block = fn_val.get_first_basic_block().unwrap_or_else(|| {
            panic!(
                "Internal(entry_block_stack_alloca): basic block not allocated for {:?}",
                fn_val.get_name()
            )
        });
        self.stack_alloca_in_block(var_name, entry_block)
    }

    #[inline]
    fn scope_up(&mut self) {
        self.variables.push_back(HashMap::new());
    }

    #[inline]
    fn scope_down(&mut self) {
        self.variables
            .pop_back()
            .expect("Cannot scope down anymore.");
    }
}

impl<'a, 'ctx> Compiler<'a, 'ctx> {
    /// Enable debug.
    /// # Panics
    /// If this method is called before setting [`ctx`] or ['module'] (because debug info builder needs them internally).
    #[inline]
    pub fn enable_debug(&mut self, file_name: &str) -> &mut Self {
        let ctx = self
            .ctx
            .expect("Fatal: Compiler::debug must be called after Context is set.");
        let module = self
            .module
            .expect("Fatal: Compiler::debug must be called after Module is set.");
        let debug = DebugInfoFactory::new(ctx, module, file_name);
        self.debug = Some(Some(debug));
        self
    }

    #[inline]
    pub fn compile(&'a self, function: &Function) -> Result<FunctionValue<'ctx>> {
        let compiler = self.build()?;
        compile(compiler, function)
    }

    /// Run compiler with target information through [`TargetMachine`].
    /// # Panics
    /// If [`emit_obj`] is toggled to [`true`] in builder chain.
    #[inline]
    pub fn compile_with_target(
        &'a self,
        function: &Function,
        target: &TargetMachine,
    ) -> Result<FunctionValue<'ctx>> {
        let compiler = self.build()?;
        let data_layout = target.get_target_data().get_data_layout();
        let triple = target.get_triple();
        compiler.module.set_data_layout(&data_layout);
        compiler.module.set_triple(&triple);
        if !compiler.emit_obj {
            panic!("Please set emit_obj to true.");
        }
        compile(compiler, function)
    }
}

#[inline]
fn compile<'ctx>(
    mut compiler: IrGenerator<'_, 'ctx>,
    function: &Function,
) -> Result<FunctionValue<'ctx>> {
    match compiler.function_gen(function) {
        Err(CompileError::Fatal(fata)) => {
            panic!("Fatal: {fata:?}");
        }
        res => res,
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
        let fpm = PassManager::create(&module);
        let _compiler = Compiler::default()
            .builder(&builder)
            .context(&ctx)
            .module(&module)
            .function_pass_manager(&fpm)
            .build()
            .unwrap();
    }

    #[test]
    fn enable_debug() {
        let ctx = Context::create();
        let builder = ctx.create_builder();
        let module = ctx.create_module("test");
        let fpm = PassManager::create(&module);
        let _compiler = Compiler::default()
            .builder(&builder)
            .context(&ctx)
            .module(&module)
            .function_pass_manager(&fpm)
            .enable_debug("Cargo.toml") // need to specify exisiting file.
            .build()
            .unwrap();
    }
    #[should_panic]
    #[test]
    fn enable_debug_non_existing_path() {
        let ctx = Context::create();
        let builder = ctx.create_builder();
        let module = ctx.create_module("test");
        let fpm = PassManager::create(&module);
        let _compiler = Compiler::default()
            .builder(&builder)
            .context(&ctx)
            .module(&module)
            .function_pass_manager(&fpm)
            .enable_debug("__FooBar.txt") // need to specify exisiting file.
            .build()
            .unwrap();
    }
}
