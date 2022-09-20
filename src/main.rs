// This implelmentation thanks to
// https://github.com/TheDan64/inkwell/blob/56b01589448dc7978451c08c5d5c5294b11bdb4d/examples/kaleidoscope/main.rs
#[macro_use]
extern crate derive_builder;

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Read, Write};

use cli_impl::{Emit, Run};
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::passes::PassManager;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::values::{AnyValue, FunctionValue};
use inkwell::OptimizationLevel;
use lexer::Lexer;
use parser::ast::BinOp;
use parser::{Function, Parser};

use crate::compiler::Compiler;
use crate::error::Error;
use crate::lexer::token::Token;
use crate::parser::GlobalVar;

mod cli_impl;
mod compiler;
mod error;
mod lexer;
mod log_impl;
mod parser;
mod preprocessor;

const ANONYMOUS_FN_NAME: &str = "__anonymous__"; // Empty string seems to be invalid for LLVM function name.
type Result<T> = std::result::Result<T, Error>;

fn get_prec() -> HashMap<BinOp, i32> {
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
}

fn lex(src: &str) -> lexer::Result<Vec<Token>> {
    Lexer::new(src).into_iter().collect()
}

fn parse(tokens: Vec<Token>) -> parser::Result<Option<GlobalVar>> {
    Parser::new(tokens, get_prec()).parse()
}

// Note: Currently, "module" must live longer than returning `FunctionValue`
// or it will lead to SEGV when printing built LLVM IRs.
// refer to: https://github.com/TheDan64/inkwell/issues/343
fn compile<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    function: &Function,
    target: Option<&TargetMachine>,
) -> compiler::Result<FunctionValue<'ctx>> {
    let builder = ctx.create_builder();
    let fpm = PassManager::create(module);
    fpm.add_promote_memory_to_register_pass();
    fpm.add_instruction_combining_pass();
    fpm.add_reassociate_pass();
    fpm.add_gvn_pass(); // Eliminate Common SubExpressions
    fpm.add_cfg_simplification_pass();
    fpm.initialize();

    let mut compiler = Compiler::default();
    compiler
        .builder(&builder)
        .context(ctx)
        .module(module)
        .function_pass_manager(&fpm)
        .emit_obj(target.is_some());

    if let Some(target) = target {
        compiler.compile_with_target(function, target)
    } else {
        compiler.compile(function)
    }
}

fn eval_anonymous_func(
    module: Module, // maybe better to take ownership of module to force no usage after creating exec engine.
) {
    type AnonymousFunction = unsafe extern "C" fn() -> f64;
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();
    let maybe_fn = unsafe { ee.get_function::<AnonymousFunction>(ANONYMOUS_FN_NAME) }
        .expect("Only module with anonymous function alreadly registered is available for jit.");
    let result = unsafe { maybe_fn.call() };
    log::info!("{result}");
}

fn run_interactive(ctx: Context) -> Result<()> {
    let config = InitializationConfig {
        base: true,
        asm_parser: true,
        asm_printer: true,
        disassembler: false,
        info: false,
        machine_code: false,
    };
    Target::initialize_all(&config);

    let mut all_expr = Vec::new();
    'main: loop {
        eprint!("?>> ");

        // Create module for every loop because each ExecutionEngine takes
        // ownership (not in Rust's semantics but in LLVM's) and it never
        // allows any changes to alreadly owned module.
        let module = ctx.create_module("repl");

        // Adding previous functions to new module is needed (reason is explained above).
        for function in all_expr.iter() {
            let _ = compile(&ctx, &module, function, None)
                .expect("Failed to compile function which was previously compiled successfully.");
        }

        let mut tokens = vec![];
        let mut first_loop = true;
        loop {
            let mut input = String::new();
            io::stdin()
                .read_line(&mut input)
                .expect("Could not read from stdin.");
            let src = input.trim();
            match src {
                "exit" | "quit" if first_loop => {
                    break 'main;
                }
                "" => break,
                _ => {
                    first_loop = false;
                }
            }
            let mut new_tokens = lex(&input)?;
            tokens.append(&mut new_tokens);
            eprint!("... ");
        }
        let function = match parse(tokens) {
            Ok(Some(GlobalVar::Function(func))) => {
                log::debug!("{func:#?}");
                func
            }
            Ok(None) => {
                log::debug!("Empty.");
                continue;
            }
            Err(err) => {
                log::error!("Parse failed: {err:?}");
                continue;
            }
        };
        match compile(&ctx, &module, &function, None) {
            Ok(func) => {
                log::debug!("Compile succeeded!");
                log::debug!(
                    "Generated IR: {}",
                    func.print_to_string().to_string().trim_end()
                );

                // Anonymous function is "top level expression" and we evaluate it (JIT exec) immediately.
                if function.is_anonymous() {
                    eval_anonymous_func(module);
                } else {
                    all_expr.push(function);
                }
            }
            Err(err) => log::error!("{err:?}"),
        }
    }
    eprintln!("Bye.");
    Ok(())
}

/// Compile source and emit.
/// # Panics
/// If [`output_file.is_none() && emit_obj`] is [`true`] (internal error).
fn run_non_interactive(
    ctx: Context,
    src_file: &str,
    output_file: Option<String>,
    emit: Option<Emit>,
    triple: Option<String>,
) -> Result<()> {
    if !src_file.ends_with(".krr") {
        log::warn!("file `{src_file}` seems not to be Karreidos source file. Is it really OK to compile this texts?(y/n)");
        loop {
            let mut input = String::new();
            io::stdin()
                .read_line(&mut input)
                .expect("Could not read from stdin.");
            match input.trim() {
                "y" => break,
                "n" => {
                    log::info!("Abort compilation.");
                    return Ok(());
                }
                _ => log::warn!("Specify y or n."),
            }
        }
    }
    let mod_name = src_file
        .find('.')
        .map(|i| &src_file[..i])
        .unwrap_or(src_file);
    let mut src_file = OpenOptions::new().read(true).open(src_file)?;
    let mut src = String::new();
    src_file.read_to_string(&mut src)?;

    let module = ctx.create_module(mod_name);
    let target = if emit.is_some() {
        let config = InitializationConfig {
            base: true,
            asm_parser: true,
            asm_printer: true,
            disassembler: false,
            info: true, // need to get Target from triple etc.
            machine_code: true,
        };
        Target::initialize_all(&config);
        let (triple, cpu) = if let Some(triple) = triple.as_deref() {
            (TargetTriple::create(triple), String::from("generic"))
        } else {
            (
                TargetMachine::get_default_triple(),
                TargetMachine::get_host_cpu_name().to_string(),
            )
        };
        log::debug!("Target cpu: {cpu}");
        log::debug!("Target triple: {}", triple.to_string());
        let target =
            Target::from_triple(&triple).map_err(|_| Error::InvalidTarget(triple.to_string()))?;
        let target_machine = target
            .create_target_machine(
                &triple,
                &cpu,
                "",
                OptimizationLevel::Default,
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or_else(|| Error::InvalidTarget(triple.to_string()))?;
        Some(target_machine)
    } else {
        None
    };

    let tokens = lex(&src)?;
    log::debug!("{tokens:#?}");
    let result = parse(tokens);
    let function = if let Ok(Some(GlobalVar::Function(func))) = result {
        log::debug!("{func:#?}");
        func
    } else {
        result?;
        log::debug!("Funtion with no content.");
        return Ok(());
    };
    let func = compile(&ctx, &module, &function, target.as_ref())?;
    log::info!("Compilation Succeeded!");
    match (emit, output_file) {
        (Some(Emit::Obj), Some(output_file)) => {
            let target = target.unwrap();
            target
                .write_to_file(&module, FileType::Object, output_file.as_ref())
                .map_err(|s| Error::OutputToFileFailed(s.to_string()))?;
        }
        (Some(Emit::Asm), output_file) => {
            let target = target.unwrap();
            if let Some(output_file) = output_file {
                target
                    .write_to_file(&module, FileType::Assembly, output_file.as_ref())
                    .map_err(|s| Error::OutputToFileFailed(s.to_string()))?;
            } else {
                let buf = target
                    .write_to_memory_buffer(&module, FileType::Assembly)
                    .map_err(|s| Error::OutputToFileFailed(s.to_string()))?;
                let asm = String::from_utf8_lossy(buf.as_slice()).to_string();
                log::info!("Generated ASM:\n{asm}");
            }
        }
        (None, output_file) => {
            let llvm_ir = func.print_to_string().to_string();
            if let Some(file) = output_file.as_deref() {
                let file = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(file)?;
                let mut writer = BufWriter::new(file);
                writer.write_all(llvm_ir.as_bytes())?;
            } else {
                log::info!("Generated IR:\n{llvm_ir}");
            }
        }
        _ => unreachable!(), // Invalid combination of Some(Emit::Obj) and None(no output file) is already checked in cli_impl
    }
    Ok(())
}

fn main() -> Result<()> {
    log_impl::init_logger();

    let ctx = Context::create();
    match cli_impl::parse_arguments()? {
        Run::Interactive => run_interactive(ctx),
        Run::NonInteractive {
            src_files,
            output_file,
            emit,
            triple,
        } => match src_files.len() {
            0 => Err(Error::NoSourceProvided),
            1 => run_non_interactive(ctx, src_files[0].as_str(), output_file, emit, triple),
            _ => {
                log::error!("Currently, more than single file is not supported.");
                Ok(())
            }
        },
    }
}
