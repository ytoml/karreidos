// This implelmentation thanks to
// https://github.com/TheDan64/inkwell/blob/56b01589448dc7978451c08c5d5c5294b11bdb4d/examples/kaleidoscope/main.rs
#[macro_use]
extern crate derive_builder;

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{self, Read};

use cli_impl::Run;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::passes::PassManager;
use inkwell::targets::{InitializationConfig, Target};
use inkwell::values::{AnyValue, FunctionValue};
use inkwell::OptimizationLevel;
use lexer::Lexer;
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

fn get_prec() -> HashMap<char, i32> {
    let mut prec = HashMap::with_capacity(7);
    prec.insert('=', 0);
    prec.insert('<', 10);
    prec.insert('>', 10);
    prec.insert('+', 20);
    prec.insert('-', 20);
    prec.insert('*', 40);
    prec.insert('/', 40);
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
) -> compiler::Result<FunctionValue<'ctx>> {
    let builder = ctx.create_builder();
    let fpm = PassManager::create(module);
    fpm.add_promote_memory_to_register_pass();
    fpm.add_instruction_combining_pass();
    fpm.add_reassociate_pass();
    fpm.add_gvn_pass(); // Eliminate Common SubExpressions
    fpm.add_cfg_simplification_pass();
    fpm.initialize();

    Compiler::default()
        .builder(&builder)
        .context(ctx)
        .module(module)
        .function_pass_manager(&fpm)
        .compile(function)
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
    let mut all_expr = Vec::new();
    'main: loop {
        eprint!("?>> ");

        // Create module for every loop because each ExecutionEngine takes
        // ownership (not in Rust's semantics but in LLVM's) and it never
        // allows any changes to alreadly owned module.
        let module = ctx.create_module("repl");

        // Adding previous functions to new module is needed (reason is explained above).
        for function in all_expr.iter() {
            let _ = compile(&ctx, &module, function)
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
        match compile(&ctx, &module, &function) {
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

fn run_non_interactive(ctx: Context, file_name: &str) -> Result<()> {
    let mod_name = file_name
        .find('.')
        .map(|i| &file_name[..i])
        .unwrap_or(file_name);
    let mut src_file = OpenOptions::new().read(true).open(file_name)?;
    let mut src = String::new();
    src_file.read_to_string(&mut src)?;

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
    let module = ctx.create_module(mod_name);
    let func = compile(&ctx, &module, &function)?;
    log::info!("Compilation Succeeded!");
    log::info!("Generated IR:");
    func.print_to_stderr();
    Ok(())
}

fn main() -> Result<()> {
    log_impl::init_logger();

    let config = InitializationConfig {
        base: true,
        asm_parser: true,
        asm_printer: true,
        disassembler: false,
        info: false,
        machine_code: false,
    };

    #[cfg(target_arch = "aarch64")]
    Target::initialize_aarch64(&config);

    #[cfg(not(target_arch = "aarch64"))]
    log::warn!("Jit for non-aarch64 is currently not supported!");

    let ctx = Context::create();
    match cli_impl::parse_arguments() {
        Run::Interactive => run_interactive(ctx),
        Run::NonInteractive(files) => match files.len() {
            0 => Err(Error::NoSourceProvided),
            1 => run_non_interactive(ctx, files[0].as_str()),
            _ => {
                log::error!("Currently, more than single file is not supported.");
                Ok(())
            }
        },
    }
}
