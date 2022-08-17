// This implelmentation thanks to
// https://github.com/TheDan64/inkwell/blob/56b01589448dc7978451c08c5d5c5294b11bdb4d/examples/kaleidoscope/main.rs
#[macro_use]
extern crate derive_builder;

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{self, Read};

use cli_impl::Run;
use inkwell::context::Context;
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

fn compile(ctx: &Context, mod_name: &str, function: &Function) -> compiler::Result<()> {
    let builder = ctx.create_builder();
    let module = ctx.create_module(mod_name);
    let result = Compiler::default()
        .builder(&builder)
        .context(ctx)
        .module(&module)
        .compile(function)?;
    log::info!("{result:?}");
    Ok(())
}

fn run_interactive(ctx: Context) -> Result<()> {
    'main: loop {
        eprint!("?> ");
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
        }
        let function = match parse(tokens) {
            Ok(Some(GlobalVar::Function(func))) => {
                log::info!("{func:#?}");
                func
            }
            Ok(None) => {
                log::info!("Empty.");
                continue;
            }
            Err(err) => {
                log::error!("Parse failed: {err:?}");
                continue;
            }
        };
        if function.is_anonymous() {
        } else if let Err(err) = compile(&ctx, "repr", &function) {
            log::error!("{err:?}");
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
    log::info!("{tokens:#?}");
    let result = parse(tokens);
    let function = if let Ok(Some(GlobalVar::Function(func))) = result {
        log::info!("{func:#?}");
        func
    } else {
        result?;
        log::info!("Funtion with no content.");
        return Ok(());
    };
    compile(&ctx, mod_name, &function)?;
    log::info!("Compilation OK!");
    Ok(())
}

fn main() -> Result<()> {
    log_impl::init_logger();
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
