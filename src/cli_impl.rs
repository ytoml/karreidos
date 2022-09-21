use clap::Parser as CliArgs;
use inkwell::targets::FileType;

use crate::{error::Error, Result};

#[derive(CliArgs, Debug)]
pub struct Arguments {
    #[clap(short = 'i', long = "it", takes_value = false)]
    interactive: bool,
    #[clap(default_value = "")]
    src_files: String,
    #[clap(short = 'o', long = "output")]
    output_file: Option<String>,
    #[clap(long = "emit")]
    emit: Option<String>,
    #[clap(short = 't', long = "target")]
    triple: Option<String>,
    #[clap(short = 'd', long = "debug-info", takes_value = false)]
    debug: bool,
}

#[derive(Debug)]
pub enum Run {
    Interactive,
    NonInteractive {
        src_files: Vec<String>,
        output_file: Option<String>,
        emit: Option<Emit>,
        triple: Option<String>,
    },
}

#[derive(Debug)]
pub enum Emit {
    Asm,
    Obj,
}
impl TryFrom<String> for Emit {
    type Error = Error;
    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        match value.as_str() {
            "asm" | "assembly" => Ok(Emit::Asm),
            "obj" | "object" => Ok(Emit::Obj),
            _ => Err(Error::InvalidArgFormat(value, String::from("--emit"))),
        }
    }
}
impl From<Emit> for FileType {
    fn from(e: Emit) -> Self {
        match e {
            Emit::Asm => FileType::Assembly,
            Emit::Obj => FileType::Object,
        }
    }
}

pub fn parse_arguments() -> Result<(Run, bool)> {
    let args = Arguments::parse();
    log::debug!("{args:?}");
    let Arguments {
        interactive,
        src_files,
        output_file,
        emit,
        triple,
        debug,
    } = args;
    if interactive {
        if !src_files.is_empty() {
            log::warn!("Source files are ignored in interactive mode.")
        }
        if output_file.is_some() {
            log::warn!("Output file path is ignored in interactive mode.")
        }
        if triple.is_some() {
            log::warn!(
                "Target triple is ignored in interactive mode (always fall back to host triple)."
            )
        }
        if let Some(format) = emit {
            log::warn!("Emit `{format}` is ignored in interactive model.")
        }
        Ok((Run::Interactive, debug))
    } else {
        if let Some("") = output_file.as_deref() {
            return Err(Error::OutputPathEmpty);
        }
        match emit.as_deref() {
            Some("obj") | Some("object") if output_file.is_none() => {
                return Err(Error::TryToEmitObjectToStdio);
            }
            _ => {}
        }
        if emit.is_none() && triple.is_some() {
            log::warn!("Target triple is ignored in non-emit mode.")
        }
        let src_files = src_files
            .split_ascii_whitespace()
            .map(|path| path.to_string())
            .collect();
        let emit = emit.map(TryInto::try_into).transpose()?;
        Ok((
            Run::NonInteractive {
                src_files,
                output_file,
                emit,
                triple,
            },
            debug,
        ))
    }
}
