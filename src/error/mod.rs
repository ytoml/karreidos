#[macro_use]
pub mod info;

use crate::{compiler::CompileError, lexer::LexError, parser::ParseError};
use paste::paste;
use thiserror::Error as ThisError;

macro_rules! from_inner {
    (
        $(#[$outer:meta])*
        $v:vis enum $name:ident {
            $(
                // from modules
                $(#[$err_att:meta])*
                $proc:ident
            ),* $(,)?
            $(
                // top-level errors
                [top]
                $(
                    // Ugly Hack: Walk around single case where using #[from]
                    $(#[$top_att:meta])*
                    $variant:ident $((#[from] $inner:path))? $(($($tuple_elem:path),+ $(,)?))?
                ),* $(,)?
            )?
        }
    ) => {
        paste! {
            $(#[$outer])*
            $v enum $name {
                $(
                    $(#[$err_att])*
                    $proc([<$proc $name>]),
                )*
                $(
                    $(
                        $(#[$top_att])*
                        $variant $((#[from] $inner))? $(($($tuple_elem),+))?,
                    )*
                )?
            }
            $(impl From<[<$proc $name>]> for $name {
                fn from(value: [<$proc $name>]) -> Self {
                    Self::$proc(value)
                }
            })*
        }
    };
}

from_inner! {
    #[derive(Debug, ThisError)]
    pub enum Error {
        #[error("Lexical analysis failed: {0:?}")]
        Lex,
        #[error("Parse failed: {0:?}")]
        Parse,
        #[error("Parse failed: {0:?}")]
        Compile,

        [top]
        #[error("No source file provided.")]
        NoSourceProvided,
        #[error(transparent)]
        Io(#[from] std::io::Error), // (std::io::Error)
        #[error("Failed to write to file ({0}).")]
        OutputToFileFailed(String),
        #[error("Output path is specified with empty string.")]
        OutputPathEmpty,
        #[error("Cannot emit object file to stdio.")]
        TryToEmitObjectToStdio,
        #[error("Invalid target specified `{0}`.")]
        InvalidTarget(String),
        #[error("Invalid argument format `{0}` for `{1}`.")]
        InvalidArgFormat(String, String)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test() {
        let _err: Error = LexError::BlockCommentNotTerminated.into();
    }
}
