use std::fmt::Display;

#[derive(Debug, Clone, Copy)]
pub struct SrcInfo {
    pub line: u32,
    pub col: u32,
}
impl SrcInfo {
    pub const fn new(line: u32, col: u32) -> Self {
        Self { line, col }
    }
}
impl Display for SrcInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

/// Define struct that wraps token or ast node with source location.
/// Note that you have to import [`crate::error::info::SrcInfo`] to scope where macro is expanded.
macro_rules! info_impl {
    ($Struct:ident, $member:ident) => {
        paste::paste! {
            #[derive(Debug, Clone)]
            pub struct [<$Struct Info>] {
                pub $member: $Struct,
                pub info: SrcInfo,
            }

            impl $Struct {
                #[inline]
                #[allow(dead_code)]
                pub const fn with_line_and_col(self, line: u32, col: u32) -> [<$Struct Info>] {
                    [<$Struct Info>] { $member: self, info: SrcInfo::new(line, col) }
                }

                #[inline]
                pub const fn with_info(self, info: SrcInfo) -> [<$Struct Info>] {
                    [<$Struct Info>] { $member: self, info }
                }
            }
        }
    };
}
