macro_rules! info_impl {
    ($Struct:ident, $member:ident) => {
        paste::paste! {
            #[derive(Debug, Clone)]
            pub struct [<$Struct Info>] {
                pub $member: $Struct,
                pub line: u32,
                pub col: u32,
            }

            impl $Struct {
                #[inline]
                pub const fn with_src_info(self, line: u32, col: u32) -> [<$Struct Info>] {
                    [<$Struct Info>] { $member: self, line, col, }
                }
            }
        }
    };
}
