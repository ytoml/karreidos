// These implementation follows workaround proposed in https://github.com/TheDan64/inkwell/issues/258
use llvm_sys::support::LLVMAddSymbol;
use std::ffi::{c_void, CStr};

struct BuiltinFunction {
    name: &'static CStr,
    ptr: *mut c_void,
}

macro_rules! cstr {
    ($str:expr) => {
        unsafe { CStr::from_bytes_with_nul_unchecked(concat!($str, "\0").as_bytes()) }
    };
}

macro_rules! builtins {
    ($(pub fn $funcname:ident(
        $($arg:ident: $argty:ty),* $(,)?
    ) $(-> $($ret:ty)*)?
    $body:block)*
) => {
    $(
        mod $funcname {
            use std::ffi::{CStr, c_void};
            use super::BuiltinFunction;
            const NAME: &CStr = cstr!(stringify!($funcname));
            #[no_mangle]
            extern "C" fn $funcname($($arg: $argty,)*) $(-> $($ret)*)? $body
            pub(super) const FN: BuiltinFunction = BuiltinFunction {
                name: NAME,
                ptr: $funcname as *mut c_void,
            };
        }
    )*

        pub fn init() {
            for func in [$($funcname::FN,)*] {
                unsafe {
                    LLVMAddSymbol(func.name.as_ptr(), func.ptr);
                }
            }
        }
    };
}

macro_rules! print_flush {
    ( $( $x:expr ),* ) => {{
        use std::io::Write;
        print!( $($x, )* );
        std::io::stdout().flush().expect("Could not flush to standard output.");
    }};
}

builtins! {
    pub fn putchard(c: f64) -> f64 {
        print_flush!("{}", c as u8 as char);
        c
    }
    pub fn printd(c: f64) -> f64 {
        println!("{c}");
        c
    }
}
