use std::env;
use std::io::Write;

use env_logger::fmt::Color;
use log::Level;

#[cfg(debug_assertions)]
const LEVEL: &str = "debug";
#[cfg(not(debug_assertions))]
const LEVEL: &str = "info";

fn level_to_color(level: Level) -> Color {
    match level {
        Level::Info | Level::Debug => Color::Cyan,
        Level::Warn => Color::Yellow,
        Level::Error => Color::Red,
        _ => Color::White,
    }
}

pub fn init_logger() {
    env::set_var("RUST_LOG", LEVEL);
    env_logger::Builder::from_default_env()
        .format(|buf, record| {
            let color = level_to_color(record.level());
            let mut style = buf.style();
            style.set_color(color).set_bold(true);

            if let Level::Debug = record.level() {
                let line = record
                    .line()
                    .map(|l| l.to_string())
                    .unwrap_or_else(|| "n/a".to_string());
                let file = record.file().unwrap_or("unknown");
                writeln!(
                    buf,
                    "[{}] {}:{}\n{}\n",
                    style.value("DEBUG"),
                    file,
                    line,
                    record.args()
                )
            } else {
                writeln!(buf, "[{}] {}", style.value(record.level()), record.args())
            }
        })
        .init();
}
