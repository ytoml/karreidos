use clap::Parser as CliArgs;

#[derive(CliArgs, Debug)]
pub struct Arguments {
    #[clap(short = 'i', long = "it", takes_value = false)]
    interactive: bool,
    #[clap(default_value = "")]
    src_files: String,
}

#[derive(Debug)]
pub enum Run {
    Interactive,
    NonInteractive(Vec<String>),
}

pub fn parse_arguments() -> Run {
    let args = Arguments::parse();
    log::debug!("{args:?}");
    let Arguments {
        interactive,
        src_files,
    } = args;
    if interactive {
        Run::Interactive
    } else {
        let files = src_files
            .split_ascii_whitespace()
            .map(|path| path.to_string())
            .collect();
        Run::NonInteractive(files)
    }
}
