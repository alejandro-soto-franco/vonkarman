mod config;
mod diagnostics_writer;
mod run;

use clap::{Parser, Subcommand};
use tracing_subscriber::fmt;

#[derive(Parser)]
#[command(name = "vonkarman", about = "Multi-precision pseudospectral Navier-Stokes solver")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a simulation from a TOML config file.
    Run {
        /// Path to the experiment TOML config.
        #[arg(short, long)]
        config: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fmt::init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Run { config: config_path } => {
            let contents = std::fs::read_to_string(&config_path)?;
            let config: config::ExperimentConfig = toml::from_str(&contents)?;
            run::run(&config)?;
        }
    }

    Ok(())
}
