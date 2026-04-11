mod config;
mod diagnostics_writer;
mod run;
mod validate;

use clap::{Parser, Subcommand};
use tracing_subscriber::fmt;

#[derive(Parser)]
#[command(
    name = "vonkarman",
    about = "Multi-precision pseudospectral Navier-Stokes solver"
)]
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
        /// FFT backend: auto, cufft, cpu. Overrides TOML config.
        #[arg(long, default_value = "")]
        backend: String,
        /// Path to a checkpoint file to restart from.
        #[arg(long)]
        restart: Option<String>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fmt::init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Run {
            config: config_path,
            backend,
            restart,
        } => {
            let contents = std::fs::read_to_string(&config_path)?;
            let mut config: config::ExperimentConfig = toml::from_str(&contents)?;
            if !backend.is_empty() {
                config.domain.backend = backend;
            }

            // Validate config before solver construction
            let errors = validate::validate_config(&config);
            if !errors.is_empty() {
                eprintln!("Configuration errors:");
                for e in &errors {
                    eprintln!("  - {e}");
                }
                std::process::exit(1);
            }

            run::run(&config, restart.as_deref())?;
        }
    }

    Ok(())
}
