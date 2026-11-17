use clap::{Parser, ValueEnum};
use poissonfusion::{BlendConfig, blend};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Preset {
    Faithful,
    Balanced,
    Seamless,
}

#[derive(Debug, Parser)]
#[command(
    name = "poissonfusion",
    about = "Identity-preserving screened-harmonic image compositing"
)]
struct Arguments {
    #[arg(long)]
    background: PathBuf,
    #[arg(long)]
    foreground: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 0)]
    x: i32,
    #[arg(long, default_value_t = 0)]
    y: i32,
    #[arg(long, value_enum, default_value_t = Preset::Balanced)]
    preset: Preset,
    #[arg(long)]
    appearance_sigma: Option<f32>,
    #[arg(long)]
    integration_length: Option<f32>,
    #[arg(long)]
    diagnostics: Option<PathBuf>,
    /// Repeat the in-memory blend and report the median runtime.
    #[arg(long, default_value_t = 1)]
    repeat: usize,
}

#[derive(Serialize)]
struct Diagnostics<'a> {
    runtime_ms_median: f64,
    repeat: usize,
    solver: &'a poissonfusion::BlendStats,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    if arguments.repeat == 0 {
        return Err("--repeat must be at least 1".into());
    }
    let background = image::open(&arguments.background)?.to_rgb8();
    let foreground = image::open(&arguments.foreground)?.to_rgba8();
    let mut config = match arguments.preset {
        Preset::Faithful => BlendConfig::faithful(),
        Preset::Balanced => BlendConfig::default(),
        Preset::Seamless => BlendConfig::seamless(),
    };
    if let Some(value) = arguments.appearance_sigma {
        config.appearance_sigma = value;
    }
    if let Some(value) = arguments.integration_length {
        config.integration_length = value;
    }

    let mut durations = Vec::with_capacity(arguments.repeat);
    let mut last_result = None;
    for _ in 0..arguments.repeat {
        let started = Instant::now();
        let result = blend(&background, &foreground, (arguments.x, arguments.y), config)?;
        durations.push(started.elapsed().as_secs_f64() * 1000.0);
        last_result = Some(result);
    }
    durations.sort_by(f64::total_cmp);
    let runtime_ms_median = durations[durations.len() / 2];
    let result = last_result.expect("repeat is validated as non-zero");
    if let Some(parent) = arguments.output.parent() {
        fs::create_dir_all(parent)?;
    }
    result.image.save(&arguments.output)?;
    let diagnostics = Diagnostics {
        runtime_ms_median,
        repeat: arguments.repeat,
        solver: &result.stats,
    };
    let json = serde_json::to_string_pretty(&diagnostics)?;
    if let Some(path) = arguments.diagnostics {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, format!("{json}\n"))?;
    }
    println!("{json}");
    Ok(())
}
