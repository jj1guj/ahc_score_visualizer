use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rayon::prelude::*;
use serde::Deserialize;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

#[derive(Clone)]
struct Result {
    input_file: String,
    score: usize,
    score_string: String,
    visualizer: String,
}

#[derive(Clone, Deserialize)]
struct Config {
    paths: PathsConfig,
    tester: TesterConfig,
    #[serde(default)]
    parallel: Option<ParallelConfig>,
}

#[derive(Clone, Deserialize)]
struct ParallelConfig {
    num_threads: Option<usize>,
}

#[derive(Clone, Deserialize)]
struct PathsConfig {
    input_dir: String,
    output_dir: String,
    visualizer_dir: String,
    html_output: String,
    #[serde(default)]
    answers_dir: Option<String>,
}

#[derive(Clone, Deserialize)]
struct TesterConfig {
    command: String,
    script: Option<String>,
    solver_script: Option<String>,
}

fn main() {
    // Parse command line arguments for --config option
    let args: Vec<String> = env::args().collect();
    let config_path = {
        let mut path = "./config.toml".to_string();
        let mut i = 1;
        while i < args.len() {
            if args[i] == "--config" {
                if i + 1 < args.len() {
                    path = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: --config requires a path argument");
                    return;
                }
            } else if args[i].starts_with("--config=") {
                path = args[i]["--config=".len()..].to_string();
                i += 1;
            } else {
                i += 1;
            }
        }
        path
    };

    // Load configuration
    let config = match load_config(&config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Error loading config: {}", e);
            return;
        }
    };

    let input_dir = &config.paths.input_dir;
    let output_dir = &config.paths.output_dir;
    let visualizer_dir = &config.paths.visualizer_dir;
    let html_output = &config.paths.html_output;
    let tools_dir = Path::new(input_dir)
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();

    // Create output directories
    fs::create_dir_all(output_dir).ok();
    fs::create_dir_all(visualizer_dir).ok();

    // Get input files
    let mut input_files = match get_input_files(input_dir) {
        Ok(files) => files,
        Err(e) => {
            eprintln!("Error reading input files: {}", e);
            return;
        }
    };

    // Sort input files by number
    input_files.sort_by_key(|f| extract_number(f));

    // Process files in parallel, visualize as each completes
    let total_inputs = input_files.len() as u64;
    let score_bar = ProgressBar::new(total_inputs);
    score_bar.set_draw_target(ProgressDrawTarget::stderr());
    score_bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.cyan} {msg:<12} {bar:40.cyan/blue} {pos:>3}/{len:<3} {percent:>3}% | {per_sec} | ETA {eta}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    score_bar.set_message("Scoring");
    let vis_bar = ProgressBar::new(total_inputs);
    vis_bar.set_draw_target(ProgressDrawTarget::stderr());
    vis_bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} {msg:<12} {bar:40.green/blue} {pos:>3}/{len:<3} {percent:>3}% | {per_sec} | ETA {eta}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    vis_bar.set_message("Visualizing");
    let (tx, rx) = mpsc::channel::<Result>();
    let input_files_for_thread = input_files.clone();
    let output_dir_for_thread = output_dir.to_string();
    let tools_dir_for_thread = tools_dir.clone();
    let config_for_thread = config.clone();

    let producer = thread::spawn(move || {
        let num_threads = config_for_thread
            .parallel
            .as_ref()
            .and_then(|p| p.num_threads)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        pool.install(|| {
            input_files_for_thread
                .par_iter()
                .for_each_with(tx, |sender, input_file| {
                    let result = process_file(
                        input_file,
                        &output_dir_for_thread,
                        &config_for_thread,
                        &tools_dir_for_thread,
                    );
                    let _ = sender.send(result);
                });
        });
    });

    let mut results: Vec<Result> = Vec::with_capacity(total_inputs as usize);
    for result in rx {
        score_bar.inc(1);
        let result = visualize_result(result, output_dir, visualizer_dir, &tools_dir);
        vis_bar.inc(1);
        results.push(result);
    }
    let _ = producer.join();
    score_bar.finish_with_message("Scoring done");
    vis_bar.finish_with_message("Visualizing done");

    // Sort results by file number
    results.sort_by_key(|r| extract_number(&r.input_file));

    // Calculate total score
    let total_score: usize = results.iter().map(|r| r.score).sum();

    // Get current timestamp in JST
    let jst_now = chrono::Local::now();
    let timestamp = jst_now.format("%Y-%m-%d %H:%M:%S").to_string();

    // Generate HTML
    generate_html(&results, total_score, &timestamp, html_output);

    // Copy solver output files to answers directory
    if let Some(answers_dir) = &config.paths.answers_dir {
        fs::create_dir_all(answers_dir).ok();
        match fs::read_dir(output_dir) {
            Ok(entries) => {
                for entry in entries {
                    if let Ok(entry) = entry {
                        let src = entry.path();
                        if src.is_file() {
                            let dest = Path::new(answers_dir).join(entry.file_name());
                            if let Err(e) = fs::copy(&src, &dest) {
                                eprintln!("Error copying {}: {}", src.display(), e);
                            }
                        }
                    }
                }
            }
            Err(e) => eprintln!("Error reading output dir: {}", e),
        }
        eprintln!("Answers saved to {}", answers_dir);
    }

    println!("Total Score: {}", total_score);
    println!("Results saved to {}", html_output);
}

fn get_input_files(dir: &str) -> io::Result<Vec<String>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "txt") {
            files.push(path.to_string_lossy().to_string());
        }
    }
    Ok(files)
}

fn extract_number(filename: &str) -> usize {
    let base = Path::new(filename).file_name().unwrap().to_string_lossy();
    let parts: Vec<&str> = base.split('.').collect();
    if !parts.is_empty() {
        parts[0].parse::<usize>().unwrap_or(0)
    } else {
        0
    }
}

fn format_score(score: usize) -> String {
    format!("{}", score)
}

fn process_file(input_file: &str, output_dir: &str, config: &Config, _tools_dir: &Path) -> Result {
    let base_name = Path::new(input_file)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let output_file = format!("{}/{}", output_dir, base_name);

    // Open input file
    let input_data = match fs::read(input_file) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading input file: {}", e);
            return Result {
                input_file: input_file.to_string(),
                score: 0,
                score_string: "0".to_string(),
                visualizer: String::new(),
            };
        }
    };

    // Run tester command
    let mut command = config.tester.command.clone();
    if let Some(script) = config.tester.script.as_deref() {
        command = command.replace("{{script}}", script);
    }
    if let Some(solver_script) = config.tester.solver_script.as_deref() {
        command = command.replace("{{solver_script}}", solver_script);
    }
    let parts: Vec<&str> = command.split_whitespace().collect();

    if parts.is_empty() {
        return Result {
            input_file: input_file.to_string(),
            score: 0,
            score_string: "0".to_string(),
            visualizer: String::new(),
        };
    }

    let mut cmd = Command::new(parts[0]);
    cmd.args(&parts[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            eprintln!("Error starting tester: {}", e);
            return Result {
                input_file: input_file.to_string(),
                score: 0,
                score_string: "0".to_string(),
                visualizer: String::new(),
            };
        }
    };

    // Write input data to stdin
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(&input_data);
    }

    // Get output
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(e) => {
            eprintln!("Error waiting for tester: {}", e);
            return Result {
                input_file: input_file.to_string(),
                score: 0,
                score_string: "0".to_string(),
                visualizer: String::new(),
            };
        }
    };

    // Save stdout to file
    let _ = fs::write(&output_file, &output.stdout);

    // Parse score from stderr
    let mut score = 0;
    let stderr_string = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        eprintln!(
            "[WARN] Tester failed for {}: exit code {:?}",
            input_file,
            output.status.code()
        );
        eprintln!("[WARN] stderr: {}", stderr_string);
    }
    for line in stderr_string.lines() {
        if line.starts_with("Score = ") {
            let score_str = line.trim_start_matches("Score = ");
            score = score_str.parse::<usize>().unwrap_or(0);
        }
    }

    Result {
        input_file: input_file.to_string(),
        score,
        score_string: format_score(score),
        visualizer: String::new(),
    }
}

fn visualize_result(
    mut result: Result,
    output_dir: &str,
    visualizer_dir: &str,
    _tools_dir: &Path,
) -> Result {
    let base_name = Path::new(&result.input_file)
        .file_name()
        .unwrap()
        .to_string_lossy();
    let visualizer_file = format!("{}/{}", visualizer_dir, base_name.replace(".txt", ".html"));

    let output = Command::new("./target/release/vis")
        .arg(&result.input_file)
        .arg(format!("{}/{}", output_dir, base_name))
        .output();

    if let Ok(out) = output {
        if !out.status.success() {
            eprintln!(
                "Error running visualizer for {}: {}",
                base_name,
                String::from_utf8_lossy(&out.stderr)
            );
            return result;
        }
        // vis writes vis.html in the current directory
        let vis_html = Path::new("vis.html");
        if vis_html.exists() {
            if let Err(_e) = fs::rename(&vis_html, &visualizer_file) {
                // rename may fail across filesystems, fall back to copy+remove
                if let Err(e) = fs::copy(&vis_html, &visualizer_file) {
                    eprintln!("Error copying vis.html: {}", e);
                    return result;
                }
                let _ = fs::remove_file(&vis_html);
            }
            result.visualizer = format!("visualizations/{}", base_name.replace(".txt", ".html"));
        } else {
            eprintln!("vis.html not found");
        }
    } else {
        eprintln!("Error running visualizer");
    }

    result
}

fn generate_html(results: &[Result], total_score: usize, timestamp: &str, output_path: &str) {
    let mut html = String::from(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Score Results</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        th {
            background-color: #f2f2f2;
            text-align: left;
            cursor: pointer;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
    </style>
    <script>
        let sortOrder = {
            score: 'desc',
            file: 'asc'
        };

        function sortTable(columnIndex, isNumeric, key) {
            const table = document.getElementById("resultsTable");
            const rows = Array.from(table.rows).slice(1);
            const order = sortOrder[key] === 'asc' ? 1 : -1;

            rows.sort((a, b) => {
                const cellA = a.cells[columnIndex].innerText;
                const cellB = b.cells[columnIndex].innerText;
                if (isNumeric) {
                    return order * (parseInt(cellA.replace(/,/g, '')) - parseInt(cellB.replace(/,/g, '')));
                }
                return order * cellA.localeCompare(cellB);
            });

            rows.forEach(row => table.appendChild(row));
            sortOrder[key] = sortOrder[key] === 'asc' ? 'desc' : 'asc';

            const sortIndicator = document.getElementById("sortIndicator");
            sortIndicator.innerText = 'Sorted by ' + key + ' (' + (sortOrder[key] === 'asc' ? 'Ascending' : 'Descending') + ')';
        }
    </script>
</head>
<body>
    <h1>Score Results</h1>
    <p>Total Score: "#,
    );

    html.push_str(&format!("{}", total_score));
    html.push_str(&format!(
        r#"</p>
    <p>Timestamp (JST): {}</p>
    <p id="sortIndicator">Sorted by file (Ascending)</p>
    <table id="resultsTable">
        <thead>
            <tr>
                <th onclick="sortTable(0, false, 'file')">Input File</th>
                <th onclick="sortTable(1, true, 'score')">Score</th>
                <th>Visualizer</th>
            </tr>
        </thead>
        <tbody>
"#,
        timestamp
    ));

    for result in results {
        html.push_str(&format!(
            r#"            <tr>
                <td>{}</td>
                <td>{}</td>
                <td><a href="{}" target="_blank">View</a></td>
            </tr>
"#,
            result.input_file, result.score_string, result.visualizer
        ));
    }

    html.push_str(
        r#"        </tbody>
    </table>
</body>
</html>
"#,
    );

    if let Err(e) = fs::write(output_path, html) {
        eprintln!("Error writing HTML file: {}", e);
    }
}

fn load_config(path: &str) -> io::Result<Config> {
    let config_str = fs::read_to_string(path)?;
    let config = toml::from_str(&config_str)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(config)
}
