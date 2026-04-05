use std::process::Command;

#[test]
fn cli_run_taylor_green() {
    let dir = std::env::temp_dir().join("vonkarman_cli_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let config_path = dir.join("test.toml");
    let output_dir = dir.join("output");

    std::fs::write(&config_path, format!(r#"
[run]
name = "cli-test"
output_dir = "{}"

[domain]
type = "periodic3d"
n = 8

[physics]
nu = 0.1

[initial_condition]
type = "taylor-green"

[termination]
max_steps = 10
"#, output_dir.display())).unwrap();

    // Build first, then run the binary directly
    let build = Command::new("cargo")
        .args(["build", "-p", "vonkarman-bin"])
        .current_dir(env!("CARGO_MANIFEST_DIR").replace("vonkarman-bin", ""))
        .output()
        .expect("failed to build");
    assert!(build.status.success(), "build failed: {}", String::from_utf8_lossy(&build.stderr));

    let workspace_root = env!("CARGO_MANIFEST_DIR").replace("vonkarman-bin", "");
    let binary = format!("{workspace_root}target/debug/vonkarman");

    let output = Command::new(&binary)
        .args(["run", "--config"])
        .arg(config_path.to_str().unwrap())
        .output()
        .expect("failed to run vonkarman");

    assert!(
        output.status.success(),
        "vonkarman exited with error:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    // Check diagnostics.parquet was created
    let parquet_path = output_dir.join("diagnostics.parquet");
    assert!(parquet_path.exists(), "diagnostics.parquet not created");
    let meta = std::fs::metadata(&parquet_path).unwrap();
    assert!(meta.len() > 0, "diagnostics.parquet is empty");

    let _ = std::fs::remove_dir_all(&dir);
}
