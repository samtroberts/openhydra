use sysinfo::System;

/// Returns total **physical** RAM in whole GB (rounded to nearest).
/// Uses sysinfo crate which reads hw.memsize on macOS — excludes swap.
#[tauri::command]
fn get_system_ram() -> u64 {
    let sys = System::new_all();
    let bytes = sys.total_memory(); // Physical RAM only, no swap
    // Round to nearest GB to handle slight OS reporting variations
    // (e.g. 8GB Mac reports 8,589,934,592 bytes = exactly 8 GB)
    (bytes + (512 * 1024 * 1024)) / (1024 * 1024 * 1024)
}

/// Switch the OpenHydra daemon between local and swarm mode.
///
/// Calls `POST http://127.0.0.1:{port}/v1/internal/mode` on the Python
/// sidecar.  The sidecar sets the 503 drain gate, swaps the active
/// inference engine, and returns `{"status": "ok", "mode": "..."}`.
#[tauri::command]
fn switch_mode(mode: String, port: u16, model_id: Option<String>) -> Result<String, String> {
    let url = format!("http://127.0.0.1:{}/v1/internal/mode", port);
    let mut body = serde_json::json!({ "mode": mode });
    if let Some(mid) = model_id {
        body["model_id"] = serde_json::Value::String(mid);
    }

    let resp = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(15))
        .send_json(&body)
        .map_err(|e| format!("mode switch failed: {}", e))?;

    let status = resp.status();
    let text = resp.into_string()
        .map_err(|e| format!("failed to read response: {}", e))?;

    if status != 200 {
        return Err(format!("mode switch returned HTTP {}: {}", status, text));
    }
    Ok(text)
}

/// Return the current mode (local or swarm) from the Python daemon.
#[tauri::command]
fn get_mode(port: u16) -> Result<String, String> {
    let url = format!("http://127.0.0.1:{}/v1/internal/mode", port);
    let resp = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(5))
        .call()
        .map_err(|e| format!("get mode failed: {}", e))?;

    resp.into_string()
        .map_err(|e| format!("failed to read response: {}", e))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![get_system_ram, switch_mode, get_mode])
        .setup(|_app| {
            #[cfg(debug_assertions)]
            {
                use tauri::Manager;
                let window = _app.get_webview_window("main").unwrap();
                window.open_devtools();
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
