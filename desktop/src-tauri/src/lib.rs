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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![get_system_ram])
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
