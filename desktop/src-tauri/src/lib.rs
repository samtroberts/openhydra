use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::{AppHandle, Manager, State};
use tauri::menu::{MenuBuilder, MenuItemBuilder};
use tauri::tray::TrayIconBuilder;

pub struct NodeState {
    dht_process: Mutex<Option<Child>>,
    node_process: Mutex<Option<Child>>,
}

impl Default for NodeState {
    fn default() -> Self {
        NodeState {
            dht_process: Mutex::new(None),
            node_process: Mutex::new(None),
        }
    }
}

#[tauri::command]
pub fn start_node(
    peer_id: String,
    dht_url: String,
    model_id: String,
    compaction_mode: Option<String>,   // "off" | "auto" | "on" (6.6)
    state: State<NodeState>,
) -> Result<String, String> {
    // Start openhydra-dht if dht_url is localhost
    if dht_url.contains("localhost") || dht_url.contains("127.0.0.1") {
        let dht = Command::new("openhydra-dht")
            .args(["--host", "127.0.0.1", "--port", "8468"])
            .spawn()
            .map_err(|e| format!("Failed to start DHT: {e}"))?;
        *state.dht_process.lock().unwrap() = Some(dht);
        std::thread::sleep(std::time::Duration::from_millis(1500));
    }

    // Start openhydra-node
    let mut cmd = Command::new("openhydra-node");
    cmd.args([
        "--peer-id",
        &peer_id,
        "--model-id",
        &model_id,
        "--dht-url",
        &dht_url,
        "--api-host",
        "127.0.0.1",
    ]);

    // Pass compaction mode when set and not "off" (6.6)
    if let Some(ref mode) = compaction_mode {
        if mode != "off" {
            cmd.args(["--kv-compaction-mode", mode.as_str()]);
        }
    }

    let node = cmd
        .spawn()
        .map_err(|e| format!("Failed to start node: {e}"))?;
    *state.node_process.lock().unwrap() = Some(node);

    Ok(format!("Node started: peer_id={peer_id}"))
}

#[tauri::command]
pub fn stop_node(state: State<NodeState>) -> Result<(), String> {
    if let Some(mut child) = state.node_process.lock().unwrap().take() {
        child.kill().ok();
    }
    if let Some(mut child) = state.dht_process.lock().unwrap().take() {
        child.kill().ok();
    }
    Ok(())
}

#[tauri::command]
pub fn is_node_running(state: State<NodeState>) -> bool {
    if let Ok(mut guard) = state.node_process.lock() {
        if let Some(child) = guard.as_mut() {
            return matches!(child.try_wait(), Ok(None));
        }
    }
    false
}

#[tauri::command]
pub fn get_node_pid(state: State<NodeState>) -> Option<u32> {
    state.node_process.lock().ok()?.as_ref().map(|c| c.id())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_sql::Builder::default().build())
        .manage(NodeState::default())
        .setup(|app| {
            // System tray menu (6.6)
            let quit = MenuItemBuilder::new("Quit OpenHydra").id("quit").build(app)?;
            let show = MenuItemBuilder::new("Show Window").id("show").build(app)?;
            let tray_menu = MenuBuilder::new(app).items(&[&show, &quit]).build()?;
            TrayIconBuilder::new()
                .icon(app.default_window_icon().unwrap().clone())
                .menu(&tray_menu)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "quit" => {
                        app.exit(0);
                    }
                    "show" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                    _ => {}
                })
                .build(app)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_node,
            stop_node,
            is_node_running,
            get_node_pid,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                // Kill node on window close
                if let Some(state) = window.try_state::<NodeState>() {
                    if let Some(mut child) = state.node_process.lock().unwrap().take() {
                        child.kill().ok();
                    }
                    if let Some(mut child) = state.dht_process.lock().unwrap().take() {
                        child.kill().ok();
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running OpenHydra desktop");
}
