//! Every process the app starts goes through [`command`]: on Windows it gets `CREATE_NO_WINDOW`, so
//! no console window opens for it. The agents are console programs and the app is a windowed one, so
//! without the flag each agent got a console window of its own on top of the app (closing it stopped
//! sharing), and the engine probes flashed one on every poll (11 Nov Windows test, D-M1). A GUI
//! program it starts (an installer, an editor) still shows its own windows.

use std::ffi::OsStr;
use std::process::Command;

/// `CREATE_NO_WINDOW` (winbase.h).
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// A `Command` for `program` that opens no console window on Windows (a plain `Command` elsewhere).
pub fn command<S: AsRef<OsStr>>(program: S) -> Command {
    #[allow(unused_mut)]
    let mut c = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(CREATE_NO_WINDOW);
    }
    c
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_command_still_runs_and_returns_its_output() {
        #[cfg(windows)]
        let out = super::command("cmd").args(["/C", "echo", "ok"]).output().unwrap();
        #[cfg(not(windows))]
        let out = super::command("echo").arg("ok").output().unwrap();
        assert!(out.status.success());
        assert!(String::from_utf8_lossy(&out.stdout).contains("ok"));
    }
}
