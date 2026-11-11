//! Windows: the app's agents live in a Job Object that kills them when the app's last handle to it
//! closes, which happens however the app ends (a normal quit, a crash, Task Manager, the updater's
//! `exit(0)` before its installer runs). Without it the agents outlived the app, kept sharing and
//! held 4111/4112/16527, so the next start couldn't run its own (11 Nov Windows test, D-H1).
//! Declared here directly (a few kernel32 calls) rather than through a new crate.

use std::os::windows::io::AsRawHandle;
use std::sync::OnceLock;

type Handle = *mut core::ffi::c_void;

#[repr(C)]
#[derive(Default)]
struct BasicLimit {
    per_process_user_time_limit: i64,
    per_job_user_time_limit: i64,
    limit_flags: u32,
    minimum_working_set_size: usize,
    maximum_working_set_size: usize,
    active_process_limit: u32,
    affinity: usize,
    priority_class: u32,
    scheduling_class: u32,
}

#[repr(C)]
#[derive(Default)]
struct IoCounters {
    read_operation_count: u64,
    write_operation_count: u64,
    other_operation_count: u64,
    read_transfer_count: u64,
    write_transfer_count: u64,
    other_transfer_count: u64,
}

/// `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`.
#[repr(C)]
#[derive(Default)]
struct ExtendedLimit {
    basic: BasicLimit,
    io: IoCounters,
    process_memory_limit: usize,
    job_memory_limit: usize,
    peak_process_memory_used: usize,
    peak_job_memory_used: usize,
}

const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: i32 = 9;
const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x2000;
/// Processes the agent starts (an engine it launches) stay outside the job, as on macOS and Linux:
/// only the agent itself ends with the app.
const JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK: u32 = 0x1000;

const PROCESS_TERMINATE: u32 = 0x0001;
const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;

#[link(name = "kernel32")]
extern "system" {
    fn CreateJobObjectW(attributes: *mut core::ffi::c_void, name: *const u16) -> Handle;
    fn SetInformationJobObject(job: Handle, class: i32, info: *const core::ffi::c_void, len: u32) -> i32;
    fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
    fn OpenProcess(access: u32, inherit: i32, pid: u32) -> Handle;
    fn QueryFullProcessImageNameW(process: Handle, flags: u32, name: *mut u16, size: *mut u32) -> i32;
    fn TerminateProcess(process: Handle, code: u32) -> i32;
    fn CloseHandle(handle: Handle) -> i32;
}

/// A path as Windows compares it: no `\\?\` prefix, `\` separators, case folded.
fn path_key(p: &str) -> String {
    p.trim_start_matches(r"\\?\").replace('/', "\\").to_lowercase()
}

/// Kill `pid` if, and only if, it is running `exe` (its full path, as Windows reports it). Whether it
/// did. The startup sweep uses it for pids an earlier run recorded: a pid Windows has since given to
/// another program, or an agent the user started from anywhere else, is left alone.
pub fn kill_if_running(pid: u32, exe: &std::path::Path) -> bool {
    use std::os::windows::ffi::OsStringExt;
    unsafe {
        let h = OpenProcess(PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return false;
        }
        let mut buf = [0u16; 1024];
        let mut len = buf.len() as u32;
        let same = QueryFullProcessImageNameW(h, 0, buf.as_mut_ptr(), &mut len) != 0
            && path_key(&std::ffi::OsString::from_wide(&buf[..len as usize]).to_string_lossy())
                == path_key(&exe.to_string_lossy());
        let killed = same && TerminateProcess(h, 1) != 0;
        CloseHandle(h);
        killed
    }
}

/// The app's job (created once, never closed: the handle closes when the app's process ends).
fn job() -> Option<usize> {
    static JOB: OnceLock<Option<usize>> = OnceLock::new();
    *JOB.get_or_init(|| unsafe {
        let h = CreateJobObjectW(core::ptr::null_mut(), core::ptr::null());
        if h.is_null() {
            return None;
        }
        let mut info = ExtendedLimit::default();
        info.basic.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK;
        let ok = SetInformationJobObject(
            h,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            &info as *const ExtendedLimit as *const core::ffi::c_void,
            std::mem::size_of::<ExtendedLimit>() as u32,
        );
        (ok != 0).then_some(h as usize)
    })
}

/// Put `child` in the app's job, so it ends with the app. Best effort: a failure leaves it as before
/// (the startup sweep and the installer's stop step still cover it).
pub fn adopt(child: &std::process::Child) {
    match job() {
        Some(j) => unsafe {
            if AssignProcessToJobObject(j as Handle, child.as_raw_handle() as Handle) == 0 {
                eprintln!("openhydra-desktop: couldn't put agent pid {} in the app's job", child.id());
            }
        },
        None => eprintln!("openhydra-desktop: no job object; agents may outlive the app"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn paths_compare_as_windows_does() {
        assert_eq!(super::path_key(r"\\?\C:\Users\Me\AppData\Local\OpenHydra\openhydra-agent.exe"),
                   super::path_key(r"c:/users/me/appdata/local/openhydra/OPENHYDRA-AGENT.EXE"));
        assert_ne!(super::path_key(r"C:\Tools\openhydra-agent.exe"), super::path_key(r"C:\OpenHydra\openhydra-agent.exe"));
    }

    /// A pid that isn't the given program is never killed (this test process is not the agent).
    #[test]
    fn a_pid_running_something_else_is_left_alone() {
        assert!(!super::kill_if_running(std::process::id(), std::path::Path::new(r"C:\nowhere\openhydra-agent.exe")));
        assert!(!super::kill_if_running(0, std::path::Path::new(r"C:\nowhere\openhydra-agent.exe")));
    }

    /// The struct must match the Windows layout, or SetInformationJobObject refuses it.
    #[test]
    fn the_limit_struct_has_the_windows_size() {
        assert_eq!(std::mem::size_of::<super::ExtendedLimit>(), if cfg!(target_pointer_width = "64") { 144 } else { 112 });
    }
}
