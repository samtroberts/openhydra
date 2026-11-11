//! Windows: the app's agents live in a Job Object that kills them when the app's last handle to it
//! closes, which happens however the app ends (a normal quit, a crash, Task Manager, the updater's
//! `exit(0)` before its installer runs). Without it the agents outlived the app, kept sharing and
//! held 4111/4112/16527, so the next start couldn't run its own (11 Nov Windows test, D-H1).
//! Declared here directly (three kernel32 calls) rather than through a new crate.

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

#[link(name = "kernel32")]
extern "system" {
    fn CreateJobObjectW(attributes: *mut core::ffi::c_void, name: *const u16) -> Handle;
    fn SetInformationJobObject(job: Handle, class: i32, info: *const core::ffi::c_void, len: u32) -> i32;
    fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
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
    /// The struct must match the Windows layout, or SetInformationJobObject refuses it.
    #[test]
    fn the_limit_struct_has_the_windows_size() {
        assert_eq!(std::mem::size_of::<super::ExtendedLimit>(), if cfg!(target_pointer_width = "64") { 144 } else { 112 });
    }
}
