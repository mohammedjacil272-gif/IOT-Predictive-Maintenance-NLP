use std::net::TcpListener;
use std::sync::Mutex;

use tauri::{Emitter, Manager, RunEvent, State};
use tauri_plugin_shell::{process::CommandEvent, ShellExt};
use uuid::Uuid;

struct SidecarProcess(Mutex<Option<tauri_plugin_shell::process::CommandChild>>);

#[cfg(windows)]
struct SidecarJob(*mut core::ffi::c_void);

#[cfg(windows)]
unsafe impl Send for SidecarJob {}

#[cfg(windows)]
unsafe impl Sync for SidecarJob {}

#[cfg(windows)]
impl Drop for SidecarJob {
    fn drop(&mut self) {
        unsafe { windows_sys::Win32::Foundation::CloseHandle(self.0) };
    }
}

#[cfg(windows)]
fn assign_kill_on_parent_exit(pid: u32) -> Result<SidecarJob, String> {
    use std::mem::size_of;
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
    };

    let process = unsafe {
        OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SET_QUOTA | PROCESS_TERMINATE,
            0,
            pid,
        )
    };
    if process.is_null() {
        return Err(format!(
            "sidecar process handle unavailable: {}",
            std::io::Error::last_os_error()
        ));
    }
    let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
    if job.is_null() {
        unsafe { CloseHandle(process) };
        return Err(format!(
            "job object creation failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    let configured = unsafe {
        SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            (&mut limits as *mut JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
    } != 0;
    let assigned = configured && unsafe { AssignProcessToJobObject(job, process) } != 0;
    unsafe { CloseHandle(process) };
    if !assigned {
        unsafe { CloseHandle(job) };
        return Err(format!(
            "sidecar job assignment failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(SidecarJob(job))
}

#[derive(Clone, serde::Serialize)]
struct BackendConfig {
    port: u16,
    token: String,
}

#[tauri::command]
fn backend_config(config: State<'_, BackendConfig>) -> BackendConfig {
    config.inner().clone()
}

fn free_loopback_port() -> Result<u16, Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let data_root = match std::env::var_os("IOT_APP_DATA_DIR_OVERRIDE") {
                Some(path) => std::path::PathBuf::from(path),
                None => app
                    .path()
                    .app_data_dir()
                    .map_err(|error| format!("application data directory unavailable: {error}"))?,
            };
            std::fs::create_dir_all(&data_root)
                .map_err(|error| format!("application data directory creation failed: {error}"))?;
            let port =
                free_loopback_port().map_err(|error| format!("port allocation failed: {error}"))?;
            let token = Uuid::new_v4().to_string();
            let (mut events, child) = app
                .shell()
                .sidecar("iot-backend")
                .map_err(|error| format!("sidecar configuration failed: {error}"))?
                .args(["--host", "127.0.0.1", "--port", &port.to_string()])
                .env(
                    "IOT_INTEGRATED_ROOT",
                    data_root.to_string_lossy().to_string(),
                )
                .env("IOT_API_TOKEN", &token)
                // The packaged sidecar does not inherit a project .env file.
                // Keep the local assistant explicitly configured for the
                // already-installed Ollama model without downloading weights.
                .env("IOT_ASSISTANT_MODEL_PROVIDER", "ollama")
                .env("IOT_ASSISTANT_MODEL", "qwen2.5:3b")
                .env("IOT_ASSISTANT_MODEL_BASE_URL", "http://127.0.0.1:11434")
                .env("IOT_ASSISTANT_MODEL_TIMEOUT", "60")
                .env("IOT_ASSISTANT_MODEL_RETRIES", "1")
                // The shell plugin implements this as a Windows Job Object. It
                // starts the process suspended, assigns it before resuming, and
                // configures kill-on-job-close, so a crash/forced parent exit
                // cannot leave this sidecar (or its descendants) behind.
                .set_process_group(true)
                .kill_on_drop(true)
                .spawn()
                .map_err(|error| format!("sidecar launch failed: {error}"))?;

            let sidecar_pid = child.pid();
            app.manage(SidecarProcess(Mutex::new(Some(child))));
            #[cfg(windows)]
            app.manage(assign_kill_on_parent_exit(sidecar_pid)?);
            app.manage(BackendConfig {
                port,
                token: token.clone(),
            });
            app.emit(
                "backend-starting",
                serde_json::json!({ "port": port, "token": token }),
            )?;
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                while let Some(event) = events.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            let _ = handle.emit("backend-log", line);
                        }
                        CommandEvent::Stderr(line) => {
                            let _ = handle.emit("backend-error", line);
                        }
                        CommandEvent::Terminated(payload) => {
                            let _ = handle.emit("backend-exited", payload.code);
                            break;
                        }
                        CommandEvent::Error(message) => {
                            let _ = handle.emit("backend-error", message);
                        }
                        _ => {}
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![backend_config])
        .build(tauri::generate_context!())
        .expect("error while building desktop application")
        .run(|app, event| {
            if let RunEvent::ExitRequested { .. } = event {
                if let Some(state) = app.try_state::<SidecarProcess>() {
                    if let Ok(mut child) = state.0.lock() {
                        if let Some(process) = child.take() {
                            let _ = process.kill();
                        }
                    }
                }
            }
        });
}
