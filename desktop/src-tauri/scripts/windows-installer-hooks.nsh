; OpenHydra NSIS installer hooks (Layer 3): expose the bundled `openhydra-agent.exe` sidecar as an
; `openhydra` command on the user PATH, so the Connectors terminal snippets and `openhydra launch`
; work out of the box for installer installs. See docs/CLI_ON_PATH_PLAN_v1.md.
;
; Verified on a real NSIS build (Windows Server 2022, 11 Nov 2026): the sidecar is at
; "$INSTDIR\openhydra-agent.exe" and the install leaves "$INSTDIR\openhydra.exe" beside it.
; Not yet checked: a NEW shell resolving `openhydra`, and its removal on uninstall.

!include "LogicLib.nsh"
!include "WinMessages.nsh"
!include "WordFunc.nsh"
; WordFunc.nsh only DEFINES these; they must be instantiated before ${WordAdd}/${un.WordAdd} exist
; (installer + uninstaller are separate contexts, hence the un. variant). Without these it won't compile.
!insertmacro WordAdd
!insertmacro un.WordAdd

; Stop OpenHydra's own agents (those running from this install folder) before files are replaced
; or removed: an agent left running by a crash or an update locks its .exe, and holds the app's
; ports so the new version can't start its own (11 Nov Windows test, D-H1). Agents the user runs
; from anywhere else are left alone.
;
; - The folder goes to PowerShell in an environment variable, never inside its quotes, so a path
;   with an apostrophe (C:\Users\O'Brien\...) can't break the command; a trailing \ keeps
;   ...\OpenHydraDev\ from matching ...\OpenHydra.
; - Processes are found through CIM (Win32_Process.ExecutablePath): the installer is 32-bit, and a
;   32-bit PowerShell's Get-Process can't read a 64-bit process's path (review 11 Nov).
; - -InputFormat None: PowerShell under nsExec can otherwise wait on stdin.
; Backtick-quoted so ' and " pass through unescaped ($$ is a literal $).
!macro OH_STOP_AGENTS
  System::Call 'kernel32::SetEnvironmentVariable(t "OH_INSTDIR", t "$INSTDIR")'
  nsExec::ExecToLog `powershell -NoProfile -NonInteractive -InputFormat None -ExecutionPolicy Bypass -Command "$$d = $$env:OH_INSTDIR.TrimEnd([char]92) + [char]92; Get-CimInstance Win32_Process | Where-Object { ($$_.Name -eq 'openhydra-agent.exe' -or $$_.Name -eq 'openhydra.exe') -and $$_.ExecutablePath -and $$_.ExecutablePath.StartsWith($$d, [StringComparison]::OrdinalIgnoreCase) } | ForEach-Object { Stop-Process -Id $$_.ProcessId -Force -ErrorAction SilentlyContinue }"`
  Pop $0
!macroend

!macro NSIS_HOOK_PREINSTALL
  !insertmacro OH_STOP_AGENTS
!macroend

!macro NSIS_HOOK_POSTINSTALL
  ; 1) The CLI is subcommand-dispatched, so a copy named openhydra.exe IS the full CLI.
  CopyFiles /SILENT "$INSTDIR\openhydra-agent.exe" "$INSTDIR\openhydra.exe"
  ; 2) Put the install dir on the USER PATH (no admin). ${WordAdd} "+word" appends only if absent,
  ;    so re-running on update never duplicates the entry.
  ReadRegStr $0 HKCU "Environment" "Path"
  ${WordAdd} "$0" ";" "+$INSTDIR" $1
  WriteRegExpandStr HKCU "Environment" "Path" "$1"
  ; Broadcast so already-open shells / Explorer pick up the new PATH.
  SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  !insertmacro OH_STOP_AGENTS
  Delete "$INSTDIR\openhydra.exe"
  ; Remove our install dir from the user PATH (${UnWordAdd} "-word" removes it if present).
  ReadRegStr $0 HKCU "Environment" "Path"
  ${un.WordAdd} "$0" ";" "-$INSTDIR" $1
  WriteRegExpandStr HKCU "Environment" "Path" "$1"
  SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
!macroend
