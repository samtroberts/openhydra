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
!macro OH_STOP_AGENTS
  ; Backtick-quoted so both ' and " pass through to PowerShell unescaped ($$ is a literal $).
  nsExec::ExecToLog `powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "Get-Process openhydra-agent,openhydra -ErrorAction SilentlyContinue | Where-Object { $$_.Path -and $$_.Path.StartsWith('$INSTDIR', [StringComparison]::OrdinalIgnoreCase) } | Stop-Process -Force"`
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
