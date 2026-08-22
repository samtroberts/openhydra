; OpenHydra NSIS installer hooks (Layer 3): expose the bundled `openhydra-agent.exe` sidecar as an
; `openhydra` command on the user PATH, so the Connectors terminal snippets and `openhydra launch`
; work out of the box for installer installs. See docs/CLI_ON_PATH_PLAN_v1.md.
;
; ⚠ NOT YET VERIFIED ON A WINDOWS BUILD — authored on macOS. Validate on a real NSIS build:
;   - confirm the sidecar is at "$INSTDIR\openhydra-agent.exe",
;   - confirm ${WordAdd}/${UnWordAdd} dedup behaves (WordFunc.nsh is guard-safe to re-include),
;   - confirm a NEW shell resolves `openhydra` after install and it's gone after uninstall.

!include "LogicLib.nsh"
!include "WinMessages.nsh"
!include "WordFunc.nsh"
; WordFunc.nsh only DEFINES these; they must be instantiated before ${WordAdd}/${un.WordAdd} exist
; (installer + uninstaller are separate contexts, hence the un. variant). Without these it won't compile.
!insertmacro WordAdd
!insertmacro un.WordAdd

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
  Delete "$INSTDIR\openhydra.exe"
  ; Remove our install dir from the user PATH (${UnWordAdd} "-word" removes it if present).
  ReadRegStr $0 HKCU "Environment" "Path"
  ${un.WordAdd} "$0" ";" "-$INSTDIR" $1
  WriteRegExpandStr HKCU "Environment" "Path" "$1"
  SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
!macroend
