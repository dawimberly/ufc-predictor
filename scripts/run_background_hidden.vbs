' Hidden launcher for scheduled UFC background runs (no console flash).
' Usage: wscript.exe run_background_hidden.vbs [mode] [trigger]
'   mode:    auto | full | lightweight  (default: auto)
'   trigger: startup | scheduled | midnight | manual  (default: manual)
Option Explicit
Dim sh, fso, root, mode, trigger, bat, cmd
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
mode = "auto"
trigger = "manual"
If WScript.Arguments.Count >= 1 Then mode = WScript.Arguments(0)
If WScript.Arguments.Count >= 2 Then trigger = WScript.Arguments(1)
bat = root & "\scripts\run_background.bat"
If Not fso.FileExists(bat) Then
    WScript.Quit 1
End If
cmd = "cmd /c """ & bat & """ " & mode & " " & trigger
' 0 = hidden window; True = wait so Task Scheduler records the real exit code
sh.Run cmd, 0, True
