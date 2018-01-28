import sys
import os
import traceback
import ctypes
from ctypes import wintypes
import win32con
import win32api
import win32gui
import win32process

hwnds = []


def enum_windows_proc(hwnd, l_param):
    if (l_param is None) or ((l_param is not None) and (win32process.GetWindowThreadProcessId(hwnd)[1] == l_param)):
        text = win32gui.GetWindowText(hwnd)
        if text:
            w_style = win32api.GetWindowLong(hwnd, win32con.GWL_STYLE)
            if w_style & win32con.WS_VISIBLE:
                hwnds.append(hwnd)


def enum_proc_wnds(pid=None):
    win32gui.EnumWindows(enum_windows_proc, pid)


def enum_procs(proc_name=None):
    pids = win32process.EnumProcesses()
    if proc_name is not None:
        buf_len = 0x100
        bytes = wintypes.DWORD(buf_len)
        _OpenProcess = ctypes.cdll.kernel32.OpenProcess
        _GetProcessImageFileName = ctypes.cdll.psapi.GetProcessImageFileNameA
        _CloseHandle = ctypes.cdll.kernel32.CloseHandle
        filtered_pids = ()
        for pid in pids:
            try:
                h_proc = _OpenProcess(wintypes.DWORD(win32con.PROCESS_ALL_ACCESS), ctypes.c_int(0), wintypes.DWORD(pid))
            except:
                print("Process [%d] couldn't be opened: %s" % (pid, traceback.format_exc()))
                continue
            try:
                buf = ctypes.create_string_buffer(buf_len)
                _GetProcessImageFileName(h_proc, ctypes.pointer(buf), ctypes.pointer(bytes))
                if buf.value:
                    name = buf.value.decode().split(os.path.sep)[-1]
                    # print name
                else:
                    _CloseHandle(h_proc)
                    continue
            except:
                print("Error getting process name: %s" % traceback.format_exc())
                _CloseHandle(h_proc)
                continue
            if name.lower() == proc_name.lower():
                filtered_pids += (pid,)
        return filtered_pids
    else:
        return pids


def eval_hwnds(hwnds, name):  # Will create list of all apps with given name
    valid_hwnds = []
    for hwnd in hwnds:
        text = win32gui.GetWindowText(hwnd)
        if text == name:
            valid_hwnds.append(hwnd)

    return valid_hwnds


def main(args):
    if args:
        proc_name = args[0]
    else:
        proc_name = None
    pids = enum_procs(proc_name)
    # print(pids)
    for pid in pids:
        enum_proc_wnds(pid)
        # print(pid)

if __name__ == "__main__":
    main(sys.argv[1:])
