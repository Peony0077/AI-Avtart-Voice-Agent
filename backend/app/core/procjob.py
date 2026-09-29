"""Child-process hygiene on Windows.

* ``JOB``: a Windows Job Object with KILL_ON_JOB_CLOSE. Every helper process (llama-server,
  avatar worker, XTTS worker) is assigned to it, so they die when the backend dies — even when
  the backend is killed hard. (Before this, a force-stopped backend left an orphaned avatar
  worker holding the port and ~3 GB VRAM; the next backend silently connected to the stale one.)
* ``free_port``: kill a stale process of ours that still listens on a port before we spawn.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys

import psutil
from loguru import logger


class _Job:
    def __init__(self) -> None:
        self.handle = None
        if sys.platform != "win32":
            return
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.OpenProcess.restype = wintypes.HANDLE
        self._k32 = k32
        handle = k32.CreateJobObjectW(None, None)
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):  # Extended info
            logger.warning(f"job object setup failed: {ctypes.get_last_error()}")
            return
        self.handle = handle

    def adopt(self, proc: subprocess.Popen) -> None:
        if self.handle is None:
            return
        h = self._k32.OpenProcess(0x0001 | 0x0100, False, proc.pid)  # PROCESS_TERMINATE | PROCESS_SET_QUOTA
        if not h or not self._k32.AssignProcessToJobObject(self.handle, h):
            logger.warning(f"could not add pid {proc.pid} to the job object ({ctypes.get_last_error()})")
        if h:
            self._k32.CloseHandle(h)


JOB = _Job()


def free_port(port: int, must_contain: str) -> None:
    """Kill the process tree listening on ``port`` if its command line contains ``must_contain``."""
    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN and conn.pid:
            try:
                proc = psutil.Process(conn.pid)
                cmd = " ".join(proc.cmdline())
            except psutil.Error:
                continue
            if must_contain not in cmd or conn.pid == os.getpid():
                logger.error(f"port {port} is used by an unrelated process {conn.pid}: {cmd[:120]}")
                continue
            victims = [proc, *proc.children(recursive=True)]
            try:
                victims += [proc.parent()] if proc.parent() and must_contain in " ".join(proc.parent().cmdline()) else []
            except psutil.Error:
                pass
            logger.warning(f"killing stale process on port {port}: pid {conn.pid} ({cmd[:100]})")
            for v in victims:
                try:
                    v.kill()
                except psutil.Error:
                    pass
            psutil.wait_procs(victims, timeout=5)
