"""Measuring the system during a meeting call.

The committee asked what the system does when a meeting application runs beside it on the same
computer. A meeting application encodes video, often on the same graphics card, and takes CPU time
for its own camera pipeline and network. This module records, once a second while a camera session
is live, three things:

- the engine's own figures, read from its status over HTTP without changing it: frame rate, delay
  from camera to output, render time, dropped frames, and whether the still picture was up and why;
- the whole graphics card, from nvidia-smi: how busy it is, its memory, power and clock, which
  includes what the meeting application takes;
- the CPU time of each program that matters: the engine, the browser that runs Google Meet, Zoom.

A phase is a run of these samples under one condition (no meeting, in Meet, in Zoom), and its
summary is what the report compares. Only the seconds in which the session was running count.
"""

import subprocess

import numpy as np

from ..targets import TARGETS

STATUS_URL = "http://127.0.0.1:8000/api/status"
GPU_FIELDS = ("gpu_util_pct", "gpu_memory_mb", "gpu_power_w", "gpu_clock_mhz", "gpu_temp_c")
GPU_QUERY = "utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu"

# Programs whose CPU time is counted, by the name of their executable in lower case, without ".exe".
# The engine is found by its command line instead, since it runs as python.
PROGRAMS = {
    "browser": ("chrome", "msedge", "firefox", "brave", "opera"),
    "zoom": ("zoom", "zoomwebviewhost", "cpthost", "aomhost64"),
    "teams": ("ms-teams", "teams"),
}
ENGINE_COMMAND = "neuropresence.server"
PYTHONS = ("python", "pythonw", "python3")

# The figures summarised for each phase. For a frame rate the low end matters, for a delay the high end,
# so every figure gets both tails.
FIGURES = (
    "fps", "delay_ms", "render_ms", "pipeline_ms", "dropped_share", "camera_fps", "identity", "engine_gpu_gb",
    "voice_delay_ms", *GPU_FIELDS, "cpu_machine_pct", "cpu_engine_pct", "cpu_browser_pct", "cpu_zoom_pct",
    "cpu_teams_pct",
)


# ------------------------------------------------------------------ readings


def engine_sample(status):
    """The engine's figures from its /api/status reply, or a mark that it could not be reached."""
    if status is None:
        return {"engine_reachable": False, "state": None}
    session = status.get("session") or {}
    metrics = session.get("metrics") or {}
    holds = session.get("holds") or {}
    camera = status.get("virtual_camera") or {}
    voice = status.get("virtual_microphone") or {}
    return {
        "engine_reachable": True,
        "state": session.get("state"),
        "source": session.get("source"),
        "fps": metrics.get("fps"),
        "delay_ms": metrics.get("end_to_end_ms"),
        "render_ms": metrics.get("render_ms"),
        "pipeline_ms": metrics.get("pipeline_ms"),
        "dropped_share": metrics.get("dropped_share"),
        "camera_fps": session.get("camera_fps"),
        # Why the still picture is up, the first reason that holds, as the Live Studio orders them.
        "held": next((reason for reason in ("consent", "identity", "delay") if holds.get(reason)), None),
        "identity": (status.get("identity") or {}).get("csim"),
        "engine_gpu_gb": (status.get("gpu") or {}).get("allocated_gb"),
        "virtual_camera": (camera.get("showing") or "off") if camera.get("on") else "off",
        "voice_delay_ms": voice.get("delay_ms") if voice.get("on") else None,
    }


def parse_gpu(line):
    """One line of nvidia-smi's CSV answer to GPU_QUERY, as numbers; a field it cannot read is None."""
    values = [part.strip() for part in line.strip().split(",")]
    if len(values) != len(GPU_FIELDS):
        return dict.fromkeys(GPU_FIELDS)
    out = {}
    for key, value in zip(GPU_FIELDS, values):
        try:
            out[key] = float(value)
        except ValueError:  # "[N/A]" on cards that do not report it
            out[key] = None
    return out


def read_gpu(run=subprocess.run):
    """The first graphics card, now. All None when nvidia-smi is missing or fails."""
    try:
        done = run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return dict.fromkeys(GPU_FIELDS)
    return parse_gpu(done.stdout.splitlines()[0] if done.stdout.strip() else "")


def gpu_name(run=subprocess.run):
    try:
        return run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                   timeout=5, check=True).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def program_of(name, cmdline=None):
    """Which counted program a process belongs to: "engine", "browser", "zoom", "teams" or None."""
    base = (name or "").lower()
    if base.endswith(".exe"):
        base = base[:-4]
    if base in PYTHONS:
        return "engine" if cmdline and any(ENGINE_COMMAND in part for part in cmdline) else None
    for program, names in PROGRAMS.items():
        if base in names:
            return program
    return None


class CpuReader:
    """CPU time of the counted programs, as a share of the whole machine (100 = every core busy).

    A process's first reading only starts its clock, so a program shows from its second sample on.
    """

    def __init__(self, psutil_module=None):
        if psutil_module is None:
            import psutil as psutil_module
        self._ps = psutil_module
        self._cores = psutil_module.cpu_count() or 1
        self._watched = {}  # pid -> (process, program)
        self._skipped = set()  # pids that are none of ours
        psutil_module.cpu_percent(None)

    def _program(self, process):
        name = process.info.get("name") or ""
        base = name.lower().removesuffix(".exe")
        cmdline = None
        if base in PYTHONS:  # only a python process needs its command line read, which is slow
            try:
                cmdline = process.cmdline()
            except (self._ps.NoSuchProcess, self._ps.AccessDenied, self._ps.ZombieProcess):
                return None
        return program_of(name, cmdline)

    def read(self):
        totals, seen = {program: 0.0 for program in ("engine", *PROGRAMS)}, set()
        alive = set()
        for process in self._ps.process_iter(["name"]):
            pid = process.pid
            alive.add(pid)
            if pid in self._skipped:
                continue
            if pid not in self._watched:
                program = self._program(process)
                if program is None:
                    self._skipped.add(pid)
                    continue
                self._watched[pid] = (process, program)
                try:
                    process.cpu_percent(None)  # starts its clock
                except (self._ps.NoSuchProcess, self._ps.AccessDenied):
                    pass
                continue
            watched, program = self._watched[pid]
            try:
                totals[program] += watched.cpu_percent(None) / self._cores
                seen.add(program)
            except (self._ps.NoSuchProcess, self._ps.AccessDenied):
                pass
        for pid in set(self._watched) - alive:  # a pid can be used again by a new process
            del self._watched[pid]
        self._skipped &= alive
        out = {"cpu_machine_pct": round(float(self._ps.cpu_percent(None)), 1)}
        out.update({f"cpu_{program}_pct": round(value, 1) if program in seen else None
                    for program, value in totals.items()})
        return out


# ------------------------------------------------------------------- a phase


def measure(seconds, read_status, read_gpu_now, read_cpu, clock, sleep, on_sample=None):
    """Take one sample a second for `seconds` seconds. Returns the samples, each with its time `t` from the start."""
    samples = []
    start = clock()
    for index in range(int(seconds)):
        due = start + index
        wait = due - clock()
        if wait > 0:
            sleep(wait)
        sample = {"t": round(clock() - start, 2)}
        sample.update(engine_sample(read_status()))
        sample.update(read_gpu_now())
        sample.update(read_cpu())
        samples.append(sample)
        if on_sample is not None:
            on_sample(sample)
    return samples


def _spread(values):
    values = np.asarray(values, dtype=float)
    return {"median": round(float(np.median(values)), 2), "p5": round(float(np.percentile(values, 5)), 2),
            "p95": round(float(np.percentile(values, 95)), 2), "min": round(float(values.min()), 2),
            "max": round(float(values.max()), 2)}


def summarise_phase(samples):
    """What a phase showed, over the seconds in which the session was running."""
    running = [sample for sample in samples if sample.get("state") == "running"]
    summary = {"seconds": len(samples), "running_seconds": len(running)}
    for key in FIGURES:
        values = [sample[key] for sample in running if sample.get(key) is not None]
        summary[key] = _spread(values) if values else None
    held = [sample["held"] for sample in running if sample.get("held")]
    summary["held_share"] = round(len(held) / len(running), 3) if running else None
    summary["held_by"] = {reason: held.count(reason) for reason in sorted(set(held))}
    delays = [sample["delay_ms"] for sample in running if sample.get("delay_ms") is not None]
    limit = TARGETS["end_to_end_ms"]
    summary["delay_over_limit_share"] = round(sum(d > limit for d in delays) / len(delays), 3) if delays else None
    shown = [sample.get("virtual_camera") for sample in running]
    summary["virtual_camera_session_share"] = round(shown.count("session") / len(running), 3) if running else None
    return summary


# ------------------------------------------------------------------ the table

ROWS = (
    ("Frame rate (median, low 5%)", "fps", ("median", "p5"), "{:.1f}", " fps"),
    ("Delay camera to output (median, high 5%)", "delay_ms", ("median", "p95"), "{:.0f}", " ms"),
    ("Render time (median)", "render_ms", ("median",), "{:.1f}", " ms"),
    ("Camera frames dropped (median)", "dropped_share", ("median",), "{:.0%}", ""),
    ("Graphics card busy (median)", "gpu_util_pct", ("median",), "{:.0f}", "%"),
    ("Graphics memory in use (median)", "gpu_memory_mb", ("median",), "{:.0f}", " MB"),
    ("Graphics card power (median)", "gpu_power_w", ("median",), "{:.0f}", " W"),
    ("CPU, engine (median)", "cpu_engine_pct", ("median",), "{:.0f}", "%"),
    ("CPU, browser (median)", "cpu_browser_pct", ("median",), "{:.0f}", "%"),
    ("CPU, Zoom (median)", "cpu_zoom_pct", ("median",), "{:.0f}", "%"),
    ("CPU, whole machine (median)", "cpu_machine_pct", ("median",), "{:.0f}", "%"),
)


def _cell(summary, key, stats, form, unit):
    spread = summary.get(key)
    if not spread:
        return "-"
    parts = [form.format(spread[stat]) for stat in stats]
    return ", ".join(parts) + unit


def table(phases):
    """A plain text table of the phases side by side, in the order given: {key: {"title", "summary"}}."""
    titles = [phase["title"] for phase in phases.values()]
    rows = [[label] + [_cell(phase["summary"], key, stats, form, unit) for phase in phases.values()]
            for label, key, stats, form, unit in ROWS]
    rows.append(["Still picture shown"] + [
        "-" if phase["summary"].get("held_share") is None else f"{phase['summary']['held_share']:.0%}"
        for phase in phases.values()])
    rows.append(["Seconds measured (session live)"] + [
        f"{phase['summary']['running_seconds']} of {phase['summary']['seconds']}" for phase in phases.values()])
    header = ["", *titles]
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    line = lambda row: "  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip()
    return "\n".join([line(header), line(["-" * width for width in widths]), *(line(row) for row in rows)])
