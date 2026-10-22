"""The call measurement: readings, phases and the table. No engine, GPU or meeting application is used."""

import subprocess

import pytest

from neuropresence.evaluation.call import (GPU_FIELDS, CpuReader, engine_sample, measure, parse_gpu, program_of,
                                           read_gpu, summarise_phase, table)


def status(fps=20.5, delay=98.0, state="running", holds=None, camera=True, voice=True):
    return {
        "session": {"state": state, "source": "enrolment", "camera_fps": 30.0,
                    "metrics": {"fps": fps, "end_to_end_ms": delay, "render_ms": 38.0, "pipeline_ms": 55.0,
                                "dropped_share": 0.3, "live_share": 1.0},
                    "holds": holds or {"identity": False, "delay": False, "consent": False}},
        "identity": {"csim": 0.92},
        "gpu": {"allocated_gb": 1.3},
        "virtual_camera": {"on": camera, "showing": "session"},
        "virtual_microphone": {"on": voice, "delay_ms": 115.0},
    }


def test_the_engine_figures_are_read_from_its_status():
    sample = engine_sample(status())
    assert sample["engine_reachable"] is True and sample["state"] == "running"
    assert (sample["fps"], sample["delay_ms"], sample["render_ms"], sample["camera_fps"]) == (20.5, 98.0, 38.0, 30.0)
    assert sample["held"] is None and sample["identity"] == 0.92 and sample["engine_gpu_gb"] == 1.3
    assert sample["virtual_camera"] == "session" and sample["voice_delay_ms"] == 115.0


def test_the_reason_for_the_still_picture_is_the_one_the_live_studio_shows_first():
    both = engine_sample(status(holds={"identity": True, "delay": True, "consent": True}))
    assert both["held"] == "consent"
    assert engine_sample(status(holds={"identity": False, "delay": True, "consent": False}))["held"] == "delay"


def test_an_engine_that_does_not_answer_is_marked_and_switches_that_are_off_say_so():
    assert engine_sample(None) == {"engine_reachable": False, "state": None}
    sample = engine_sample(status(camera=False, voice=False))
    assert sample["virtual_camera"] == "off" and sample["voice_delay_ms"] is None


def test_the_graphics_card_line_is_read_and_a_field_it_does_not_report_is_none():
    assert parse_gpu("71, 3412, 96.40, 2775, 63") == {"gpu_util_pct": 71.0, "gpu_memory_mb": 3412.0,
                                                       "gpu_power_w": 96.4, "gpu_clock_mhz": 2775.0, "gpu_temp_c": 63.0}
    assert parse_gpu("71, 3412, [N/A], 2775, 63")["gpu_power_w"] is None
    assert parse_gpu("garbage") == dict.fromkeys(GPU_FIELDS)


def test_without_nvidia_smi_the_card_is_simply_not_measured():
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    assert read_gpu(run=missing) == dict.fromkeys(GPU_FIELDS)

    def answers(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="12, 1000, 40.0, 1500, 50\n")

    assert read_gpu(run=answers)["gpu_util_pct"] == 12.0


@pytest.mark.parametrize("name, cmdline, program", [
    ("python.exe", ["python", "-m", "neuropresence.server", "--open"], "engine"),
    ("python.exe", ["python", "scripts/measure_call.py"], None),  # this script is not the engine
    ("chrome.exe", None, "browser"),
    ("msedge.exe", None, "browser"),
    ("Zoom.exe", None, "zoom"),
    ("CptHost.exe", None, "zoom"),
    ("ms-teams.exe", None, "teams"),
    ("explorer.exe", None, None),
])
def test_each_process_is_counted_under_its_program(name, cmdline, program):
    assert program_of(name, cmdline) == program


class Process:
    def __init__(self, pid, name, cmdline=(), cpu=0.0):
        self.pid, self.info, self._cmdline, self.cpu, self.reads = pid, {"name": name}, list(cmdline), cpu, 0

    def cmdline(self):
        return self._cmdline

    def cpu_percent(self, interval=None):
        self.reads += 1
        return 0.0 if self.reads == 1 else self.cpu  # the first reading only starts the clock


class FakePsutil:
    NoSuchProcess = AccessDenied = ZombieProcess = RuntimeError

    def __init__(self, processes, cores=4):
        self.processes, self.cores, self.machine = processes, cores, 37.5

    def cpu_count(self):
        return self.cores

    def cpu_percent(self, interval=None):
        return self.machine

    def process_iter(self, attrs=None):
        return list(self.processes)


def test_cpu_is_counted_per_program_as_a_share_of_the_whole_machine():
    processes = [Process(1, "python.exe", ["python", "-m", "neuropresence.server"], cpu=120.0),
                 Process(2, "chrome.exe", cpu=40.0), Process(3, "chrome.exe", cpu=20.0),
                 Process(4, "explorer.exe", cpu=99.0)]
    reader = CpuReader(FakePsutil(processes, cores=4))
    first = reader.read()  # starts each process's clock: nothing to report yet
    assert first["cpu_engine_pct"] is None and first["cpu_browser_pct"] is None
    second = reader.read()
    assert second["cpu_engine_pct"] == 30.0  # 120% of one core is 30% of four
    assert second["cpu_browser_pct"] == 15.0  # two browser processes together
    assert second["cpu_zoom_pct"] is None  # not running
    assert second["cpu_machine_pct"] == 37.5


def test_a_program_that_starts_during_a_phase_is_picked_up():
    processes = [Process(1, "python.exe", ["python", "-m", "neuropresence.server"], cpu=40.0)]
    fake = FakePsutil(processes)
    reader = CpuReader(fake)
    reader.read()
    processes.append(Process(9, "Zoom.exe", cpu=80.0))
    assert reader.read()["cpu_zoom_pct"] is None  # first sight
    assert reader.read()["cpu_zoom_pct"] == 20.0


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_a_phase_takes_one_sample_a_second():
    clock = Clock()
    statuses = iter([status(fps=20.0), status(fps=21.0), None])
    seen = []
    samples = measure(3, lambda: next(statuses), lambda: {"gpu_util_pct": 50.0}, lambda: {"cpu_engine_pct": 25.0},
                      clock, clock.sleep, on_sample=seen.append)
    assert [sample["t"] for sample in samples] == [0.0, 1.0, 2.0]
    assert [sample.get("fps") for sample in samples] == [20.0, 21.0, None]
    assert samples[2]["engine_reachable"] is False
    assert seen == samples


def sample(fps, delay, held=None, state="running", gpu=50.0):
    return {"state": state, "fps": fps, "delay_ms": delay, "held": held, "gpu_util_pct": gpu,
            "virtual_camera": "session", "render_ms": 38.0}


def test_a_phase_is_summarised_over_the_seconds_the_session_was_live():
    samples = [sample(20.0 + i * 0.1, 100.0) for i in range(18)]
    samples += [sample(12.0, 180.0, held="delay"), sample(None, None, state="starting")]
    summary = summarise_phase(samples)
    assert summary["seconds"] == 20 and summary["running_seconds"] == 19
    assert summary["fps"]["min"] == 12.0 and summary["fps"]["median"] == pytest.approx(20.8, abs=0.01)
    assert summary["delay_ms"]["max"] == 180.0
    assert summary["held_share"] == pytest.approx(1 / 19, abs=0.001) and summary["held_by"] == {"delay": 1}
    assert summary["delay_over_limit_share"] == pytest.approx(1 / 19, abs=0.001)
    assert summary["virtual_camera_session_share"] == 1.0
    assert summary["cpu_zoom_pct"] is None  # never measured


def test_a_phase_with_no_live_seconds_has_nothing_to_compare():
    summary = summarise_phase([sample(None, None, state="error")])
    assert summary["running_seconds"] == 0 and summary["fps"] is None and summary["held_share"] is None


def test_the_table_sets_the_phases_side_by_side():
    phases = {
        "baseline": {"title": "No meeting", "summary": summarise_phase([sample(22.0, 98.0, gpu=60.0)] * 5)},
        "zoom": {"title": "Zoom", "summary": summarise_phase([sample(18.0, 120.0, gpu=90.0)] * 5)},
    }
    text = table(phases)
    lines = text.splitlines()
    assert lines[0].split()[-2:] == ["meeting", "Zoom"]
    frame_rate = next(line for line in lines if line.startswith("Frame rate"))
    assert "22.0, 22.0 fps" in frame_rate and "18.0, 18.0 fps" in frame_rate
    assert "98 ms" in next(line for line in lines if line.startswith("Delay")) and "90%" in text
    assert next(line for line in lines if line.startswith("CPU, Zoom")).split()[-2:] == ["-", "-"]
    assert "5 of 5" in text
