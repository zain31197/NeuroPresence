"""The virtual microphone: the user's voice, held back so that it stays in step with the picture.

The picture a meeting sees is late by the time the system takes to make it, about 90 to 110 ms.
The voice, sent straight from the microphone, is not. Lips that move after the words are heard
look wrong, so the voice is delayed by the same amount and sent on to a virtual cable. The
meeting application selects the other end of that cable as its microphone.

On Windows the cable is VB-Audio Virtual Cable: what is played into "CABLE Input" comes out of
"CABLE Output". On Linux a PipeWire or PulseAudio null sink does the same, with nothing to install.

How long to hold the voice back is taken from the running session: the delay from camera to
output, as the Live Studio shows it. The delay does not jump with every frame. It follows that
figure only when it has moved by more than a few milliseconds, and it gets there by playing the
voice one per cent faster or slower, which no one hears. Only a large change, when a session
starts or stops, is made at once, with a short fade. Outside a running camera session the
meeting sees the plain card, so the voice is passed on with no added delay.

The meeting application adds delays of its own that cannot be measured from here, to the
picture and to the sound. They are what the adjustment in the Live Studio is for.

Nothing is recorded. The voice is held in memory for the length of the delay and then sent on;
the buffer holds two seconds and is written over all the time.

Measured on 15 October 2026 (Windows 11, VB-Audio Virtual Cable, WASAPI at 48 kHz), with the
cable's far end fed back in as the microphone: a click went round in the set delay plus 100 to
112 ms, the same on every trip. Those 100 to 112 ms are the cable and the sound buffers.
"""

import json
import sys
import threading
import time

import numpy as np

RATE = 48000  # samples a second: what Windows runs both devices at
BLOCK = 480  # 10 ms handed over at a time
BUFFER_SECONDS = 2.0  # longer than any delay that is worth keeping
# The least the voice is held. The two devices hand sound over unevenly, the reader's lead swinging by
# up to 20 ms either way (measured on the cable), and the reader must never overtake the writer.
MIN_DELAY_MS = 40.0
MAX_DELAY_MS = 600.0
FOLLOW_MS = 8.0  # the picture delay has to move by this much before the voice follows it
NUDGE = 0.01  # how much faster or slower the voice is played while it catches up: one per cent
SMOOTHING = 0.02  # each read moves the judged lead this share of the way: about half a second
# Catching up starts this far from the delay, and goes on until it is this close, so that it does not
# start and stop with every read. Both are far inside what anyone notices: sound ahead of the picture
# by 45 ms, or behind it by 125 ms (ITU-R BT.1359).
CATCH_UP_MS = 8.0
CAUGHT_UP_MS = 2.0
# A device that has just opened asks for several blocks at once. The lead is averaged over the reads
# after that and set right once, so the delay is right from the start, not two seconds later.
PRIMING_READS = 10
SETTLE_READS = 50
JUMP_MS = 40.0  # a change larger than this is made at once
FADE = 120  # samples, 2.5 ms: the crossfade that keeps a jump from clicking
UPDATE_SECONDS = 0.25  # how often the delay is checked against the picture
DEVICES_SECONDS = 5.0  # how long the list of devices is kept before it is asked for again
OFFSET_LIMIT_MS = 200.0  # the adjustment in the Live Studio goes this far either way

CABLE_NAMES = ("CABLE Input", "NeuroPresence")  # VB-Audio on Windows; a null sink of our name on Linux
INSTALL = {
    "win32": "Install VB-Audio Virtual Cable (vb-audio.com/Cable), restart the computer, and start the engine again.",
    "linux": "Create the cable with: pactl load-module module-null-sink sink_name=NeuroPresence "
             "sink_properties=device.description=NeuroPresence, then start the engine again.",
}


class DelayLine:
    """Holds a stream of samples back by a set delay, and changes that delay without a gap or a click.

    `write` is called with what the microphone delivers and `read` with what the cable asks for.
    The two devices keep their own clocks and drift apart slowly; the same catching up that follows
    a new delay keeps that drift from adding up. They also hand sound over unevenly, so the lead is
    judged over half a second, not read by read: otherwise the voice would be sped up and slowed
    down all the time.
    """

    def __init__(self, rate=RATE, seconds=BUFFER_SECONDS, delay_ms=MIN_DELAY_MS):
        self.rate = rate
        self._ring = np.zeros(int(rate * seconds), np.float32)
        self._lock = threading.Lock()
        self.written = 0  # samples written since the start
        self.position = None  # where the reader is, in samples since the start; None until the first read
        self.target = self._samples(delay_ms)
        self.behind = None  # how far the reader is behind the writer, judged over the last reads
        self._catching = 0  # 1 while playing faster to shorten the delay, -1 while playing slower
        self.reads = 0
        self._settling = []  # the leads of the first reads, until the delay is set right once
        self.jumps = 0

    def _samples(self, delay_ms):
        delay_ms = min(max(delay_ms, MIN_DELAY_MS), MAX_DELAY_MS)
        return delay_ms * self.rate / 1000.0

    def set_delay(self, delay_ms):
        with self._lock:
            self.target = self._samples(delay_ms)

    @property
    def delay_ms(self):
        """How far behind the microphone the voice is, as it is judged when a read begins."""
        with self._lock:
            behind = self.target if self.behind is None else self.behind
            return behind * 1000.0 / self.rate

    def write(self, samples):
        samples = np.asarray(samples, np.float32).reshape(-1)
        with self._lock:
            size = len(self._ring)
            if len(samples) > size:
                samples = samples[-size:]
            start = self.written % size
            first = min(len(samples), size - start)
            self._ring[start:start + first] = samples[:first]
            self._ring[:len(samples) - first] = samples[first:]
            self.written += len(samples)

    def read(self, frames):
        with self._lock:
            if self.position is None:
                self.position = self.written - self.target
            lead = self.written - self.position
            self.reads += 1
            if self.reads <= SETTLE_READS:
                return self._settle(frames, lead)
            self.behind = self.behind + SMOOTHING * (lead - self.behind)
            error = self.behind - self.target  # above 0: the voice is held too long
            if abs(error) > JUMP_MS * self.rate / 1000.0:
                return self._jump(frames, error)
            per_ms = self.rate / 1000.0
            if self._catching == 0 and abs(error) > CATCH_UP_MS * per_ms:
                self._catching = 1 if error > 0 else -1
            elif self._catching and error * self._catching < CAUGHT_UP_MS * per_ms:
                self._catching = 0
            step = 1.0 + NUDGE * self._catching
            out = self._take(self.position, frames, step)
            self.position += frames * step
            return out

    def _settle(self, frames, lead):
        """The first reads: played as they come, then the delay set right from their average lead."""
        if self.reads > PRIMING_READS:
            self._settling.append(lead)
        if self.reads < SETTLE_READS:
            self.behind = lead
            out = self._take(self.position, frames, 1.0)
            self.position += frames
            return out
        self.behind = float(np.mean(self._settling))
        error = self.behind - self.target
        if abs(error) > CAUGHT_UP_MS * self.rate / 1000.0:
            return self._jump(frames, error)
        out = self._take(self.position, frames, 1.0)
        self.position += frames
        return out

    def _jump(self, frames, error):
        """Go to the new delay at once: the old stream fades out as the new one fades in."""
        self.jumps += 1
        self._catching = 0
        fade = min(FADE, frames)
        old = self._take(self.position, fade, 1.0)
        self.position += error
        self.behind -= error
        new = self._take(self.position, frames, 1.0)
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        new[:fade] = new[:fade] * ramp + old * (1.0 - ramp)
        self.position += frames
        return new

    def _take(self, position, frames, step):
        """`frames` samples from `position` on, `step` apart, read between samples where they fall.
        Anything not yet written, or already written over, is silence."""
        at = position + step * np.arange(frames)
        low = np.floor(at).astype(np.int64)
        share = (at - low).astype(np.float32)
        size = len(self._ring)
        valid = (low >= max(0, self.written - size)) & (low + 1 < self.written)
        a = np.where(valid, self._ring[low % size], 0.0)
        b = np.where(valid, self._ring[(low + 1) % size], 0.0)
        return (a * (1.0 - share) + b * share).astype(np.float32)


# ---------------------------------------------------------------- devices


def _host_api(sd):
    """WASAPI on Windows, for its short buffers; the default elsewhere."""
    for index, api in enumerate(sd.query_hostapis()):
        if sys.platform == "win32" and api["name"] == "Windows WASAPI":
            return index
    return sd.default.hostapi


def meeting_name(cable):
    """What the meeting application lists the other end of the cable under, as its microphone."""
    if cable is None:
        return None
    if cable.startswith("CABLE Input"):
        return cable.replace("CABLE Input", "CABLE Output", 1)
    return f"Monitor of {cable}"  # a null sink on Linux


def _is_cable(name):
    return any(part.lower() in name.lower() for part in ("CABLE", "VB-Audio", "NeuroPresence"))


def list_devices():
    """The microphones the user can choose, the cable if there is one, and the system's own microphone.

    Returns {"microphones": [names], "default": name or None, "cable": name or None}.
    """
    import sounddevice as sd

    api = _host_api(sd)
    devices = [d for d in sd.query_devices() if d["hostapi"] == api]
    microphones = [d["name"] for d in devices if d["max_input_channels"] > 0 and not _is_cable(d["name"])]
    cable = next((d["name"] for d in devices if d["max_output_channels"] > 0
                  and any(d["name"].startswith(name) for name in CABLE_NAMES)), None)
    default_index = sd.query_hostapis(api)["default_input_device"]
    default = sd.query_devices(default_index)["name"] if default_index >= 0 else None
    return {"microphones": microphones, "default": default if default in microphones else None, "cable": cable}


def _device_index(sd, name, kind):
    api = _host_api(sd)
    for index, device in enumerate(sd.query_devices()):
        if device["hostapi"] == api and device["name"] == name and device[f"max_{kind}_channels"] > 0:
            return index
    raise RuntimeError(f"The device \"{name}\" is no longer there.")


def open_streams(microphone, cable, on_input, on_output):
    """Start reading `microphone` into on_input(samples) and feeding `cable` from on_output(frames).

    Returns an object with close(). Raises RuntimeError with what to do when a device cannot be opened.
    """
    try:
        import sounddevice as sd
    except ImportError as err:
        raise RuntimeError("The virtual microphone needs the sounddevice package: pip install sounddevice.") from err
    extra = sd.WasapiSettings(auto_convert=True) if sys.platform == "win32" else None
    try:
        mic_index, cable_index = _device_index(sd, microphone, "input"), _device_index(sd, cable, "output")

        def heard(indata, frames, moment, status):
            on_input(indata[:, 0])

        def wanted(outdata, frames, moment, status):
            outdata[:] = on_output(frames)[:, None]

        streams = [sd.InputStream(device=mic_index, channels=1, samplerate=RATE, blocksize=BLOCK, dtype="float32",
                                  latency="low", callback=heard, extra_settings=extra),
                   sd.OutputStream(device=cable_index, channels=2, samplerate=RATE, blocksize=BLOCK, dtype="float32",
                                   latency="low", callback=wanted, extra_settings=extra)]
    except RuntimeError:
        raise
    except Exception as err:  # PortAudio refused: the device is taken or does not take this format
        raise RuntimeError(f"The microphone or the cable could not be opened ({err}).") from err

    class Streams:
        def close(self):
            for stream in streams:
                stream.stop()
                stream.close()

    for stream in streams:
        stream.start()
    return Streams()


# ------------------------------------------------------------- the switch


class VirtualMicrophone:
    """Sends the chosen microphone to the cable, held back by `picture_delay()` (ms, or None when
    no picture of the user is being sent) plus the user's adjustment.

    `devices` and `opener` are list_devices and open_streams; the tests pass stand-ins.
    `settings` is a file that keeps the chosen microphone and the adjustment between runs.
    """

    def __init__(self, picture_delay, devices=list_devices, opener=open_streams, settings=None):
        self._picture_delay, self._devices, self._opener = picture_delay, devices, opener
        self._settings = settings
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._streams = None
        self.line = None
        self.on = False
        self.reason = None
        self.cable = None
        self.microphone, self.offset_ms = None, 0.0
        self.picture_ms = None  # the picture delay the voice now follows, None outside a camera session
        self.level = 0.0  # how loud the microphone is, 0 to 1, so the person can see that it hears them
        self._found, self._found_at = None, 0.0
        self._load()

    # The chosen microphone and the adjustment are kept; whether it is on is not: it starts off every time.
    def _load(self):
        if self._settings is None or not self._settings.exists():
            return
        try:
            saved = json.loads(self._settings.read_text(encoding="utf-8"))
            self.microphone = saved.get("microphone")
            self.offset_ms = _limit_offset(float(saved.get("offset_ms", 0.0)))
        except (ValueError, OSError, TypeError):
            pass  # a damaged file: start from the defaults

    def _save(self):
        if self._settings is None:
            return
        self._settings.parent.mkdir(parents=True, exist_ok=True)
        self._settings.write_text(json.dumps({"microphone": self.microphone, "offset_ms": self.offset_ms}), encoding="utf-8")

    def change(self, on=None, microphone=None, offset_ms=None):
        """Switch on or off, choose the microphone, or set the adjustment. Returns the state."""
        with self._lock:
            if offset_ms is not None:
                self.offset_ms = _limit_offset(offset_ms)
            restart = False
            if microphone is not None and microphone != self.microphone:
                self.microphone, restart = microphone, self.on
            if microphone is not None or offset_ms is not None:
                self._save()
            if restart:
                self._end()
                self._start()
            if on is True and not self.on:
                self._start()
            elif on is False and self.on:
                self._end()
        return self.snapshot()

    def close(self):
        with self._lock:
            if self.on:
                self._end()

    def total_ms(self):
        """The delay the voice is set to now: the picture's, adjusted, or none beyond the least."""
        if self.picture_ms is None:
            return MIN_DELAY_MS
        return min(max(self.picture_ms + self.offset_ms, MIN_DELAY_MS), MAX_DELAY_MS)

    def found(self, fresh=False):
        """The devices, asked for again at most every few seconds: the status goes out four times a second."""
        now = time.perf_counter()
        if fresh or self._found is None or now - self._found_at > DEVICES_SECONDS:
            try:
                self._found = self._devices()
            except Exception:  # no sound library or no sound system: say so, do not fail the status
                self._found = {"microphones": [], "default": None, "cable": None}
            self._found_at = now
        return self._found

    def snapshot(self):
        found = self.found()
        line = self.line
        cable = self.cable or found["cable"]
        return {
            "on": self.on,
            "reason": self.reason,
            "cable": cable,
            "meeting_device": meeting_name(cable),  # what to choose as the microphone in the meeting application
            "microphone": self.microphone or found["default"],
            "microphones": found["microphones"],
            "offset_ms": self.offset_ms,
            "picture_ms": self.picture_ms,
            "delay_ms": round(line.delay_ms, 1) if self.on and line is not None else None,
            "level": round(self.level, 3) if self.on else 0.0,
        }

    def _start(self):
        found = self.found(fresh=True)
        if found["cable"] is None:
            self.reason = "No virtual cable was found. " + INSTALL.get(sys.platform, "Install a virtual audio cable.")
            return
        microphone = self.microphone if self.microphone in found["microphones"] else found["default"]
        if microphone is None:
            self.reason = "No microphone was found. Connect one and try again."
            return
        self.picture_ms = self._read_picture()
        line = DelayLine(delay_ms=self.total_ms())
        try:
            self._streams = self._opener(microphone, found["cable"], self._heard(line), line.read)
        except RuntimeError as err:
            self.reason = str(err)
            return
        self.line, self.microphone, self.cable = line, microphone, found["cable"]
        self.on, self.reason, self.level = True, None, 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._follow, name="virtual-microphone", daemon=True)
        self._thread.start()

    def _end(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._streams is not None:
            self._streams.close()
        self.on, self._thread, self._streams, self.line, self.level = False, None, None, None, 0.0

    def _heard(self, line):
        def heard(samples):
            line.write(samples)
            peak = float(np.abs(samples).max()) if len(samples) else 0.0
            self.level = max(peak, self.level * 0.9)  # falls back slowly, as a meter does
        return heard

    def _read_picture(self):
        try:
            delay = self._picture_delay()
        except Exception:
            return None
        return None if delay is None else float(delay)

    def _follow(self):
        """Keep the delay with the picture's, without chasing every small change."""
        while not self._stop.wait(UPDATE_SECONDS):
            picture = self._read_picture()
            if picture is None or self.picture_ms is None or abs(picture - self.picture_ms) > FOLLOW_MS:
                self.picture_ms = picture
            line = self.line
            if line is not None:
                line.set_delay(self.total_ms())


def _limit_offset(offset_ms):
    return round(min(max(float(offset_ms), -OFFSET_LIMIT_MS), OFFSET_LIMIT_MS))
