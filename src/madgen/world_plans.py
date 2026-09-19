"""Cache WORLD core analysis and freeze the spans used for lyrics selection/rendering."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf

from .phonemes import CONSONANTS
from .synth import (
    CORE_MAX_GAP_FRAMES,
    CORE_MIN_FRAMES,
    CORE_RANGE_DB,
    FRAME_PERIOD,
    PAD_SEC,
    SR,
    XFADE_SEC,
    ConsonantPlan,
    CorePlan,
    analyze_core,
)


class WorldPlanStore:
    def __init__(self, corpus, cache_dir: Path):
        self.corpus = corpus
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sources = {}
        self.analyses = {}

    def _analysis(self, index):
        index = int(index)
        if index in self.analyses:
            return self.analyses[index]
        c = self.corpus
        path = c.audio_cache[c.source_id[index]]
        if path not in self.sources:
            # Hash contents, not just the DB id: replacement audio must invalidate the cache.
            with open(path, "rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            info = sf.info(path)
            if info.samplerate != SR or info.channels != 1:
                raise ValueError("WORLD planning requires mono 44100 Hz cached audio")
            self.sources[path] = (digest, info.duration)
        digest, duration = self.sources[path]
        start, end = float(c.start[index]), float(c.end[index])
        read_start_sec = max(0., start - PAD_SEC)
        lo, hi = int(read_start_sec * SR), int(min(duration, end + PAD_SEC) * SR)
        offset = int((start - read_start_sec) * SR)
        settings = (1, digest, pyworld.__version__, SR, FRAME_PERIOD, CORE_RANGE_DB,
                    CORE_MIN_FRAMES, CORE_MAX_GAP_FRAMES, lo, hi, offset, end - start)
        key = hashlib.sha256(json.dumps(settings).encode()).hexdigest()
        cache = self.cache_dir / f"{key}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                f0, times = data["f0"], data["times"]
                a, b = int(data["a"]), int(data["b"])
        else:
            x, _ = sf.read(path, start=lo, stop=hi, dtype="float64")
            f0, times, a, b = analyze_core(x, offset, end - start)
            temporary = cache.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, f0=f0, times=times, a=a, b=b)
            temporary.replace(cache)
        result = (key, lo, hi, f0, times, a, b)
        self.analyses[index] = result
        return result

    def plan(self, index: int, duration: float) -> CorePlan:
        key, lo, hi, f0, times, a, b = self._analysis(index)
        length = duration + XFADE_SEC
        n_out = max(1, int(np.ceil(length * 1000 / FRAME_PERIOD)) + 1)
        # WORLD truncates long cores. Judge only the frames that will actually be used.
        b = min(b, a + n_out)
        voiced = f0[a:b][f0[a:b] > 0]
        ref = float(np.median(voiced)) if len(voiced) >= 3 else float("nan")
        return CorePlan(lo, hi, a, b, f0, times, ref, n_out, int(length * SR),
                        max(1., n_out / (b - a)), key)


def plan_consonant(job, phoneme: str, *, joins_vowel: bool, max_boost_db: float = 6.) -> ConsonantPlan:
    """Keep the raw attack, fit within the consonant slot, and never amplify quiet closure as speech.

    Only a clear energy rise in an unvoiced plosive can trim a leading prefix. Fricatives
    and voiced consonants retain their annotated beginning. No time stretching or denoising.
    """
    start, end = round(job.seg_start * SR), round(job.seg_end * SR)
    audio, _ = sf.read(job.audio_path, start=start, stop=end, dtype="float64")
    end = start + len(audio)
    slot_start = round(job.note_start * SR)
    slot_end = round((job.note_start + job.note_dur) * SR)
    slot = slot_end - slot_start
    trimmed = False
    phone = CONSONANTS.get(phoneme)
    if len(audio) and phone is not None and phone[1] == "plosive" and not phone[2]:
        hop = max(1, round(.001 * SR))
        blocks = np.array([np.mean(audio[i:i + hop] ** 2) for i in range(0, len(audio), hop)])
        if len(blocks) >= 8:
            previous = np.array([np.mean(blocks[max(0, i - 5):i]) if i else blocks[0]
                                 for i in range(len(blocks))])
            peak = int(np.argmax(blocks - previous))
            if peak >= 3 and blocks[peak] > 4 * max(previous[peak], 1e-12):
                # Keep 2 ms before the burst, not just its strongest sample.
                trim = max(0, (peak - 2) * hop)
                start += trim
                audio = audio[trim:]
                trimmed = trim > 0
    audio = audio[:slot]
    end = start + len(audio)
    # Active 5 ms windows, rather than closure/silence, set consonant loudness.
    hop = max(1, round(.005 * SR))
    energy = np.array([np.mean(audio[i:i + hop] ** 2) for i in range(0, len(audio), hop)])
    if len(energy):
        active = energy[energy >= np.max(energy) * .1]
        rms = math.sqrt(float(np.mean(active)))
    else:
        rms = 0.
    target = 10 ** (job.level_db / 20) * job.velocity / 127
    gain = min(10 ** (max_boost_db / 20), target / max(rms, 1e-9))
    output_start = slot_end - len(audio) if joins_vowel else slot_start
    return ConsonantPlan(start, end, output_start, gain, trimmed)
