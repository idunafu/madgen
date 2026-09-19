"""Small, CPU-only probes of the existing lyrics renderer; no corpus/model downloads.

Run from the repository root:
    uv run --no-sync --offline python docs/research/probe_lyrics_render_contract.py

The synthetic signals demonstrate control/data mismatches, not perceived speech quality.
Only temporary WAV files are written; JSON results go to stdout.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyworld
import soundfile as sf

from madgen.db import Corpus
from madgen.match import LyricsWeights, _lower_tier
from madgen.synth import SR, NoteJob, render_note
from madgen.target import TargetUnit


def signal(sr: int, varying: bool) -> np.ndarray:
    t = np.arange(sr, dtype=np.float64) / sr
    frequency = np.where(t < 0.65, 180.0, 330.0) if varying else np.full(sr, 220.0)
    phase = 2 * np.pi * np.cumsum(frequency) / sr
    amplitude = np.where(t < 0.65, 0.04, 0.25) if varying else np.full(sr, 0.2)
    # Harmonics provide a periodic, speech-like excitation without using any recordings.
    return amplitude * (np.sin(phase) + 0.25 * np.sin(2 * phase))


def analyze_segment_f0(x: np.ndarray, sr: int, start: float, end: float) -> float:
    f0, t = pyworld.dio(x, sr, frame_period=5.0, f0_floor=60.0, f0_ceil=1100.0)
    f0 = pyworld.stonemask(x, f0, t, sr)
    used = f0[(t >= start) & (t < end) & (f0 > 0)]
    return float(np.median(used))


def main() -> None:
    result = {"versions": {name: version(name) for name in ("numpy", "pyworld", "soundfile")}}
    with tempfile.TemporaryDirectory(prefix="madgen-render-probe-") as temp:
        wav = Path(temp) / "source.wav"
        sf.write(wav, signal(SR, varying=True), SR, subtype="FLOAT")
        reference = analyze_segment_f0(signal(16000, varying=True), 16000, 0.1, 0.9)
        corpus = Corpus(
            ids=np.array([1]), source_id=np.array(["synthetic"]),
            start=np.array([0.1]), end=np.array([0.9]), phoneme=np.array(["a"]),
            f0=np.array([reference]), f0_std=np.array([0.0]), rms_db=np.array([-20.0]),
            video_ref=[None], next_idx=np.array([-1]), audio_cache={}, source_path={},
        )
        unit = TargetUnit(0, 0.0, 0.4, reference, 127, phoneme="a")
        local = _lower_tier(unit, corpus, np.array([0]), True, LyricsWeights(pitch_free_cents=0.0))
        job = NoteJob(str(wav), 0.1, 0.9, reference, reference, 0.0, 0.4, 127,
                      flatten=1.0, threshold_cents=0.0, measure_used_f0=True, sustain_to_note=True)
        _, y, used_f0, corrected, ratio = render_note(job)
        result["selection_vs_render"] = {
            "source_interval_sec": 0.8, "target_duration_sec": 0.4,
            "selection_f0_hz": reference, "selection_pitch_plus_length_cost": int(local[0]),
            "render_used_f0_hz": used_f0, "pitch_corrected": bool(corrected),
            "render_pitch_shift_cents": float(1200 * np.log2(reference / used_f0)),
            "reported_stretch_ratio": ratio, "output_samples_including_tail": len(y),
        }
        with patch("madgen.synth.pyworld.synthesize", wraps=pyworld.synthesize) as synthesize:
            _, _, _, corrected, _ = render_note(replace(job, threshold_cents=None))
            result["pitch_disabled_sustain"] = {
                "pitch_corrected": bool(corrected), "world_synthesize_calls": synthesize.call_count,
            }

        sf.write(wav, signal(SR, varying=False), SR, subtype="FLOAT")
        direct_job = NoteJob(str(wav), 0.1, 0.4, 220.0, 220.0, 0.0, 0.3, 127,
                             flatten=0.0, threshold_cents=None)
        with patch("madgen.synth.pyworld.synthesize", wraps=pyworld.synthesize) as synthesize:
            _, y, _, corrected, _ = render_note(direct_job)
            source, _ = sf.read(wav, start=int(0.1 * SR), frames=len(y), dtype="float32")
            result["pitch_disabled_non_sustain"] = {
                "pitch_corrected": bool(corrected), "world_synthesize_calls": synthesize.call_count,
                "sample_equal_to_source": bool(np.array_equal(source, y)),
                "max_absolute_difference": float(np.max(np.abs(source - y))),
                "source_rms": float(np.sqrt(np.mean(source ** 2))),
                "output_rms": float(np.sqrt(np.mean(y ** 2))),
            }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
