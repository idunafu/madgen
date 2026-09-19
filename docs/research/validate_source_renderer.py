"""Small, reproducible real-material comparison; read-only DB, no recognition or GPU work.

uv run --no-sync --offline python docs/research/validate_source_renderer.py --db work/corpus.sqlite
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import parselmouth
import pyworld
import soundfile as sf

from madgen.db import load_corpus
from madgen.source_synth import SR, AnalysisStore, make_plan, render_group
from madgen.synth import NoteJob, render_note
from madgen.target import TargetUnit


def measured_pitch(audio, target):
    # Independent estimator; this is not the Praat pitch used to construct the plan.
    f0, _ = pyworld.harvest(audio.astype(np.float64), SR, f0_floor=60, f0_ceil=1100)
    voiced = f0[f0 > 0]
    return {"voiced_fraction": float(np.mean(f0 > 0)),
            "median_abs_cents": float(np.median(np.abs(1200 * np.log2(voiced / target))))
            if len(voiced) else None}


def run(db_path, output):
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        corpus = load_corpus(conn, "phoneme")
    store = AnalysisStore(corpus, output / "analysis")
    eligible = np.flatnonzero((corpus.phoneme == "a") & (corpus.end - corpus.start > .15)
                             & (corpus.end - corpus.start < .6))
    rows, seen = [], set()
    for index in eligible:
        if corpus.source_id[index] in seen:
            continue
        seen.add(corpus.source_id[index])
        start, end = round(corpus.start[index] * SR), round(corpus.end[index] * SR)
        duration = (end - start) / SR
        identity = make_plan(TargetUnit(0, 0, duration, 220, 100, phoneme="a"), int(index),
                             store, threshold=None)
        original, _ = sf.read(identity.audio_path, start=start, stop=end, dtype="float32")
        copied = render_group([identity])
        target = (identity.reference_f0 or 220) * 2 ** (3 / 12)
        unit = TargetUnit(0, 0, duration * 2, target, 100, phoneme="a")
        plan = make_plan(unit, int(index), store)
        tick = time.perf_counter()
        audio = render_group([plan])
        source_seconds = time.perf_counter() - tick
        job = NoteJob(plan.audio_path, start / SR, end / SR, float(corpus.f0[index]), target,
                      0, duration * 2, 100, 1.0, 0.0, measure_used_f0=True, sustain_to_note=True)
        tick = time.perf_counter()
        _, legacy, legacy_f0, _, legacy_ratio = render_note(job)
        legacy_seconds = time.perf_counter() - tick
        prefix = str(int(corpus.ids[index]))
        for label, wave in [("original", original), ("source", audio), ("world", legacy)]:
            sf.write(output / f"{prefix}-{label}.wav", wave, SR, subtype="FLOAT")
        following = int(corpus.next_idx[index])
        joined_exact = None
        if following >= 0 and round(corpus.start[following] * SR) == end:
            next_end = round(corpus.end[following] * SR)
            next_unit = TargetUnit(1, duration, (next_end - end) / SR, target, 100,
                                   phoneme=str(corpus.phoneme[following]))
            next_plan = make_plan(next_unit, following, store, threshold=None)
            joined = render_group([identity, next_plan])
            actual, _ = sf.read(plan.audio_path, start=start, stop=next_end, dtype="float32")
            joined_exact = bool(np.array_equal(joined, actual))
        rows.append({"segment_id": int(corpus.ids[index]), "source_id": str(corpus.source_id[index]),
                     "raw_exact": bool(np.array_equal(copied, original)), "joined_raw_exact": joined_exact,
                     "source_frames": len(audio), "expected_frames": plan.output_length,
                     "source_finite": bool(np.isfinite(audio).all()), "target_hz": target,
                     "source_pitch": measured_pitch(audio, target), "world_pitch": measured_pitch(legacy, target),
                     "source_render_seconds": source_seconds, "world_render_seconds": legacy_seconds,
                     "world_frames": len(legacy),
                     "world_used_f0": float(legacy_f0) if np.isfinite(legacy_f0) else None,
                     "world_stretch_ratio": legacy_ratio,
                     "plan": plan.to_dict()})
        if len(rows) == 3:
            break
    result = {"database": str(db_path), "parselmouth": parselmouth.__version__,
              "praat": parselmouth.PRAAT_VERSION, "cases": rows,
              "scope": "3 distinct recordings, original vs +3 semitones and 2x duration; no listening judgement. "
                       "WORLD changes the used span and gain: end-to-end paths, not a controlled engine ranking."}
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                                       encoding="utf-8")
    print(json.dumps([{k: v for k, v in row.items() if k != "plan"} for row in rows], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("work/source-render-validation"))
    args = parser.parse_args()
    run(args.db, args.out)
