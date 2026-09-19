"""Hold the old WORLD selections fixed and add Praat singing processing step by step."""

import argparse
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from madgen.db import load_corpus
from madgen.phonemes import is_voiced_sustained
from madgen.source_synth import SR, AnalysisStore, make_plan, render_plans, singing_overlap
from madgen.synth import NoteJob, render_note
from madgen.ust import load_ust


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    voices = load_ust(args.ust)
    if len(voices) != 1:
        raise ValueError("this comparison requires one singing voice")
    units = [u for u in voices[0].units if u.start_sec < args.end + .25
             and u.start_sec + u.duration_sec > args.start - .25]
    old = {e["index"]: e for e in json.loads((args.old / "plan.json").read_text(encoding="utf-8"))
           if e["voice"] == voices[0].label}
    with sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        corpus = load_corpus(conn, "phoneme")
    positions = {int(id_): i for i, id_ in enumerate(corpus.ids)}
    indices = [positions[old[u.index]["segment_id"]] for u in units]
    store = AnalysisStore(corpus, args.db.with_suffix(args.db.suffix + ".cache") / "source-v1")
    base, cores, levels, joined = [], [], [], []
    for i, (u, index) in enumerate(zip(units, indices, strict=True)):
        base.append(make_plan(u, index, store))
        level = make_plan(u, index, store, sustain=True)
        cores.append(replace(level, target_rms_db=None, level_sustain=False))
        levels.append(level)
        joined.append(make_plan(u, index, store, sustain=True, overlap_samples=singing_overlap(units, i)))
    duration = max(p.output_start + p.output_length for p in joined) / SR + .02
    lo, hi = round(args.start * SR), round(args.end * SR)
    existing, sr = sf.read(args.old / "vocals.wav", dtype="float32")
    assert sr == SR and existing.ndim == 1
    # Reconstruct the old jobs only to recover the old mix's single master gain.
    reference = np.zeros(round(duration * SR) + 2, dtype=np.float32)
    for u, index in zip(units, indices, strict=True):
        sustained = is_voiced_sustained(u.phoneme)
        job = NoteJob(corpus.audio_cache[corpus.source_id[index]], float(corpus.start[index]),
                      float(corpus.end[index]), float(corpus.f0[index]), u.target_f0_hz,
                      u.start_sec, u.duration_sec, u.velocity, 1.0, 0 if sustained else None,
                      level_db=-18 if sustained else -24, measure_used_f0=sustained, sustain_to_note=sustained)
        start, audio, *_ = render_note(job)
        reference[start:start + len(audio)] += audio
    master = float(np.sqrt(np.mean(existing[lo:hi] ** 2) / np.mean(reference[lo:hi] ** 2)))
    sf.write(args.out / "00-world-existing.wav", existing[lo:hi], SR, subtype="FLOAT")
    report = {"range_sec": [args.start, args.end], "common_master_gain": master,
              "same_segment_ids_for_all_praat_variants": [old[u.index]["segment_id"] for u in units],
              "variants": {}}
    for name, plans in [("01-praat-preserve", base), ("02-praat-core", cores),
                        ("03-praat-core-level", levels), ("04-praat-core-level-overlap", joined)]:
        wave = render_plans(plans, duration) * master
        sf.write(args.out / f"{name}.wav", wave[lo:hi], SR, subtype="FLOAT")
        vowel_db = []
        for u in units:
            if is_voiced_sustained(u.phoneme) and args.start <= u.start_sec < args.end:
                a, b = round(u.start_sec * SR), round((u.start_sec + u.duration_sec) * SR)
                vowel_db.append(float(20 * np.log10(np.sqrt(np.mean(wave[a:b] ** 2)) + 1e-12)))
        report["variants"][name] = {
            "vowel_rms_db_p10_p50_p90": np.percentile(vowel_db, [10, 50, 90]).tolist(),
            "peak": float(np.max(np.abs(wave[lo:hi]))), "finite": bool(np.isfinite(wave).all()),
            "plans": [p.to_dict() for p in plans]}
    (args.out / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                                              encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in value.items() if k != "plans"}
                      for name, value in report["variants"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=Path("work/iwashi_lyrics_sample"))
    parser.add_argument("--db", type=Path, default=Path("work/corpus_sample.sqlite"))
    parser.add_argument("--ust", type=Path, default=Path("target/iwashi_madgen.ust"))
    parser.add_argument("--out", type=Path, default=Path("work/praat-sustain-18-30"))
    parser.add_argument("--start", type=float, default=18)
    parser.add_argument("--end", type=float, default=30)
    run(parser.parse_args())
