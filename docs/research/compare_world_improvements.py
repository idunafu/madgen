"""Controlled WORLD comparison: fixed material, core-aware selection, and consonant processing."""

import argparse
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from madgen.db import load_corpus
from madgen.match import LyricsWeights, select_units_lyrics
from madgen.phonemes import is_voiced_sustained
from madgen.synth import SR, NoteJob, render_note
from madgen.ust import load_ust
from madgen.world_plans import WorldPlanStore, plan_consonant


def run(args):
    voices = load_ust(args.ust)
    assert len(voices) == 1
    units = voices[0].units
    old = {e["index"]: e for e in json.loads((args.before / "plan.json").read_text(encoding="utf-8"))
           if e["voice"] == voices[0].label}
    with sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        corpus = load_corpus(conn, "phoneme")
    positions = {int(sid): i for i, sid in enumerate(corpus.ids)}
    legacy = [positions[old[u.index]["segment_id"]] for u in units]
    store = WorldPlanStore(corpus, args.db.with_suffix(args.db.suffix + ".cache") / "world-core-v1")
    print("Selecting from actual WORLD cores (first run populates analysis cache)...", flush=True)
    selected = select_units_lyrics(units, corpus, LyricsWeights(pitch_free_cents=0.), core_store=store)
    print(f"Selection done: {len(store.analyses)} analyzed segments", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    total = max(u.start_sec + u.duration_sec for u in units) + .5
    output_length = max(round(total * SR) + 1, sf.info(args.before / "vocals.wav").frames,
                        sf.info(args.before / "accompaniment.wav").frames)
    variants, metadata = {}, {}
    modes = [("00-legacy", legacy, False, False), ("01-core-fixed", legacy, True, False),
             ("02-consonants-fixed", legacy, False, True), ("03-core-selected", selected, True, False),
             ("04-combined", selected, True, True)]
    lo, hi = round(args.start * SR), round(args.end * SR)
    for name, indices, cores, consonants in modes:
        print(f"Rendering {name}...", flush=True)
        wave = np.zeros(output_length, dtype=np.float32)
        plans = []
        for i, (u, index) in enumerate(zip(units, indices, strict=True)):
            if not args.full and not (u.start_sec < args.end and u.start_sec + u.duration_sec + .02 > args.start):
                continue
            sustained = is_voiced_sustained(u.phoneme)
            job = NoteJob(corpus.audio_cache[corpus.source_id[index]], float(corpus.start[index]),
                          float(corpus.end[index]), float(corpus.f0[index]), u.target_f0_hz,
                          u.start_sec, u.duration_sec, u.velocity, 1., 0. if sustained else None,
                          level_db=-18 if sustained else -24,
                          measure_used_f0=sustained, sustain_to_note=sustained)
            if cores and sustained:
                job.core_plan = store.plan(int(index), u.duration_sec)
            if consonants and not sustained:
                next_u = units[i + 1] if i + 1 < len(units) else None
                joins = (next_u is not None and is_voiced_sustained(next_u.phoneme)
                         and round((u.start_sec + u.duration_sec) * SR) == round(next_u.start_sec * SR))
                job.consonant_plan = plan_consonant(job, u.phoneme, joins_vowel=joins)
            start, audio, f0, corrected, ratio = render_note(job)
            wave[start:start + len(audio)] += audio
            plans.append({"index": u.index, "phoneme": u.phoneme, "segment_id": int(corpus.ids[index]),
                          "output_start": start, "output_length": len(audio),
                          "reference_f0": None if np.isnan(f0) else f0, "stretch_ratio": ratio,
                          "corrected": bool(corrected),
                          "core": None if job.core_plan is None else job.core_plan.to_dict(),
                          "consonant": None if job.consonant_plan is None else asdict(job.consonant_plan)})
        variants[name] = wave
        metadata[name] = plans
    existing, sr = sf.read(args.before / "vocals.wav", dtype="float32")
    backing, back_sr = sf.read(args.before / "accompaniment.wav", dtype="float32")
    assert sr == back_sr == SR
    master = float(np.sqrt(np.mean(existing[lo:hi].astype(float) ** 2)
                           / np.mean(variants["00-legacy"][lo:hi].astype(float) ** 2)))
    length = len(next(iter(variants.values())))
    backing = np.pad(backing, (0, max(0, length - len(backing))))[:length]
    peak = max(float(np.max(np.abs(y * master + backing))) for y in variants.values())
    headroom = min(1., .95 / max(peak, 1e-12))
    report = {"range_sec": [args.start, args.end], "full": args.full,
              "common_master_gain": master, "common_headroom_gain": headroom,
              "changed_materials": int(np.count_nonzero(np.array(legacy) != selected)),
              "analyzed_segments": len(store.analyses), "variants": {}}
    for name, y in variants.items():
        voice = y * master * headroom
        mix = voice + backing * headroom
        assert np.isfinite(mix).all() and np.max(np.abs(mix)) <= .951
        for group, audio in [("vocals", voice), ("mix", mix)]:
            sf.write(args.out / f"{name}-{group}-18-30.wav", audio[lo:hi], SR, subtype="FLOAT")
            if args.full and name in ("00-legacy", "04-combined"):
                sf.write(args.out / f"{name}-{group}.wav", audio, SR, subtype="FLOAT")
        report["variants"][name] = {"peak": float(np.max(np.abs(mix))), "plans": metadata[name]}
    (args.out / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                                               encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "variants"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("work/corpus_sample.sqlite"))
    parser.add_argument("--ust", type=Path, default=Path("target/iwashi_madgen.ust"))
    parser.add_argument("--before", type=Path, default=Path("work/iwashi_lyrics_sample"))
    parser.add_argument("--out", type=Path, default=Path("work/world-improvements"))
    parser.add_argument("--start", type=float, default=18)
    parser.add_argument("--end", type=float, default=30)
    parser.add_argument("--full", action="store_true")
    run(parser.parse_args())
