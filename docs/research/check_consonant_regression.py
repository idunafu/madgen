"""Reproduce the reported 18-30 s regression, keeping existing outputs untouched.

Reads an existing plan and vocals WAV, renders with fixed selected materials and then
with fresh selection. The before audio is the user's actual output, not a recreation.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import soundfile as sf

from madgen.db import load_corpus
from madgen.match import select_source_plans
from madgen.source_synth import SR, AnalysisStore, balance_consonant_joins, make_plan, render_plans
from madgen.target import TargetUnit


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    entries = json.loads((args.previous / "plan.json").read_text(encoding="utf-8"))
    entries = [e for e in entries if e.get("transform")
               and e["note_start_sec"] < args.end + .25
               and e["note_start_sec"] + e["note_duration_sec"] > args.start - .25]
    if len({e["voice"] for e in entries}) != 1:
        raise ValueError("select an excerpt with exactly one source-rendered singing voice")
    with sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        corpus = load_corpus(conn, "phoneme")
    store = AnalysisStore(corpus, args.db.with_suffix(args.db.suffix + ".cache") / "source-v1")
    positions = {int(id_): i for i, id_ in enumerate(corpus.ids)}
    units = [TargetUnit(e["index"], e["note_start_sec"], e["note_duration_sec"],
                        e["target_f0_hz"], 100, phoneme=e["phoneme"]) for e in entries]
    plans = [make_plan(u, positions[e["segment_id"]], store) for u, e in zip(units, entries, strict=True)]
    plans = balance_consonant_joins(plans, units)
    end = max(p.output_start + p.output_length for p in plans) / SR
    fixed = render_plans(plans, end)
    previous, sr = sf.read(args.previous / "vocals.wav", dtype="float32")
    if sr != SR or previous.ndim != 1:
        raise ValueError("expected mono 44.1 kHz vocals")
    # Freeze the original master level using unchanged vowel interiors; do not normalize A/B independently.
    gains = []
    for p in plans:
        if p.preserve_waveform or p.output_length < .08 * SR:
            continue
        a, b = p.output_start + round(.02 * SR), p.output_start + p.output_length - round(.02 * SR)
        power = np.mean(fixed[a:b] ** 2)
        if power > 1e-12:
            gains.append(float(np.sqrt(np.mean(previous[a:b] ** 2) / power)))
    gain = float(np.median(gains))
    lo, hi = round(args.start * SR), round(args.end * SR)
    sf.write(args.out / "before.wav", previous[lo:hi], SR, subtype="FLOAT")
    sf.write(args.out / "after-fixed-material.wav", fixed[lo:hi] * gain, SR, subtype="FLOAT")
    selected, fresh = select_source_plans(units, corpus, store)
    after = render_plans(fresh, end)
    sf.write(args.out / "after-reselected.wav", after[lo:hi] * gain, SR, subtype="FLOAT")
    rows = []
    for old, p, q, index in zip(entries, plans, fresh, selected, strict=True):
        rows.append({"time": old["note_start_sec"], "phoneme": old["phoneme"],
                     "before_segment": old["segment_id"], "after_segment": int(corpus.ids[index]),
                     "before_transform": old["transform"], "fixed_transform": p.to_dict(),
                     "reselected_transform": q.to_dict()})
    result = {"start": args.start, "end": args.end, "common_master_gain": gain,
              "scope": "default source settings, single voice; same-material correction and full reselection",
              "units": rows}
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                                              encoding="utf-8")
    print(json.dumps({"units": len(rows), "common_master_gain": gain,
                      "protected_consonants": sum(p.preserve_waveform for p in plans),
                      "attenuated_consonants": sum(p.gain < 1 for p in plans),
                      "output": str(args.out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("work/consonant-regression-18-30"))
    parser.add_argument("--start", type=float, default=18)
    parser.add_argument("--end", type=float, default=30)
    run(parser.parse_args())
