"""Replay saved singing plans with fixed material, accompaniment, and master gain."""

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from madgen.source_synth import SIBILANTS, SR, TransformPlan, deess, render_group, render_plans


def run(args):
    entries = json.loads((args.before / "plan.json").read_text(encoding="utf-8"))
    singing = [e for e in entries if e.get("transform", {}).get("renderer") == "source"]
    if len({e["voice"] for e in singing}) != 1:
        raise ValueError("comparison requires one source singing voice")
    keys = {f.name for f in fields(TransformPlan)}
    plans = [TransformPlan(**{k: v for k, v in e["transform"].items() if k in keys}) for e in singing]
    if any(p.target_rms_db is None or p.deess_db for p in plans):
        raise ValueError("comparison requires singing-level plans without de-essing")
    vocals, sr = sf.read(args.before / "vocals.wav", dtype="float32")
    mix, mix_sr = sf.read(args.before / "mix.wav", dtype="float32")
    assert sr == mix_sr == SR and vocals.ndim == mix.ndim == 1 and len(vocals) == len(mix)
    # render_plans allocates ceil(total_sec * SR)+1. Extra padding avoids float rounding issues.
    duration = (len(vocals) + 1) / SR
    cache = {p.output_start: render_group([p]) for p in plans}
    assert len(cache) == len(plans), "comparison expects unique unit start positions"

    def cached_group(group):
        assert len(group) == 1
        p = group[0]
        return deess(cache[p.output_start].copy(), p.deess_db)

    with patch("madgen.source_synth.render_group", cached_group):
        raw = render_plans(plans, duration)[:len(vocals)]
    stable = np.zeros(len(raw), dtype=bool)
    for p in plans:
        if p.preserve_waveform:
            a, b = p.output_start + round(.025 * SR), p.output_start + p.output_length - round(.025 * SR)
            if b > a:
                stable[a:b] = True
    master = float(np.dot(raw[stable].astype(float), vocals[stable]) / np.dot(raw[stable].astype(float), raw[stable]))
    reconstruction_error = float(np.max(np.abs(raw * master - vocals)))
    stable_error = float(np.max(np.abs(raw[stable] * master - vocals[stable])))
    if stable_error > 2 / 32768:
        raise ValueError(f"protected source copies do not match saved vocals: {stable_error}")
    # Praat's unvoiced resynthesis is not sample-identical on replay. Cache each unit once
    # for a controlled comparison; preserve the saved accompaniment and master gain.
    mix = mix - vocals + raw * master
    vocals = raw * master
    args.out.mkdir(parents=True, exist_ok=True)
    lo, hi = round(18 * SR), round(30 * SR)
    for name, wave in [("mix", mix), ("vocals", vocals)]:
        sf.write(args.out / f"{name}-before-18-30.wav", wave[lo:hi], SR, subtype="FLOAT")
    eligible = [i for i, e in enumerate(singing) if e["phoneme"] in SIBILANTS]
    subset = [plans[i] for i in eligible]
    with patch("madgen.source_synth.render_group", cached_group):
        original = render_plans(subset, duration)[:len(vocals)]
    report = {"input": str(args.before), "same_material_and_timing": True,
              "common_master_gain": master, "reconstruction_error": reconstruction_error,
              "protected_copy_error": stable_error, "cached_resynthesis_for_all_variants": True,
              "eligible_units": len(eligible), "variants": {}}
    for db in (6, 9):
        changed = [replace(p, deess_db=db) for p in subset]
        with patch("madgen.source_synth.render_group", cached_group):
            processed = render_plans(changed, duration)[:len(vocals)]
        delta = (processed - original) * master
        output = {}
        for name, before in [("mix", mix), ("vocals", vocals)]:
            wave = before + delta
            assert np.isfinite(wave).all() and np.max(np.abs(wave)) < 1
            sf.write(args.out / f"{name}-{db}db-18-30.wav", wave[lo:hi], SR, subtype="FLOAT")
            if db == 6:
                sf.write(args.out / f"{name}.wav", wave, SR, subtype="FLOAT")
            output[name] = {"samples": len(wave), "peak": float(np.max(np.abs(wave)))}
        reductions = []
        for i in eligible:
            p = plans[i]
            a, b = p.output_start, p.output_start + p.output_length
            window = np.hanning(b - a)
            band = np.fft.rfftfreq(b - a, 1 / SR) >= 6000
            energies = [float(np.sum(np.abs(np.fft.rfft(w[a:b] * window)[band]) ** 2))
                        for w in (original, processed)]
            if energies[0] > 1e-10:
                reductions.append(float(10 * np.log10(max(energies[1], 1e-30) / energies[0])))
        output["high_band_change_db_p10_p50_p90"] = np.percentile(reductions, [10, 50, 90]).tolist()
        report["variants"][str(db)] = output
    for i in eligible:
        singing[i]["transform"] = replace(plans[i], deess_db=6).to_dict()
    (args.out / "plan.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, default=Path("work/iwashi_lyrics_sample/praat_sustain"))
    parser.add_argument("--out", type=Path, default=Path("work/iwashi_lyrics_sample/praat_deess"))
    run(parser.parse_args())
