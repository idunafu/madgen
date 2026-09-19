from dataclasses import replace

import numpy as np
import pytest
import pyworld
import soundfile as sf

from madgen.db import Corpus
from madgen.match import select_units_lyrics
from madgen.synth import SR, NoteJob, render_note
from madgen.target import TargetUnit
from madgen.world_plans import WorldPlanStore, plan_consonant


def corpus_for(tmp_path, audio, starts, ends, f0):
    path = tmp_path / "material.wav"
    sf.write(path, audio, SR, subtype="FLOAT")
    n = len(starts)
    corpus = Corpus(
        ids=np.arange(n), source_id=np.array(["s"] * n), start=np.array(starts), end=np.array(ends),
        phoneme=np.array(["a"] * n), f0=np.array(f0), f0_std=np.zeros(n), rms_db=np.zeros(n),
        video_ref=[None] * n, next_idx=np.full(n, -1), audio_cache={"s": str(path)},
        source_path={"s": str(path)}, cand_phoneme=np.array([["a", "", ""]] * n),
        cand_conf=np.array([[.9, 0, 0]] * n),
    )
    return corpus, WorldPlanStore(corpus, tmp_path / "cache")


def test_selection_uses_actual_core_not_stale_db_and_reuses_analysis(tmp_path, monkeypatch):
    t = np.arange(2 * SR) / SR
    hz = np.where(t < .55, 180., np.where(t < 1., 330., 220.))
    amp = np.where(t < .55, .01, .2)
    x = amp * np.sin(2 * np.pi * np.cumsum(hz) / SR)
    c, store = corpus_for(tmp_path, x, [.1, 1.1], [.9, 1.9], [220., 700.])
    units = [TargetUnit(0, 0., .4, 220., 100, phoneme="a")]
    assert select_units_lyrics(units, c).tolist() == [0]
    assert select_units_lyrics(units, c, core_store=store).tolist() == [1]
    assert store.plan(0, .4).reference_f0 == pytest.approx(330, abs=2)
    p = store.plan(1, .4)
    assert p.reference_f0 == pytest.approx(220, abs=2)
    monkeypatch.setattr(pyworld, "dio", lambda *a, **k: pytest.fail("F0 must not be re-analyzed"))
    reload = WorldPlanStore(c, store.cache_dir)
    cached = reload.plan(1, .4)
    np.testing.assert_array_equal(cached.f0, p.f0)
    job = NoteJob(c.audio_cache["s"], 1.1, 1.9, 700., 330., 0., .4, 100, 1., 0.,
                  sustain_to_note=True, core_plan=cached)
    _, y, reference, corrected, ratio = render_note(job)
    assert reference == p.reference_f0 and corrected
    assert ratio == p.stretch_ratio and len(y) == p.output_samples
    interior = y[2000:-2000]
    frequency = np.fft.rfftfreq(len(interior), 1 / SR)[np.argmax(np.abs(np.fft.rfft(interior)))]
    assert frequency == pytest.approx(330, abs=5)


def test_selection_prefers_longer_usable_core(tmp_path):
    t = np.arange(2 * SR) / SR
    amp = np.where(((t > .4) & (t < .5)) | (t > 1.), .2, .001)
    c, store = corpus_for(tmp_path, amp * np.sin(2 * np.pi * 220 * t), [.1, 1.1], [.9, 1.9], [220., 220.])
    u = TargetUnit(0, 0., .7, 220., 100, phoneme="a")
    assert store.plan(0, .7).stretch_ratio > 4
    assert select_units_lyrics([u], c, core_store=store).tolist() == [1]


def test_truncated_core_f0_only_describes_used_frames(tmp_path):
    t = np.arange(SR) / SR
    hz = np.where(t < .45, 180., 330.)
    c, store = corpus_for(tmp_path, .2 * np.sin(2 * np.pi * np.cumsum(hz) / SR), [.1], [.9], [330.])
    short = store.plan(0, .2)
    assert short.reference_f0 == pytest.approx(180, abs=2)
    assert short.core_end - short.core_start == short.output_frames
    assert short.stretch_ratio == 1


def test_short_plosive_keeps_burst_and_ends_at_vowel(tmp_path):
    rng = np.random.default_rng(12)
    x = rng.normal(0, .0001, round(.09 * SR))
    a, b = round(.035 * SR), round(.039 * SR)
    x[a:b] = rng.normal(0, .12, b - a)
    x[b:round(.06 * SR)] = rng.normal(0, .015, round(.06 * SR) - b)
    path = tmp_path / "burst.wav"
    sf.write(path, x, SR, subtype="FLOAT")
    job = NoteJob(str(path), 0, .06, float("nan"), 330, 1., .06, 100, 1, None, level_db=-24)
    p = plan_consonant(job, "t", joins_vowel=True)
    assert p.burst_trimmed and p.source_start < a
    assert p.source_end == round(.06 * SR)
    assert p.output_start + p.source_end - p.source_start == round(1.06 * SR)
    start, y, _, corrected, _ = render_note(replace(job, consonant_plan=p))
    assert start >= SR and not corrected
    # The strongest burst sample survives; only the two outer fades may change it.
    peak = int(np.argmax(np.abs(x[a:b]))) + a
    assert y[peak - p.source_start] == pytest.approx(x[peak] * p.gain, rel=1e-5)
    assert p.gain <= 10 ** (6 / 20)


@pytest.mark.parametrize("phone", ["s", "sh", "ch", "m", "h"])
def test_consonant_no_burst_trimming_for_other_manners(tmp_path, phone):
    x = np.zeros(round(.04 * SR))
    x[200:300] = .001
    path = tmp_path / "soft.wav"
    sf.write(path, x, SR, subtype="FLOAT")
    job = NoteJob(str(path), 0., .04, float("nan"), 330, 1., .06, 100, 1., None, level_db=-24)
    p = plan_consonant(job, phone, joins_vowel=False)
    assert p.source_start == 0 and not p.burst_trimmed
    assert p.output_start == SR  # never move a stand-alone consonant across a rest
    assert p.gain == pytest.approx(10 ** (6 / 20))


def test_analysis_cache_invalidates_when_audio_changes(tmp_path):
    t = np.arange(SR) / SR
    c, store = corpus_for(tmp_path, .2 * np.sin(2 * np.pi * 220 * t), [.1], [.9], [220.])
    first = store.plan(0, .4)
    sf.write(c.audio_cache["s"], .2 * np.sin(2 * np.pi * 330 * t), SR, subtype="FLOAT")
    changed = WorldPlanStore(c, store.cache_dir).plan(0, .4)
    assert changed.analysis_key != first.analysis_key
    assert changed.reference_f0 == pytest.approx(330, abs=2)
