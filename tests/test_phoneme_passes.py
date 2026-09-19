"""Model lifetimes and resumability without downloading or running ML models."""

import gc
import sqlite3
import weakref

import numpy as np
import pytest
import soundfile as sf

from madgen import corpus
from madgen import phoneme_analysis as pa


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    sources = [tmp_path / f"{i}.wav" for i in range(3)]
    for i, path in enumerate(sources):
        sf.write(path, np.full(1600, (i + 1) / 10, dtype=np.float32), 16000)
    monkeypatch.setattr(corpus, "analyze_frames", lambda *args, **kwargs: (np.zeros(20), np.full(20, -80.)))
    state = {"asr_loads": 0, "asr_unloads": 0, "phoneme_loads": 0, "transcriptions": [], "alignments": [],
             "fail_transcription": None, "fail_alignment": None, "empty": False}

    class Transcriber:
        pass

    def load_transcriber(*args):
        state["asr_loads"] += 1
        obj = Transcriber()
        state["asr_ref"] = weakref.ref(obj)
        return obj

    def transcribe(audio, model, *, label):
        label = str(round(float(audio[0]), 1))
        if label == state["fail_transcription"]:
            raise KeyboardInterrupt
        state["transcriptions"].append(label)
        return [] if state["empty"] else [{"start": 0., "end": .1, "text": label}]

    class PhonemeModel:
        def __init__(self, device):
            gc.collect()
            assert "asr_ref" not in state or state["asr_ref"]() is None
            assert state["asr_loads"] == state["asr_unloads"]
            state["phoneme_loads"] += 1
            state["phoneme_ref"] = weakref.ref(self)

        def posteriors(self, audio):
            return np.empty(0)

    def align(audio, utterances, model):
        label = utterances[0]["text"]
        assert label == str(round(float(audio[0]), 1))
        if label == state["fail_alignment"]:
            raise RuntimeError("interrupted alignment")
        state["alignments"].append(label)
        return [pa.PhonemeSegment(0., .1, "a", [("a", .8), ("i", .1), ("u", .1)], 220., 0., -20.)]

    def unload_transcriber(model):
        state["asr_unloads"] += 1

    monkeypatch.setattr(pa, "load_transcriber", load_transcriber)
    monkeypatch.setattr(pa, "unload_transcriber", unload_transcriber)
    monkeypatch.setattr(pa, "transcribe_with_model", transcribe)
    monkeypatch.setattr(pa, "PhonemeModel", PhonemeModel)
    monkeypatch.setattr(pa, "analyze_utterances", align)
    monkeypatch.setattr(pa, "_g2p", lambda text: ["a"])
    monkeypatch.setattr(pa, "_postprocess_utterance", lambda audio, a0, phones, probs, model: (
        [(seg, "a", 1.) for seg in align(audio, [{"text": str(round(float(audio[0]), 1))}], model)], 0))
    monkeypatch.setattr(pa, "release_models", lambda device: gc.collect())
    return sources, tmp_path / "corpus.sqlite", state


def build(sources, path, model="large-v3"):
    corpus.build_corpus(sources, path, workers=1, phonemes="wav2vec2", device="cpu", whisper_model=model)


def test_two_passes_release_models_and_skip_completed(pipeline, capsys):
    sources, path, state = pipeline
    build([*sources, sources[0]], path)
    assert state["asr_loads"] == state["phoneme_loads"] == 1
    assert state["transcriptions"] == state["alignments"] == ["0.1", "0.2", "0.3"]
    assert state["asr_ref"]() is None and state["phoneme_ref"]() is None
    terminal = capsys.readouterr().err
    assert "transcription complete 1/3 (33.3%)" in terminal
    assert "transcription complete 3/3 (100.0%)" in terminal
    assert "phoneme complete 3/3 (100.0%)" in terminal
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM sources WHERE phonemes_analyzer='wav2vec2'").fetchone()[0] == 3
        assert conn.execute("SELECT count(*) FROM phoneme_candidates").fetchone()[0] == 9
    build(sources, path)
    assert state["asr_loads"] == state["phoneme_loads"] == 1
    assert not list(path.with_suffix(".sqlite.cache").glob("*.16k.wav"))


def test_resume_transcription_uses_completed_files(pipeline):
    sources, path, state = pipeline
    state["fail_transcription"] = "0.2"
    with pytest.raises(KeyboardInterrupt):
        build(sources, path)
    assert state["phoneme_loads"] == 0
    state["fail_transcription"] = None
    build(sources, path)
    assert state["transcriptions"] == ["0.1", "0.2", "0.3"]
    assert state["asr_loads"] == 2 and state["phoneme_loads"] == 1


def test_resume_alignment_does_not_load_whisper(pipeline):
    sources, path, state = pipeline
    state["fail_alignment"] = "0.2"
    with pytest.raises(RuntimeError, match="interrupted alignment"):
        build(sources, path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM sources WHERE phonemes_analyzer='wav2vec2'").fetchone()[0] == 1
    state["fail_alignment"] = None
    build(sources, path)
    assert state["asr_loads"] == 1 and state["phoneme_loads"] == 2
    assert state["alignments"] == ["0.1", "0.2", "0.3"]


def test_transcript_cache_tracks_model_and_source(pipeline):
    sources, path, state = pipeline
    state["fail_transcription"] = "0.2"
    with pytest.raises(KeyboardInterrupt):
        build(sources, path)
    state["fail_transcription"] = None
    # Changing the model invalidates the saved first transcript.
    build(sources, path, model="small")
    assert state["transcriptions"] == ["0.1", "0.1", "0.2", "0.3"]
    # Changing audio at the same path invalidates both pitch and transcript results.
    sf.write(sources[0], np.full(1600, .4, dtype=np.float32), 16000)
    build(sources, path, model="small")
    assert state["transcriptions"][-1] == "0.4"
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM sources").fetchone()[0] == 3


def test_empty_transcripts_skip_phoneme_model(pipeline):
    sources, path, state = pipeline
    state["empty"] = True
    build(sources, path)
    assert state["asr_loads"] == 1 and state["phoneme_loads"] == 0
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM sources WHERE phonemes_analyzer='wav2vec2'").fetchone()[0] == 3


def test_parallel_postprocessing_failure_resumes_from_committed_file(pipeline):
    sources, path, state = pipeline
    state["fail_alignment"] = "0.2"
    with pytest.raises(RuntimeError, match="interrupted alignment"):
        corpus.build_corpus(sources, path, workers=2, phonemes="wav2vec2", device="cuda")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT path FROM sources WHERE phonemes_analyzer='wav2vec2'").fetchall() == [
            (str(sources[0].resolve()),)]
    state["fail_alignment"] = None
    corpus.build_corpus(sources, path, workers=2, phonemes="wav2vec2", device="cuda")
    assert state["asr_loads"] == 1 and state["phoneme_loads"] == 2
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM sources WHERE phonemes_analyzer='wav2vec2'").fetchone()[0] == 3
        assert conn.execute("SELECT count(*) FROM phoneme_candidates").fetchone()[0] == 9
