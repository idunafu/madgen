"""Progress reflects file-local ASR and restores stages around overlapped DB work."""

from types import SimpleNamespace

import numpy as np
import pytest

from madgen import phoneme_analysis as pa
from madgen.progress import Progress


def test_transcription_progress_identifies_files_and_records_short_jobs(tmp_path, monkeypatch, capsys):
    progress = Progress()
    monkeypatch.setattr(pa, "progress", progress)
    path = tmp_path / "progress.log"
    progress.open(path)
    labels = ["WhisperX 1/3: first.wav", "WhisperX 2/3: second.wav", "WhisperX 3/3: silent.wav"]
    speech = {"start": 0., "end": 1., "text": "a"}

    try:
        for index, label in enumerate(labels):
            def transcribe(audio, index=index, label=label, **kwargs):
                assert kwargs["print_progress"] is False
                assert kwargs["batch_size"] == 8 and kwargs["language"] == "ja"
                # A new file must not inherit the preceding file's 100% while VAD runs.
                assert label in progress._status()
                assert "voice detection" in progress._status() and "%" not in progress._status()
                if index == 0:
                    kwargs["progress_callback"](50.)
                if index < 2:
                    kwargs["progress_callback"](100.)
                    return {"segments": [speech, {"text": " "}]}
                return {"segments": []}

            result = pa.transcribe_with_model(np.zeros(1600), SimpleNamespace(transcribe=transcribe), label=label)
            assert result == ([speech] if index < 2 else [])
    finally:
        progress.close()

    terminal = capsys.readouterr().err
    log = path.read_text(encoding="utf-8")
    for label in labels[:2]:
        for output in (terminal, log):
            assert f"{label}: transcription (file %): 100/100 (100.0%)" in output
    assert "WhisperX 1/3: first.wav: transcription (file %): 50/100 (50.0%)" in log
    assert "WhisperX 3/3: silent.wav: complete (0 utterances)" in log
    assert "WhisperX 3/3: silent.wav: transcription (file %)" not in log


def test_report_throttles_updates_but_always_records_completion(tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr("madgen.progress.time.monotonic", lambda: now[0])
    progress = Progress()
    path = tmp_path / "progress.log"
    progress.open(path)
    try:
        progress.stage("transcription", 100)
        progress.report(10)
        progress.report(20)
        now[0] += 5
        progress.report(50)
        progress.report(100)
        progress.report(100)
    finally:
        progress.close()
    updates = [line for line in path.read_text(encoding="utf-8").splitlines() if "progress |" in line]
    assert len(updates) == 3
    assert "10/100" in updates[0] and "50/100" in updates[1] and "100/100" in updates[2]


def test_temporary_db_and_wait_phases_restore_inference_and_close_heartbeat(tmp_path):
    progress = Progress()
    path = tmp_path / "progress.log"
    progress.open(path)
    thread = progress._thread
    try:
        progress.stage("phoneme source 2: inference (utterances)", 10)
        progress.report(3)
        started = progress._stage_t0
        with progress.phase("writing phoneme source 1 to DB"):
            assert "writing phoneme source 1 to DB" in progress._status()
        with pytest.raises(RuntimeError), progress.phase("waiting for CPU postprocessing"):
            assert "waiting for CPU postprocessing" in progress._status()
            raise RuntimeError("worker failed")
        progress.report(4)
        assert "phoneme source 2: inference (utterances): 4/10 (40.0%)" in progress._status()
        assert progress._stage_t0 == started
    finally:
        progress.close()
    assert not thread.is_alive()
    progress.open(tmp_path / "second.log")
    try:
        assert progress._stage == "starting" and progress._done == 0 and progress._total is None
    finally:
        progress.close()
