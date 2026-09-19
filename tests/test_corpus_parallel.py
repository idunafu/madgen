"""Parallel builds preserve source order, cache semantics, and resumability."""

import hashlib
import shutil
import sqlite3
from contextlib import closing
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from madgen import corpus, db


def _tone(path, hz, seconds=1):
    t = np.arange(int(16000 * seconds) + 37) / 16000
    sf.write(path, .3 * np.sin(2 * np.pi * hz * t), 16000)


def _snapshot(path):
    with closing(sqlite3.connect(path)) as conn:
        segments = conn.execute("SELECT * FROM segments ORDER BY id").fetchall()
        sources = conn.execute(
            "SELECT source_id,file_hash,path,has_video,phonemes_analyzer FROM sources ORDER BY rowid").fetchall()
    audio = {}
    for wav in path.with_suffix(".sqlite.cache").glob("*.wav"):
        pcm, sr = sf.read(wav, dtype="int16")
        audio[wav.name] = (sr, hashlib.sha256(pcm.tobytes()).hexdigest())
    return segments, sources, audio


def test_parallel_matches_serial_and_replaces_cached_content(tmp_path, monkeypatch):
    paths = [tmp_path / f"{i}.wav" for i in range(3)]
    for i, path in enumerate(paths):
        _tone(path, 220 + i * 110, seconds=3 if i == 0 else 1)
    duplicate = tmp_path / "duplicate.wav"
    shutil.copyfile(paths[0], duplicate)
    sources = [paths[0], duplicate, paths[1], paths[2], paths[0]]
    serial, parallel = tmp_path / "serial.sqlite", tmp_path / "parallel.sqlite"
    corpus.build_corpus(sources, serial, workers=1)
    corpus.build_corpus(sources, parallel, workers=2)
    assert _snapshot(serial) == _snapshot(parallel)
    assert len(_snapshot(parallel)[1]) == 3

    # A changed path removes its old hash. A later copy must be re-added, even while
    # the replacement is queued and the old DB row is still visible to the parent.
    _tone(paths[0], 660)
    sources = [paths[0], paths[1], duplicate, paths[2]]
    corpus.build_corpus(sources, serial, workers=1)
    corpus.build_corpus(sources, parallel, workers=2)
    assert _snapshot(serial) == _snapshot(parallel)
    assert len(_snapshot(parallel)[1]) == 4

    before = _snapshot(parallel)
    pool = Mock(side_effect=AssertionError("cached builds must not start workers"))
    monkeypatch.setattr(corpus, "ProcessPoolExecutor", pool)
    corpus.build_corpus(sources, parallel, workers=2)
    assert _snapshot(parallel) == before
    pool.assert_not_called()
    # A single changed short source among cached inputs should also avoid process startup.
    _tone(paths[1], 880)
    corpus.build_corpus(sources, parallel, workers=2)
    corpus.build_corpus(sources, serial, workers=1)
    assert _snapshot(serial) == _snapshot(parallel)
    pool.assert_not_called()


def test_parallel_failure_commits_prefix_and_can_resume(tmp_path):
    paths = [tmp_path / f"{i}.wav" for i in range(3)]
    _tone(paths[0], 220)
    paths[1].write_bytes(b"invalid audio")
    _tone(paths[2], 440)
    parallel = tmp_path / "parallel.sqlite"
    with pytest.raises((RuntimeError, ValueError)):
        corpus.build_corpus(paths, parallel, workers=2)
    with closing(sqlite3.connect(parallel)) as conn:
        assert conn.execute("SELECT path FROM sources").fetchall() == [(str(paths[0].resolve()),)]

    _tone(paths[1], 330)
    corpus.build_corpus(paths, parallel, workers=2)
    serial = tmp_path / "serial.sqlite"
    corpus.build_corpus(paths, serial, workers=1)
    assert _snapshot(serial) == _snapshot(parallel)


def test_parallel_preparation_passes_ordered_unique_phoneme_jobs(tmp_path, monkeypatch):
    paths = [tmp_path / f"{i}.wav" for i in range(3)]
    for i, path in enumerate(paths):
        _tone(path, 220 + i * 110)
    database = tmp_path / "corpus.sqlite"
    corpus.build_corpus([paths[0]], database, workers=1)
    batches = []

    def phonemes(conn, jobs, device, model, workers):
        batches.append(jobs)
        for job in jobs:
            assert job.wav16.is_file()
            assert conn.execute("SELECT 1 FROM sources WHERE source_id=?", (job.digest,)).fetchone()

    monkeypatch.setattr(corpus, "_analyze_phonemes", phonemes)
    corpus.build_corpus([paths[0], paths[1], paths[0], paths[2]], database,
                        workers=2, phonemes="wav2vec2")
    assert len(batches) == 1
    assert [job.path for job in batches[0]] == paths


def test_path_index_is_added_to_existing_databases(tmp_path):
    database = tmp_path / "existing.sqlite"
    with closing(sqlite3.connect(database)) as conn:
        conn.executescript(db.SCHEMA)
        conn.execute("INSERT INTO sources (source_id,file_hash,path,audio_cache,has_video) VALUES (?,?,?,?,?)",
                     ("hash", "hash", "source.wav", "cache.wav", 0))
        conn.commit()
    for _ in range(2):
        with closing(db.connect(database)) as conn:
            plan = conn.execute("EXPLAIN QUERY PLAN SELECT source_id FROM sources WHERE path=?",
                                ("source.wav",)).fetchall()
            assert any("USING INDEX sources_path" in row[3] for row in plan)
            assert conn.execute("SELECT source_id FROM sources WHERE path=?", ("source.wav",)).fetchall() == [("hash",)]
