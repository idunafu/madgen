"""Serial pitch analysis retains the same frames as chunked worker execution."""

from concurrent.futures import ProcessPoolExecutor
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from madgen import corpus


@pytest.mark.parametrize("frames", [16000, 16037])
def test_single_chunk_avoids_process_pool(tmp_path, monkeypatch, frames):
    wav = tmp_path / "short.wav"
    t = np.arange(frames) / corpus.ANALYSIS_SR
    sf.write(wav, 0.3 * np.sin(2 * np.pi * 220 * t), corpus.ANALYSIS_SR)
    # Reference: the pre-optimization path always submitted the chunk to a worker.
    with ProcessPoolExecutor(max_workers=1) as pool:
        expected = next(pool.map(corpus._analyze_chunk, [(str(wav), 0, frames)]))
    pool_factory = Mock(side_effect=AssertionError("short files must run in process"))
    monkeypatch.setattr(corpus, "ProcessPoolExecutor", pool_factory)

    actual = corpus.analyze_frames(wav, workers=4)
    keep = -(-frames // 80)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference[:keep])
    pool_factory.assert_not_called()


def test_multiple_chunks_match_with_one_or_many_workers(tmp_path, monkeypatch):
    # Shorten the test chunks; production still uses the original 120-second boundary.
    monkeypatch.setattr(corpus, "CHUNK_SEC", 1.0)
    wav = tmp_path / "chunks.wav"
    t = np.arange(32037) / corpus.ANALYSIS_SR
    sf.write(wav, 0.3 * np.sin(2 * np.pi * (220 * t + 30 * t**2)), corpus.ANALYSIS_SR)
    pool_factory = Mock(wraps=ProcessPoolExecutor)
    monkeypatch.setattr(corpus, "ProcessPoolExecutor", pool_factory)
    parallel = corpus.analyze_frames(wav, workers=2)
    pool_factory.assert_called_once_with(max_workers=2)
    pool_factory.reset_mock()
    serial = corpus.analyze_frames(wav, workers=1)
    pool_factory.assert_not_called()
    for result, reference in zip(serial, parallel, strict=True):
        assert len(result) == 401
        np.testing.assert_array_equal(result, reference)
