"""CPU/GPU scheduling and file-level filtering without loading ML dependencies."""

import threading
import weakref
from contextlib import closing

import numpy as np
import pytest

from madgen import phoneme_analysis as pa


class Model:
    def __init__(self, device):
        pass

    def posteriors(self, audio):
        assert threading.current_thread() is threading.main_thread()
        probs = np.full((len(audio) // 320, len(pa.INVENTORY)), .01, dtype=np.float32)
        probs[:, pa.INVENTORY.index("a")] = .8
        return probs

    def align(self, probs, phonemes):
        return None if phonemes == ["b"] else [(1, len(probs) - 1, .9)]

    def phoneme_mass(self, probs):
        return probs


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr(pa, "PhonemeModel", Model)
    monkeypatch.setattr(pa, "_g2p", str.split)
    monkeypatch.setattr(pa, "release_models", lambda device: None)


def test_parallel_matches_serial_and_filters_per_file(models):
    tone = np.sin(2 * np.pi * 220 * np.arange(pa.SR) / pa.SR).astype(np.float32)
    sources = [
        (np.concatenate([.9 * tone, np.zeros(pa.SR, dtype=np.float32), .01 * tone]),
         [{"start": 0., "end": 1., "text": "a"}, {"start": 2., "end": 3., "text": "a"}]),
        (.01 * tone, [{"start": 0., "end": 1., "text": "a"},
                     {"start": 0., "end": 1., "text": "b"},
                     {"start": 0., "end": 1., "text": ""}]),
        (tone, []),
        (tone[:100], [{"start": 0., "end": .005, "text": "a"}]),
    ]
    serial = list(pa.analyze_sources(sources, "cuda", workers=1))
    parallel = list(pa.analyze_sources(sources, "cuda", workers=3))
    assert parallel == serial
    assert len(serial[0]) == len(serial[1]) == 1
    assert serial[0][0].start_sec < 1  # Quiet utterance dropped next to loud one.
    assert serial[1][0].rms_db < -40  # Same quiet audio kept in its own file.
    assert serial[2:] == [[], []]


def test_inference_overlaps_cpu_work_across_files_and_saves_in_order(models, monkeypatch):
    first_started, second_finished = threading.Event(), threading.Event()
    inferred = []

    def infer(self, audio):
        number = len(inferred) + 1
        if number == 2:
            assert first_started.wait(5)
            assert not second_finished.is_set()
        inferred.append(number)
        return np.array([[number]])

    def postprocess(chunk, a0, phonemes, probs, model):
        assert threading.current_thread() is not threading.main_thread()
        number = int(probs[0, 0])
        if number == 1:
            first_started.set()
            assert second_finished.wait(5)
        else:
            second_finished.set()
        seg = pa.PhonemeSegment(number, number + .1, "k", [("k", 1.)], None, 0., -20.)
        return [(seg, "k", 0.)], 0

    monkeypatch.setattr(Model, "posteriors", infer)
    monkeypatch.setattr(pa, "_postprocess_utterance", postprocess)
    source = (np.zeros(pa.SR, dtype=np.float32), [{"start": 0., "end": 1., "text": "a"}])
    results = list(pa.analyze_sources([source, source], "cuda", workers=2))
    assert [segs[0].start_sec for segs in results] == [1, 2]


def test_closing_results_stops_workers_and_releases_shared_model(models, monkeypatch):
    refs, releases = [], []

    def load(device):
        model = Model(device)
        refs.append(weakref.ref(model))
        return model

    monkeypatch.setattr(pa, "PhonemeModel", load)
    monkeypatch.setattr(pa, "release_models", releases.append)
    audio = np.sin(2 * np.pi * 220 * np.arange(pa.SR) / pa.SR).astype(np.float32)
    source = (audio, [{"start": 0., "end": 1., "text": "a"}])
    with pytest.raises(RuntimeError, match="save failed"), closing(
        pa.analyze_sources([source] * 20, "cuda", workers=2)
    ) as results:
        assert next(results)
        raise RuntimeError("save failed")
    assert releases == ["cuda"]
    assert len(refs) == 1 and refs[0]() is None
    assert not any(thread.name.startswith("phoneme-cpu") for thread in threading.enumerate())
