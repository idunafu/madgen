"""Phase 1: analyze sources into pitched segments and add them to the corpus DB.

Melody mode only needs pitch, so a source is cut into *voiced, pitch-stable* segments.
Silence handling on the source side happens here: frames that are quiet (silence) or
unvoiced (breath, noise, consonant hiss) never become part of a segment, so they can
never be picked as material.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, closing, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf

from . import db, ffmpeg
from .progress import progress

ANALYSIS_SR = 16000
SYNTH_SR = 44100
FRAME_PERIOD_MS = 5.0
CHUNK_SEC = 120.0

MEDIA_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac",
              ".mp4", ".mkv", ".webm", ".mov", ".avi"}


@dataclass
class SegmentParams:
    silence_db: float = -45.0      # absolute floor (dBFS): quieter frames are silence
    silence_rel_db: float = 35.0   # also silence if this far below the source's loud level
    min_sec: float = 0.06          # shorter voiced pieces are dropped
    max_sec: float = 1.5           # long notes are split so they stay reusable
    pitch_tol_semitones: float = 1.0  # a segment ends when the pitch drifts this far


def file_hash(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def iter_media(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(q for q in p.rglob("*") if q.suffix.lower() in MEDIA_EXTS))
        elif p.exists():
            files.append(p)
        else:
            raise SystemExit(f"source not found: {p}")
    return files


def _frame_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """f0 (Hz, 0 = unvoiced) and rms (dBFS) per 5 ms frame."""
    x64 = x.astype(np.float64)
    f0, t = pyworld.dio(x64, ANALYSIS_SR, frame_period=FRAME_PERIOD_MS, f0_floor=60.0, f0_ceil=1100.0)
    f0 = pyworld.stonemask(x64, f0, t, ANALYSIS_SR)
    hop = int(ANALYSIS_SR * FRAME_PERIOD_MS / 1000)
    win = hop * 4
    padded = np.pad(x64, (win // 2, win))
    n = len(f0)
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    rms = np.sqrt(np.mean(padded[idx] ** 2, axis=1) + 1e-12)
    return f0, 20 * np.log10(rms)


def _analyze_chunk(args: tuple[str, int, int]) -> tuple[np.ndarray, np.ndarray]:
    path, start, frames = args
    x, _ = sf.read(path, start=start, frames=frames, dtype="float32")
    return _frame_features(x)


def analyze_frames(wav16: Path, workers: int, *, report_progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Frame features for a whole (possibly hours long) file, chunked across processes."""
    total = sf.info(str(wav16)).frames
    chunk = int(CHUNK_SEC * ANALYSIS_SR)
    hop = int(ANALYSIS_SR * FRAME_PERIOD_MS / 1000)
    jobs = [(str(wav16), s, min(chunk, total - s)) for s in range(0, total, chunk)]
    f0s, rmss = [], []
    if report_progress:
        progress.stage("pitch analysis (chunks)", len(jobs))
    # A single chunk has no parallel work; spawning a process per short file is costly.
    serial = len(jobs) <= 1 or workers == 1
    with nullcontext() if serial else ProcessPoolExecutor(max_workers=workers) as pool:
        results = map(_analyze_chunk, jobs) if serial else pool.map(_analyze_chunk, jobs)
        for i, (f0, rms) in enumerate(results):
            # Each chunk yields frames at 0, hop, ...; keep exactly the frames that belong to it.
            keep = -(-jobs[i][2] // hop)
            f0s.append(f0[:keep])
            rmss.append(rms[:keep])
            if report_progress:
                progress.update(i + 1)
                print(f"\r  analyzed {min((i + 1) * CHUNK_SEC, total / ANALYSIS_SR):.0f}"
                      f"/{total / ANALYSIS_SR:.0f} s", end="", file=sys.stderr, flush=True)
    if report_progress:
        print(file=sys.stderr)
    return np.concatenate(f0s), np.concatenate(rmss)


def segment_frames(f0: np.ndarray, rms_db: np.ndarray, p: SegmentParams) -> list[tuple[int, int]]:
    """Cut frame features into [start, end) frame ranges of voiced, audible, pitch-stable sound."""
    loud_level = np.percentile(rms_db, 95)
    threshold = max(p.silence_db, loud_level - p.silence_rel_db)
    usable = (f0 > 0) & (rms_db > threshold)

    frame_sec = FRAME_PERIOD_MS / 1000
    min_len = int(round(p.min_sec / frame_sec))
    max_len = int(round(p.max_sec / frame_sec))
    semis = np.zeros_like(f0)
    semis[f0 > 0] = 12 * np.log2(f0[f0 > 0] / 440.0)

    # Runs of usable frames.
    edges = np.diff(np.concatenate([[0], usable.astype(np.int8), [0]]))
    run_starts = np.nonzero(edges == 1)[0]
    run_ends = np.nonzero(edges == -1)[0]

    segments: list[tuple[int, int]] = []
    for rs, re_ in zip(run_starts, run_ends, strict=True):
        if re_ - rs < min_len:
            continue
        s = rs
        total = semis[rs]
        count = 1
        for i in range(rs + 1, re_):
            mean = total / count
            if abs(semis[i] - mean) > p.pitch_tol_semitones or i - s >= max_len:
                if i - s >= min_len:
                    segments.append((s, i))
                s, total, count = i, semis[i], 1
            else:
                total += semis[i]
                count += 1
        if re_ - s >= min_len:
            segments.append((s, re_))
    return segments


PHONEME_ANALYZERS = ("wav2vec2",)


def _add_phoneme_segments(conn, digest: str, path: Path, video_ref: str | None, segs) -> None:
    progress.stage("writing phoneme segments to DB")
    conn.execute("DELETE FROM phoneme_candidates WHERE segment_id IN "
                 "(SELECT id FROM segments WHERE source_id = ? AND kind = 'phoneme')", (digest,))
    conn.execute("DELETE FROM segments WHERE source_id = ? AND kind = 'phoneme'", (digest,))
    for seg in segs:
        cur = conn.execute(
            "INSERT INTO segments (source_id, start_sec, end_sec, phoneme, f0_hz, video_ref, f0_std_cents, rms_db, "
            "kind) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'phoneme')",
            (digest, seg.start_sec, seg.end_sec, seg.phoneme, seg.f0_hz, video_ref, seg.f0_std_cents, seg.rms_db),
        )
        conn.executemany(
            "INSERT INTO phoneme_candidates (segment_id, rank, phoneme, confidence) VALUES (?, ?, ?, ?)",
            [(cur.lastrowid, rank, ph, conf) for rank, (ph, conf) in enumerate(seg.candidates, 1)],
        )
    conn.execute("UPDATE sources SET phonemes_analyzer = ? WHERE source_id = ?", ("wav2vec2", digest))
    conn.commit()
    print(f"  {len(segs)} phoneme segments", file=sys.stderr)
    progress.log(f"{path}: {len(segs)} phoneme segments")


@dataclass
class _PhonemeJob:
    digest: str
    path: Path
    wav16: Path
    video_ref: str | None


def _read_transcript(path: Path, settings: dict) -> list[dict] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data["utterances"] if data.get("settings") == settings else None


def _analyze_phonemes(conn, jobs: list[_PhonemeJob], device: str, whisper_model: str) -> None:
    """Two passes: release WhisperX/VAD before loading wav2vec2, once per build."""
    if not jobs:
        return
    from .phoneme_analysis import (
        PhonemeModel,
        analyze_utterances,
        load_transcriber,
        release_models,
        transcribe_with_model,
        unload_transcriber,
    )

    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = {"version": 1, "whisper_model": whisper_model, "device": device, "language": "ja"}
    transcriber = None
    try:
        for i, job in enumerate(jobs, 1):
            # The filename is keyed by source content; settings invalidate incompatible transcripts.
            transcript = job.wav16.with_suffix(".transcript.json")
            if _read_transcript(transcript, settings) is not None:
                progress.log(f"transcription pass {i}/{len(jobs)}: cached {job.path}")
                continue
            progress.log(f"transcription pass {i}/{len(jobs)}: {job.path}")
            if transcriber is None:
                transcriber = load_transcriber(device, whisper_model)
            audio, _ = sf.read(str(job.wav16), dtype="float32")
            utterances = transcribe_with_model(audio, transcriber)
            # Atomic replacement keeps completed files reusable after an interruption.
            temporary = transcript.with_suffix(".tmp")
            temporary.write_text(json.dumps({"settings": settings, "utterances": utterances},
                                             ensure_ascii=False), encoding="utf-8")
            temporary.replace(transcript)
            del audio
    finally:
        if transcriber is not None:
            unload_transcriber(transcriber)
        transcriber = None
        release_models(device)

    progress.log("transcription pass complete; WhisperX and VAD released")
    model = None
    try:
        for i, job in enumerate(jobs, 1):
            progress.log(f"phoneme pass {i}/{len(jobs)}: {job.path}")
            utterances = _read_transcript(job.wav16.with_suffix(".transcript.json"), settings)
            if utterances is None:
                raise RuntimeError(f"transcript missing or incompatible: {job.path}")
            segs = []
            if utterances:
                if model is None:
                    progress.stage("loading phoneme model")
                    model = PhonemeModel(device)
                audio, _ = sf.read(str(job.wav16), dtype="float32")
                segs = analyze_utterances(audio, utterances, model)
                del audio
            _add_phoneme_segments(conn, job.digest, job.path, job.video_ref, segs)
            job.wav16.unlink()
    finally:
        model = None
        release_models(device)


def build_corpus(sources: list[Path], db_path: Path, workers: int | None = None,
                 params: SegmentParams | None = None, phonemes: str = "none",
                 device: str = "auto", whisper_model: str = "large-v3") -> None:
    if phonemes != "none" and phonemes not in PHONEME_ANALYZERS:
        raise SystemExit(f"unknown phoneme analyzer {phonemes!r}; choose from {PHONEME_ANALYZERS}")
    params = params or SegmentParams()
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    cache_dir = db_path.with_suffix(db_path.suffix + ".cache")
    with closing(db.connect(db_path)) as conn:
        _build_sources(conn, sources, cache_dir, workers, params, phonemes, device, whisper_model)


@dataclass
class _SourceJob:
    digest: str
    path: Path
    wav44: Path
    wav16: Path
    # None means pitch analysis is needed; a bool comes from an existing DB row.
    has_video: bool | None = None


def _prepare_source(job: _SourceJob, params: SegmentParams, workers: int = 1,
                    *, report_progress: bool = False) -> tuple[bool, list[tuple] | None, float]:
    """Decode and analyze without touching the DB or loading any phoneme models."""
    if job.has_video is not None:
        ffmpeg.extract_audio(job.path, job.wav16, ANALYSIS_SR)
        return job.has_video, None, 0.0

    info = ffmpeg.extract_audio_multi(job.path, [(job.wav44, SYNTH_SR), (job.wav16, ANALYSIS_SR)])
    f0, rms_db = analyze_frames(job.wav16, workers, report_progress=report_progress)
    segs = segment_frames(f0, rms_db, params)
    frame_sec = FRAME_PERIOD_MS / 1000
    video_ref = str(job.path.resolve()) if info["has_video"] else None
    records = []
    for s, e in segs:
        voiced = f0[s:e][f0[s:e] > 0]
        cents = 1200 * np.log2(voiced / np.median(voiced))
        records.append((
            job.digest, s * frame_sec, e * frame_sec, "", float(np.median(voiced)), video_ref,
            float(np.std(cents)), float(np.mean(rms_db[s:e])),
        ))
    return info["has_video"], records, sum(e - s for s, e in segs) * frame_sec


def _save_source(conn, job: _SourceJob, result: tuple[bool, list[tuple] | None, float],
                 pending: dict[str, _PhonemeJob], phonemes: str) -> None:
    has_video, records, usable_sec = result
    digest, path = job.digest, job.path
    if records is not None:
        # Apply replacements only when this source's turn to commit arrives.
        for (old,) in conn.execute("SELECT source_id FROM sources WHERE path = ?", (str(path.resolve()),)).fetchall():
            conn.execute("DELETE FROM phoneme_candidates WHERE segment_id IN "
                         "(SELECT id FROM segments WHERE source_id = ?)", (old,))
            conn.execute("DELETE FROM segments WHERE source_id = ?", (old,))
            conn.execute("DELETE FROM sources WHERE source_id = ?", (old,))
        conn.executemany(
            "INSERT INTO segments (source_id, start_sec, end_sec, phoneme, f0_hz, video_ref, f0_std_cents, rms_db) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            records,
        )
        conn.execute(
            "INSERT INTO sources (source_id, file_hash, path, audio_cache, has_video) VALUES (?, ?, ?, ?, ?)",
            (digest, digest, str(path.resolve()), str(job.wav44.resolve()), int(has_video)),
        )
        conn.commit()
        print(f"  {len(records)} segments, {usable_sec:.0f} s of usable voiced sound", file=sys.stderr)
        progress.log(f"{path}: {len(records)} pitch segments")
    if phonemes != "none":
        pending[digest] = _PhonemeJob(digest, path, job.wav16, str(path.resolve()) if has_video else None)
    else:
        job.wav16.unlink()


def _build_sources(conn, sources: list[Path], cache_dir: Path, workers: int, params: SegmentParams,
                   phonemes: str, device: str, whisper_model: str) -> None:
    files = iter_media(sources)
    parallel = workers > 1 and len(files) > 1
    limit = min(workers, len(files))
    pending: dict[str, _PhonemeJob] = {}
    scheduled: set[str] = set()
    replaced: set[str] = set()
    queue = deque()

    def save_next() -> None:
        job, future = queue.popleft()
        progress.stage(f"waiting for source: {job.path}")
        result = (_prepare_source(job, params, workers, report_progress=True)
                  if future is None else future.result())
        _save_source(conn, job, result, pending, phonemes)

    # Bound the queue so large corpora do not retain every decoded file/result in advance.
    # Shut down CPU workers before the two GPU/model passes begin.
    with ExitStack() as stack:
        pool = None
        for path in files:
            print(f"[build-corpus] {path}", file=sys.stderr)
            progress.log(f"source {path}")
            progress.stage("hashing source")
            digest = file_hash(path)
            if digest in scheduled:
                continue
            # Earlier queued replacements have not committed yet. Treat their old content
            # as absent now, just as a serial build would, even if it appears at another path.
            row = None if digest in replaced else conn.execute(
                "SELECT has_video, phonemes_analyzer FROM sources WHERE source_id = ?", (digest,)).fetchone()
            if row and (phonemes == "none" or row[1] == phonemes):
                print("  unchanged, skipping (cached)", file=sys.stderr)
                continue
            job = _SourceJob(digest, path, cache_dir / f"{digest}.wav", cache_dir / f"{digest}.16k.wav",
                             bool(row[0]) if row else None)
            if row is None:
                replaced.update(old for (old,) in conn.execute(
                    "SELECT source_id FROM sources WHERE path = ?", (str(path.resolve()),)))
            scheduled.add(digest)
            if parallel:
                if pool is None:
                    if not queue:
                        # Cache hits/duplicates may leave only one real job. Avoid starting
                        # file workers until a second job exists; keep chunk parallelism then.
                        queue.append((job, None))
                        continue
                    pool = ProcessPoolExecutor(max_workers=limit)
                    stack.callback(pool.shutdown, wait=True, cancel_futures=True)
                    first, _ = queue.popleft()
                    queue.append((first, pool.submit(_prepare_source, first, params)))
                # Each worker keeps the same chunk boundaries but does not spawn a nested pool.
                queue.append((job, pool.submit(_prepare_source, job, params)))
                if len(queue) >= limit:
                    save_next()
            else:
                progress.stage(f"decoding and analyzing source: {path}")
                _save_source(conn, job, _prepare_source(job, params, workers, report_progress=True), pending, phonemes)
        while queue:
            save_next()
    _analyze_phonemes(conn, list(pending.values()), device, whisper_model)
