"""ffmpeg handles Japanese filenames and decodes its diagnostics as UTF-8."""

import wave
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from madgen import ffmpeg


def test_probe_japanese_filename(tmp_path):
    source = tmp_path / "テスト音声「サンプル」.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 16000)

    assert ffmpeg.probe(source) == {"duration": 1.0, "has_video": False}


def test_run_reports_japanese_missing_filename(tmp_path):
    source = tmp_path / "存在しない音声.wav"
    with pytest.raises(RuntimeError) as error:
        ffmpeg.run(["-i", str(source), "-f", "null", "-"])
    assert "ffmpeg failed" in str(error.value)
    stderr = str(error.value).split("\n", 1)[1]
    assert "No such file" in stderr
    assert source.name in stderr


@pytest.mark.parametrize("media", ["wav", "mp3", "mkv"])
def test_extract_audio_multi_matches_separate_decodes(tmp_path, media):
    sr = 48000
    t = np.arange(sr + 137) / sr
    stereo = np.column_stack([0.3 * np.sin(2 * np.pi * hz * t) for hz in (220, 330)])
    wav = tmp_path / "ステレオ.wav"
    sf.write(wav, stereo, sr)
    source = wav
    if media == "mp3":
        source = tmp_path / "compressed.mp3"
        ffmpeg.run(["-i", str(wav), "-c:a", "libmp3lame", str(source)])
    elif media == "mkv":
        # Include video and distinct mono/stereo tracks to catch changes in auto selection.
        source = tmp_path / "multitrack.mkv"
        ffmpeg.run([
            "-f", "lavfi", "-i", "color=s=16x16:r=10:d=1",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
            "-i", str(wav), "-map", "0:v", "-map", "1:a", "-map", "2:a",
            "-c:v", "ffv1", "-c:a", "flac", "-disposition:a:0", "0",
            "-disposition:a:1", "0", str(source),
        ])

    outputs = [(tmp_path / "multi" / f"{rate}.wav", rate) for rate in (44100, 16000)]
    expected_info = ffmpeg.probe(source)
    with patch.object(ffmpeg.subprocess, "run", wraps=ffmpeg.subprocess.run) as execute:
        info = ffmpeg.extract_audio_multi(source, outputs)
    assert execute.call_count == 1
    assert info == expected_info
    assert info["has_video"] == (media == "mkv")
    assert info["duration"] == pytest.approx(1.0, abs=.1)
    for combined, rate in outputs:
        separate = tmp_path / f"separate-{rate}.wav"
        ffmpeg.extract_audio(source, separate, rate)
        expected, expected_sr = sf.read(separate, dtype="int16")
        actual, actual_sr = sf.read(combined, dtype="int16")
        assert expected_sr == actual_sr == rate
        assert actual.ndim == 1
        np.testing.assert_array_equal(actual, expected)


def test_input_info_ignores_output_metadata():
    report = """Input #0, wav, from 'input.wav':
  Duration: 00:00:01.25, bitrate: 256 kb/s
  Stream #0:0: Audio: pcm_s16le, 16000 Hz, mono, s16
Stream mapping:
  Stream #0:0 -> #0:0 (pcm_s16le (native) -> pcm_s16le (native))
Output #0, wav, to 'output.wav':
  Metadata:
    Duration: 00:00:42.00
    comment: Stream #0:1: Video: example
  Stream #0:0: Audio: pcm_s16le, 44100 Hz, mono, s16
"""
    assert ffmpeg._input_info(report) == {"duration": 1.25, "has_video": False}


def test_input_info_accepts_unknown_duration():
    report = """Input #0, wav, from 'input.wav':
  Duration: N/A, bitrate: 256 kb/s
  Stream #0:0: Audio: pcm_s16le, 16000 Hz, mono, s16
"""
    assert ffmpeg._input_info(report) == {"duration": 0.0, "has_video": False}
