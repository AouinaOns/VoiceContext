import argparse
import os
import tempfile
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

SAMPLE_RATE = 16000
MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"

CHUNK_MS_TO_ATT_CTX = {
    80:   [70, 0],
    160:  [70, 1],
    560:  [70, 6],
    1120: [70, 13],
}


def audio_to_wav(audio: np.ndarray) -> str:
    """Write a numpy float32 array to a temp wav file; return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, SAMPLE_RATE)
    tmp.close()
    return tmp.name


def load_audio_file(path: str) -> np.ndarray:
    """Load wav/mp3/ogg/opus to mono float32 at 16 kHz.
    librosa uses CoreAudio on macOS — handles MP3 without ffmpeg."""
    import librosa
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    return audio


def make_synthetic_audio(duration_s: float = 3.0) -> np.ndarray:
    """3-second 440 Hz sine — the model will return empty text (expected)."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def run_streaming_test(wav_path: str, model, device: torch.device) -> None:
    print(f"\nAudio : {wav_path}")
    print(f"Device: {device}")
    print("─" * 60)

    # Cache is owned by the caller — initialised from the encoder
    cache_last_channel, cache_last_time, cache_last_channel_len = (
        model.encoder.get_initial_cache_state(batch_size=1)
    )
    previous_hypotheses = None

    streaming_buffer = CacheAwareStreamingAudioBuffer(model=model, online_normalization=True)
    streaming_buffer.append_audio_file(wav_path, stream_id=-1)

    chunk_times_ms: list[float] = []
    prev_text = ""
    final_text = ""
    print("Streaming: ", end="", flush=True)

    for chunk_audio, chunk_lengths in streaming_buffer:
        chunk_audio = chunk_audio.to(device)
        chunk_lengths = chunk_lengths.to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            (
                _pred_out_stream,
                step_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
            ) = model.conformer_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=streaming_buffer.is_buffer_empty(),
                previous_hypotheses=previous_hypotheses,
                drop_extra_pre_encoded=0,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        chunk_times_ms.append(elapsed_ms)

        hyp = step_texts[0] if step_texts else None
        text = hyp.text if (hyp is not None and hyp.text) else ""

        if text and text != prev_text:
            new_fragment = text[len(prev_text):]
            if new_fragment:
                print(new_fragment, end="", flush=True)
            prev_text = text
            final_text = text

    print()  # newline after streaming output

    if not chunk_times_ms:
        print("  No chunks produced — check audio length vs. chunk size.")
        return

    print("\n" + "─" * 60)
    print(f"Final transcript : {final_text!r}")
    print(f"Chunks           : {len(chunk_times_ms)}")
    print(f"Avg latency      : {np.mean(chunk_times_ms):.1f}ms "
          f"(p95={np.percentile(chunk_times_ms, 95):.1f}ms, max={max(chunk_times_ms):.1f}ms)")
    chunk_ms = int(1000 * (
        model.encoder.att_context_size[1] + 1
    ) * model.encoder.subsampling_factor * model.preprocessor.hop_length / SAMPLE_RATE)
    rtf = np.mean(chunk_times_ms) / max(chunk_ms, 1)
    print(f"Real-time factor : {rtf:.3f}x  ({'below' if rtf < 1.0 else 'above'} real-time)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file",     type=str,   default=None, help="Path to a .wav file")
    p.add_argument("--chunk-ms", type=int,   default=160,  choices=[80, 160, 560, 1120])
    p.add_argument("--duration", type=float, default=3.0,  help="Synthetic tone length (seconds)")
    args = p.parse_args()

    print(f"Loading {MODEL_NAME} ...")
    t0 = time.perf_counter()

    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)

    # Apply chunk size before moving to device
    att_ctx = CHUNK_MS_TO_ATT_CTX[args.chunk_ms]
    model.encoder.set_default_att_context_size(att_ctx)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    load_s = time.perf_counter() - t0
    print(f"Model loaded in {load_s:.1f}s  |  chunk={args.chunk_ms}ms  att_context_size={att_ctx}  device={device}")

    tmp_path = None
    if args.file:
        # Always convert to a clean 16 kHz mono WAV — NeMo can't handle MP3 directly
        print(f"Loading {args.file} ...")
        audio = load_audio_file(args.file)
        wav_path = audio_to_wav(audio)
        tmp_path = wav_path
        print(f"  → {len(audio)/SAMPLE_RATE:.2f}s  mono  16 kHz  (temp WAV: {wav_path})")
    else:
        print(f"Generating {args.duration}s synthetic tone (440 Hz) ...")
        audio = make_synthetic_audio(args.duration)
        wav_path = audio_to_wav(audio)
        tmp_path = wav_path

    # ── Stream ──────────────────────────────────────────────────────────────
    try:
        run_streaming_test(wav_path, model, device)
    finally:
        if tmp_path:
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
