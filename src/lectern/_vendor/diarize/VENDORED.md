> Modified for Lectern, 2026-10-13: record the ONNX adaptation.
> Original: diarize 0.1.2 (Apache-2.0).

# Vendored: diarize 0.1.2

- Upstream: https://pypi.org/project/diarize/0.1.2/ (repository: https://github.com/FoxNoseTech/diarize)
- Wheel sha256 (first 16 hex): 848b798e471e452e
- Licence: Apache License 2.0, see `LICENSE` in this directory. These files stay under Apache-2.0;
  the rest of Lectern is MIT.
- The first commit adds the files unmodified. Later commits adapt them for Lectern: voice
  detection and embeddings run on ONNX without PyTorch, and the imports are made local.
  As Apache-2.0 §4(b) requires, every modified file starts with a prominent notice saying it
  was changed, by Lectern, and how. The full diff against the unmodified commit is in git
  history.

## Lectern adaptation (2026-10-13)

- `__init__.py`: exposes `diarize_embeddings(speech_segments, embeddings,
  subsegments, audio_duration, num_speakers)` returning upstream segments;
  removes file loading, VAD and embedding calls. Clustering defaults (1–20),
  random seeds, temporal smoothing and segment building are unchanged.
- `utils.py`: removes soundfile/torchaudio duration loading; retains Pydantic
  data models and serialization helpers.
- `embeddings.py`: retains only the upstream 0.4 s minimum duration, 1.2 s
  window and 0.6 s step constants.
- `vad.py`: retains only upstream defaults: threshold 0.45, minimum speech
  200 ms, minimum silence 50 ms and padding 20 ms.
- `clustering.py` and `LICENSE`: unchanged. Imports within the retained
  pipeline were already relative and remain so.

Lectern's glue lazily uses sherpa-onnx Silero VAD, ONNX Runtime WeSpeaker
embeddings and kaldi-native-fbank features with mean normalization. Vendored
code is excluded from Lectern's Ruff and strict Pyright checks; the glue
remains strict. No pretrained model is included here.
