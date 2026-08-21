"""FFmpeg/QC media layer (PLAN.md week 4).

Design rules:

- Every ffmpeg/ffprobe command *builder* is a pure function returning ``list[str]``
  argv. No builder touches the filesystem, network, or subprocess.
- The only place that spawns processes is ``app.media.runner.SubprocessRunner``,
  which is injectable so tests run fully offline without the binaries.
- No MoviePy anywhere.
- FFmpeg work is local and zero-cost: nothing in this package performs an
  external cost-bearing call, so no CostEvent rows originate here. Cost-bearing
  calls (Gemini/Veo/TTS) live in their own adapters with the cost ledger.
"""
