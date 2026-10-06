# Worker context

- Goal: discover recurring costly decision patterns, then compare them across Lichess rapid rating bands. Do not use named skill detectors or tactical labels as discovery inputs.
- Use Python 3.12, the locked uv environment, python-chess (`chess`), and local Stockfish. Run `uv run chess-research doctor` before pipeline work.
- Start with a small sample. Preserve source game IDs, moving-player ratings, seeds, engine settings, and score perspective. Keep mate scores distinct from centipawn scores.
- Stream archives; checkpoint batches. Keep raw data, caches, tools, and generated reports out of version control.
- Start with four engine processes, one thread and 128 MiB hash each. Benchmark before increasing concurrency. The CPU MVP does not require GPU access.
- Use the async python-chess engine API (`popen_uci`), as verified by `doctor`; the synchronous wrapper stalled in this sandbox. Set `MPLCONFIGDIR` to `.cache/matplotlib` before importing plotting code.
- Keep discovery, selection, and evaluation separate by moving-player identity; exclude repeated positions from held-out evaluation. LLM naming comes after pattern selection.
