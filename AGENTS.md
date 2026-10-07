# Worker context

- Goal: discover recurring costly decision families from games, then compare them across Lichess rapid rating bands. Do not predefine skill categories, named detectors, or expected rating trends. Generate candidates from discovery examples only.
- Use Python 3.12, the locked uv environment, python-chess (`chess`), and local Stockfish. Run `uv run chess-research doctor` before pipeline work.
- Start with a small sample. Preserve source game IDs, moving-player ratings, seeds, engine settings, and score perspective. Keep mate scores distinct from centipawn scores.
- Stream archives; checkpoint batches. Keep raw data, caches, tools, and generated reports out of version control.
- Start with four engine processes, one thread and 128 MiB hash each. Benchmark before increasing concurrency. The CPU MVP does not require GPU access.
- Use the async python-chess engine API (`popen_uci`), as verified by `doctor`; the synchronous wrapper stalled in this sandbox. Set `MPLCONFIGDIR` to `.cache/matplotlib` before importing plotting code.
- Keep discovery, selection, and evaluation separate by moving-player identity; exclude earlier positions from later partitions. Previously inspected players/positions cannot enter a fresh evaluation set. Selection audits assignments; freeze candidates, encoding, and rejection rules before evaluation. Evaluation cannot revise discovery.
- The active proof uses deterministic played-versus-preferred engine trajectories. Exclude ratings, player identities, square names, opening labels, and cost magnitude from clustering. Keep all sampled decisions as measurement denominators.
- Review discovery clusters outside the pipeline; persist evidence IDs, contradictions, rejection reasons, and optional names. Do not add model calls or predefined tactical labels. Names describe observed decisions, not inferred mental causes.
- Report assignment coverage, context concentration, uncertainty, and negative results. Rating associations do not establish learning, teachability, or individual development. Keep the legacy symbolic experiment available for historical comparison only.
