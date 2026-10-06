# Chess research

Discover recurring, level-dependent decision patterns from Lichess games and engine counterfactuals, without predefined skill labels.

The initial deliverable is a small discovery report with held-out evidence and annotated PGN examples. Python 3.12 dependencies are locked in `uv.lock`; Stockfish is local under `.tools/`.

Run `uv run chess-research doctor` to check the environment. Run `uv run ruff check .` and `uv run mypy src` for code checks.

Keep compressed archives and derived datasets in `data/`, and generated reports in `reports/`. Stream `.pgn.zst` files in bounded batches; retain selected games rather than expanding the monthly archive.

The proof pipeline uses one run directory per sample. It accepts a local `.pgn`
or `.pgn.zst` archive; nothing is downloaded automatically. Run the 100-decision
pilot first, inspect its report and timing, then repeat with a fresh directory
and `--count 5000`.

```bash
uv run --offline chess-research doctor
uv run --offline chess-research sample /path/to/archive.pgn.zst reports/pilot --count 100
uv run --offline chess-research analyse reports/pilot
uv run --offline chess-research features reports/pilot
uv run --offline chess-research discover reports/pilot
uv run --offline chess-research report reports/pilot
```

`analyse` defaults to four Stockfish processes, one thread and 128 MiB hash each,
100,000 nodes per search, and atomic batches of 32 decisions. Repeating it skips
committed decisions and ignores interrupted `.tmp` files. Changing the engine
binary or settings creates a separate analysis directory. The completion JSON
records timing; benchmark before increasing concurrency. Run only one writer per
run directory. Stages write independent Parquet artifacts, configurations, and
completion markers. Discovery refuses to overwrite frozen findings.

Sampling shuffles eligible decisions within each streamed game, taking at most
ten per player, and stops at the game boundary when the requested sample is full.
It is a convenience sample of the recorded archive prefix, not a uniform sample
of the whole archive. The source path, size, modification time, seed and scanned
game count are recorded. Rated rapid standard games with no BOT player are
eligible; only decisions by players rated 400–799 are retained.

Features contain elementary attack, protection, and legal-capture counts before
and after the move, relationships created/removed/retained with piece identities
preserved, and mechanically generated pairs of relations sharing a target piece.
Types use the piece's pre-move identity (including promoted pawns). Legal captures
refer to the side whose turn it is at each snapshot. No rating or engine column
is allowed into the model matrix. Outcomes retain separate centipawn and mate
fields, moving-player perspective and UCI principal variations. Expected-score
loss uses python-chess's Stockfish WDL model; signed negative values are retained
as search/model sensitivity rather than clamped to zero.

Players are assigned by a seeded hash to 60/20/20 discovery/selection/evaluation
partitions. Positions use the first four FEN fields (ignoring move counters).
Positions present in an earlier partition are removed from later partitions.
The model has 100 trees, depth three, minimum leaf 50. Rules use integer count
thresholds, are deduplicated, and rank by selection coverage times excess cost
relative to nonmatching decisions. Selection requires 30 matches from 20 players;
final evidence additionally requires positive excess and a positive lower bound
of a 95% player-bootstrap interval. This is exploratory evidence for up to three
frozen choices, not a causal claim or a multiple-testing correction.

Reports include support, distinct players, confidence intervals, selection examples
and low-cost matching counterexamples. Examples are chosen from distinct players,
reanalysed at one million nodes, and exported with histories and engine variations
as `examples.pgn`; `sensitivity.json` records changes in example costs. A human
must review these examples before giving a rule a lesson name. Reports with no
candidates explicitly say the result is inconclusive.

The repository's synthetic fixture can exercise all stages using
`sample tests/fixtures/tiny.pgn reports/fixture --count 100 --fixture`.
Its 13 decisions do not provide cohort evidence. Tests include an artificial
numerical learning signal solely to verify selection and evaluation behavior.

```bash
uv run --offline pytest -q
uv run --offline ruff check .
uv run --offline ruff format --check .
uv run --offline mypy src
```
