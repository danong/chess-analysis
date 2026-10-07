# Chess research

Discover recurring, level-dependent decision patterns from Lichess games and engine counterfactuals, without predefined skill labels.

The initial deliverable is a small discovery report with held-out evidence and annotated PGN examples. Python 3.12 dependencies are locked in `uv.lock`; Stockfish is local under `.tools/`.

Run `uv run chess-research doctor` to check the environment. Run `uv run ruff check .` and `uv run mypy src` for code checks.

Keep compressed archives and derived datasets in `data/`, and generated reports in `reports/`. Stream `.pgn.zst` files in bounded batches; retain selected games rather than expanding the monthly archive.

## Active trajectory discovery proof

The active experiment clusters deterministic differences between played and
engine-preferred continuations. It does not call an LLM, embed text, or detect
named tactical skills. An assistant or human reviews discovery cluster packets
outside the pipeline, and persists that review before candidates are measured.

Use a new run directory and reserve fresh evaluation players. The previous
5,000-decision symbolic experiment has already been inspected and is not a fresh
holdout. `--prior-run` excludes its players from new evaluation sampling and its
positions from the evaluation split; historical exclusions carry forward.

```bash
uv run --offline chess-research doctor
uv run --offline chess-research sample /path/to/archive.pgn.zst reports/trajectory-proof \
  --count 6000 --seed 20261007 --rating-min 400 --rating-max 1199 --stratified \
  --prior-run reports/beginner-5000
uv run --offline chess-research analyse reports/trajectory-proof
uv run --offline chess-research episodes reports/trajectory-proof
uv run --offline chess-research engine-audit reports/trajectory-proof
uv run --offline chess-research cluster reports/trajectory-proof --max-episodes 200
uv run --offline chess-research review-packets reports/trajectory-proof
```

Episodes retain source IDs, history and full pre-move FEN, moving-player rating
and side, human/preferred moves, both PVs, separate cp/mate scores, expected-score
loss, engine provenance, and encoding availability. The default mistake gate is
0.10 expected-score loss. All sampled eligible decisions remain denominators;
mate values are never converted to centipawns. `episodes.config.json` records the
exact feature allowlist and input hashes. `split-audit.json` lists exclusions.

The six-ply vector contains per-ply material-count changes, captures, checks,
promotions and automatic terminal outcomes for each continuation and their
difference. Claimable draws do not terminate otherwise legal PVs. Truncated
continuations have no vector and are explicitly unclassified. Complete quiet
lines without any represented event are also unclassified. Ratings, identities,
square names, opening labels and cost magnitude do not enter the vector.

Clustering uses discovery-only standard-deviation scaling without centering
(preserving shared event signals), average linkage and cosine distance 0.3.
The discovery sample is rating-stratified and position-deduplicated. It is not a
prevalence sample. Every clustering configuration creates a separate
`clusters-<hash>.json`; `active-clusters.json` identifies the current artifact.
If the pilot warrants expansion, repeat `cluster --max-episodes 1000` before
freezing. Inspect central and boundary examples, paired PGNs, and concentration
flags under `inspection/<clusters-artifact-name>/`.

Review every cluster in a copy of `review.template.json`. Record reviewer,
coherence, acceptance, rationale, contradictions, source example IDs, and an
optional observed-decision name. Support requires at least 30 episodes, 30
positions and 20 players; support alone does not establish coherence. Names must
not infer a player's mental state. Reject mixed clusters rather than naming them.

```bash
uv run --offline chess-research review-packets reports/trajectory-proof --partition selection
# Review the generated selection packet and save selection-review.json.
uv run --offline chess-research freeze reports/trajectory-proof \
  reports/trajectory-proof/review.json reports/trajectory-proof/selection-review.json
uv run --offline chess-research measure reports/trajectory-proof
```

Reviews bind to the active cluster hash. An accepted family requires every
selection-packet example reviewed, at least five correct assignments, and at
least 80% accuracy among submitted audit judgments. These central/boundary audits
are purposive checks, not unbiased classifier accuracy estimates. Rejection
radii are calibrated from those judgments. Classification routes through all
supported discovery centroids; a nearest rejected family remains unclassified
rather than falling through to an accepted family. Each episode contributes to
at most one accepted family.

`taxonomy.json` freezes the scaler, centroids, reviewed definitions and rejection
rules. Hashes prevent changed input artifacts from being used for measurement.
`measurement.json`, `trajectory-report.md` and `rating-costs.png` report incidents
and expected-score cost per 100 sampled decisions, qualifying-mistake share,
assignment coverage, player-bootstrap intervals and adjacent-band changes.
Empty taxonomies are valid negative results, not evidence of zero chess errors.

Claims are conditional on the archive prefix, player cap, rating strata and
mistake gate. Expected-score saturation can miss material deterioration in
already won/lost positions. Material/event clusters may describe consequences
without explaining decisions. Rating associations do not establish individual
development or teaching effectiveness. Complete-game/player prevalence,
hierarchy, exercise generation and personalized curricula are deferred.

Stratified sampling reserves 60/20/20 discovery/selection/evaluation quotas
within each rating band, rather than letting early discovery players fill an
entire band. Prior evaluation identities and positions are excluded before quota
consumption. Remaining cross-partition repeated positions can still reduce
evaluation counts; inspect `partition_band_counts` and `split-audit.json`.

To measure an existing frozen taxonomy on a separate fresh run, build that run's
episodes with the same horizon and gate and use `measure NEW_RUN --taxonomy-run
FROZEN_RUN`. This verifies the source taxonomy and encoder hashes, requires
matching engine settings, and rejects overlapping source players or positions.
It does not discover or revise families on the new run.

## Legacy symbolic experiment

The original symbolic proof is retained for comparison and is superseded as the
active research path. Its CLI commands are marked legacy. It uses one run directory per sample. It accepts a local `.pgn`
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
