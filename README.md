# passphoto-decisions

The baseline for [passphoto-validation](https://github.com/simon-lund/passphoto-validation): the same 36 German ID photo rules (the Fotomustertafel), judged not by a pipeline of face models and pixel checks but by one call to [OpenAI's Decisions API](https://developers.openai.com/api/docs/guides/decisions). Each rule is a yes/no question about the photo; the API answers with the probability that the rule is broken, and a threshold pair turns that into `pass`, `warn` or `fail`. If the hand-built pipeline can't beat this, it isn't worth its complexity.

One request per photo: a developer message with the framing, the photo as a base64 JPEG, a `choice` question that counts the faces (the gate: anything but exactly one face skips every rule) and 36 `predicate` questions, one per rule, named by the rule id. Rule ids, groups and titles are the ones passphoto-validation uses, so the two can be laid side by side.

## Install

```bash
git clone git@github.com:simon-lund/passphoto-decisions.git && cd passphoto-decisions
```

```bash
uv sync
```

Set `OPENAI_API_KEY` in the environment, or put `OPENAI_API_KEY=...` in a `.env` file next to `pyproject.toml` (gitignored). `PASSPHOTO_MODEL` overrides the model (default `gpt-6-luna`, the only one the Decisions API serves today).

## Run

One photo, as a table or as JSON; `--dry-run` prints the request (without the image bytes) and sends nothing:

```bash
uv run passphoto-decisions check photo.jpg
```

```bash
uv run passphoto-decisions check photo.jpg --json
```

```bash
uv run passphoto-decisions check photo.jpg --dry-run
```

The questions, one per rule:

```bash
uv run passphoto-decisions rules
```

Options on `check` and `batch`: `--warn-at` and `--fail-at` (the thresholds on the probability that a rule is broken; defaults 0.4 and 0.7), `--detail` (the image detail level, default `high`), `--max-side` (shrink the photo before sending; by default it goes at full size, as the pipeline sees it) and `--model`.

## Evaluate a set

`batch` sends every photo of an evaluation set built by passphoto-validation's `scripts/build_eval_set.py` (a folder with `manifest.csv`) and writes `results_decisions.jsonl` next to it, in the shape its `scripts/run_eval.py` writes: the manifest row, `checks` (the gate as `face_detected`, each rule once more as a check whose measurement is the probability) and `rules` (rule id to status). Photos already in the output are skipped, so an interrupted run continues where it stopped. A rate limit (HTTP 429) is waited out, up to about two minutes per photo; a request that still fails becomes a row with an `error` on `face_detected` and every rule `skipped`, and the next run sends that photo again. Four parallel requests stay under the usual per-minute token limit; eight did not.

```bash
uv run passphoto-decisions batch ../datasets/frl-london-eval --yes
```

Without `--yes` it only says how many photos it would send. Then, from the passphoto-validation checkout, compare the two per rule (caught, warned, false fails and false warns on photos that should pass):

```bash
uv run --extra test python scripts/eval_report.py --set ../datasets/frl-london-eval --results results_decisions.jsonl --compare results.jsonl --split test
```

Only the rule tables are meaningful there; its per-check table looks for the pipeline's check names, which this baseline doesn't have.

## First results

The London eval set (2,856 photos: 102 people × 28 variants built by passphoto-validation) was run on 2026-11-08 for $2.00 (20 million input tokens, about 7,000 per photo). `scripts/compare_london.py <set> --split test` compares the two systems on the test half (even person numbers, 52 photos per defect, 156 photos that should pass) and sweeps the thresholds; `uv run --group dev python scripts/compare_london.py ...` runs it.

| On the test half | Pipeline | Baseline, one threshold pair (0.4 / 0.7) | Baseline, per-rule thresholds from the calibrate half |
|---|---|---|---|
| Good photos with no failing rule | 96% | 6% | 70% |
| Good photos with neither fail nor warn | 83% | 0% | 24% |
| Small head caught | 100% | 0% | 0% |
| Off-centre face caught | 100% | 2% | 31% |
| Smoothed skin flagged | 100% (warn) | 17–25% | 0–4% |
| Background shadow caught | 83% | 100% | 100% |
| Colour cast caught | 81% | 100% | 100% |
| Low contrast | warn | fail, 100% | fail, 100% |
| Stickers on the face | warn | fail, 100% | fail, 98% |

Every other synthetic defect (blur, JPEG, dark, bright, pattern, object, red eyes, sunglasses, glare, yaw, roll, ink stains) is caught by both at 85 to 100%. What sinks the baseline is false alarms on compliant photos: `no_color_gradients` fires on 99% of them, `photo_format` on 63% (crops that are exactly 7:9), `no_face_shadows` on 47%, and no global threshold fixes that (at 0.95, 28% of good photos still get a false fail). Per-rule thresholds (fail above the 99th percentile of good calibrate photos) make it usable but leave it far behind on good photos, and it cannot measure proportions at all. The model is the better detector for several visual defects, which is worth knowing when the pipeline's checks are tuned.

## What to know before running it

- **Every photo leaves the machine.** passphoto-validation was built so that no face photo goes to a third-party API; this repo exists to measure that pipeline against the simplest alternative, and it does the opposite. The London Set's consent covers studies only: sending it to OpenAI is a decision to take, not a default. The nine sample photos in passphoto-validation's `tests/test-images` are public.
- **The thresholds are placeholders.** 0.4 and 0.7 were not calibrated on anything. Once the London Set has been run, choose them on the `calibrate` half (odd person numbers) and report the `test` half, the way passphoto-validation does; the probabilities are in the results file, so changing thresholds needs no second run.
- **Cost.** Decisions input costs $0.10 per million tokens and no output is charged; `check` and `batch` print the input tokens and the estimated cost. Two rule pairs ask the same question (the two overexposure rules; head rotation and half profile), which is a few tokens, not a problem.
- **Not here.** The API answers probabilities only, so there are no measurements, no reasons beyond the question itself, no crop proposal, no HTML report and no HTTP server. Those are the pipeline's; the baseline is rules, statuses and the results file.

## Tests

```bash
uv run pytest
```

The tests use a fake client with answers built from the API's documented JSON through the SDK's own `Decision` model; nothing reaches the network.

## Licence

EUPL 1.2, like passphoto-validation, whose rule catalogue this copies.
