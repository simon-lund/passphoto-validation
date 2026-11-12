# For agents working in this repository

What this is, where things are, what to run and what must never happen. The README has the architecture and the principles; PASSPORT_VALIDATOR.md the checks, the report format and the measured numbers; docs/architecture.md the map.

## What it is

A checker for German ID photos (the Fotomustertafel): 39 measurements (checks) feed 36 rules; a report says which rules a photo breaks and why, as JSON, as an HTML page or on the terminal. Python 3.12, `uv`, OpenCV, MediaPipe and OFIQ (the BSI's ISO/IEC 29794-5 implementation, built from source).

## Layout

| Path | What |
| --- | --- |
| `src/passphoto_validation/` | the package: `core/validator.py` (the pipeline), `core/check.py` (what a check is: `Check`, `Sequence`), `core/checks/` (the checks as classes; `__init__.py` lists them in report order), `core/rules.py` (rules, problems), `core/thresholds.py` (every threshold, with provenance), `cli.py`, `check_api.py`, `report_html.py` |
| `scripts/` | download and build the photo sets, run the pipeline over a set, summarise, make test photos, the review page |
| `notebooks/` | the UI for looking at results: `pipeline_overview`, `evaluation`, `real_photos`, `stichprobe`, `api` |
| `tests/` | pytest; the OFIQ and SigLIP tests skip when those are not installed |
| `third_party/OFIQ-Project/` | git submodule, pinned at v1.2.1; `install.sh` builds it in place (its build and models are untracked) |
| `baseline/` | the Decisions-API baseline the pipeline is measured against; its own package and README |
| `data/` | the photo sets; ignored by git except `README.md` and `london-eval/reviews.jsonl` |
| `docs/` | the architecture map and the report image |

## Commands

```bash
./install.sh                 # environment, OFIQ (15 min once), .env; safe to re-run; use it instead of a plain uv sync
uv run --extra test pytest   # the tests (about 3 minutes with OFIQ)
uv run passport-validator validate tests/test-images/9.jpg -v
uv run passport-validator thresholds > thresholds.toml   # every parameter of every check, to edit and pass back
./reproduce.sh               # from a clone to the evaluation numbers (download, build, run, report)
uv run python scripts/review.py --set data/london-eval --split test --shuffle   # judge verdicts photo by photo
```

`uv run --inexact` keeps the OFIQ wheel; a plain `uv sync` removes it. Scripts take `--set data/london-eval` and the like; the notebooks default to the same folders. With several workers `scripts/run_eval.py` runs OFIQ with `ofiq_config_eval.jaxn` (3 threads per worker, written by `install.sh`) unless `OFIQ_CONFIG_FILE` names another file; without the limit the workers starve one another. It always writes to `--out` from scratch, so give a new file name to keep the old results. `THRESHOLDS_FILE` (environment or `.env`) is read by the CLI, the API and the review page alike.

## Rules that hold

- The London Set photos, and everything made from them (crops, variants, per-photo results, screenshots of them), never enter git and never leave `data/`. Open Images photos are CC BY and may be looked at; the London ones only for the study itself.
- Check names, rule ids and the keys of the report are contracts: add, never rename. `tests/test_check_api.py` and `tests/test_thresholds.py` pin them.
- A new threshold goes into `core/thresholds.py` with a note on where the number came from, the snapshot in `tests/test_thresholds.py` is updated on purpose, and `notebooks/evaluation.ipynb` is re-run. Thresholds are chosen on the odd-numbered London people (calibrate) and reported on the even-numbered ones (test).
- A new check is a `Check` subclass in `core/checks/` (the module docstring of `core/check.py` shows one): a `name`, its parameters as annotated class attributes with both levels (`warn_…`, `fail_…`), `stops_after` and `reads` where it depends on another check, and `measure`, which grades with `self.judge` (never with its own `if value > fail_above`). It is added to the list in `default_checks()` (`core/checks/__init__.py`), which is the order of the report; its calibrated numbers go into `core/thresholds.py`; it is mapped in `core/rules.py` and gets a panel in the evaluation notebook.
- A change to the checks must leave the verdicts alone unless it means to move them: run `scripts/run_eval.py --out <new file>` and compare statuses, reasons, rules and problems with `data/london-eval/results.jsonl`.
- Every error answer of a server (the check API, the review page's server) is `{"code": "<stable_snake_case>", "detail": "<text>"}`, raised as `ApiError` (`check_api.py`).
- No third-party API inside the pipeline. The baseline in `baseline/` is the one place a photo may be sent out, and only the London Set for the study.
- No attribution lines in commits. Commit messages say what changed and why, in prose.
- Work goes straight onto `main`; nothing is deployed, renamed or published without the owner's word.
