# The photo sets

Everything in this folder except this file, `.gitignore` and `london-eval/reviews.jsonl` is ignored by git. The photos are fetched and built by the scripts in `../scripts/`; `../reproduce.sh` runs them in order. Paths below are relative to the repository root.

| Folder | What | How it gets here |
| --- | --- | --- |
| `data/london/` | The Face Research Lab London Set as downloaded from figshare: `neutral_front/`, `smiling_front/`, `neutral_left_3quarter/`, `neutral_right_3quarter/` (102 people, 1350×1350), `london_faces_info.csv`. 136 MB. | `uv run python scripts/download_london_set.py --out data/london` (MD5-checked, re-runnable) |
| `data/london-eval/` | The evaluation set built from it: 2,856 photos (per person the compliant 35×45 crop, the real smiling and 3/4 photos, 24 synthetic variants), `manifest.csv` (file, identity, split, variant, broken_rule). Odd ids are the calibrate split, even ids the test split. | `uv run python scripts/build_eval_set.py --source data/london --out data/london-eval` |
| `data/london-eval/results.jsonl` | The pipeline's verdict on every photo: every check, every rule, the problems. | `uv run python scripts/run_eval.py --set data/london-eval` (27 minutes with 4 workers on a 20-thread desktop CPU) |
| `data/london-eval/results_decisions.jsonl` | The baseline's verdicts (`../baseline/`, OpenAI's Decisions API, $2 per run). | `cd baseline && uv run passphoto-decisions batch ../data/london-eval --yes` |
| `data/london-eval/reviews.jsonl` | Answers given on the review page (`scripts/review.py`): file, right or wrong, rules marked wrong, comment, the verdict as shown. Tracked. | `uv run python scripts/review.py --set data/london-eval --split test --shuffle` |
| `data/openimages/` | 240 real candid photos from Open Images (Flickr, CC BY 2.0) with glasses, hats and closed eyes: `eval/` (the photos, `manifest.csv`, `credits.csv`, `vlm_labels.jsonl`, `results.jsonl`), `meta/` (Google's metadata), `raw/`. | `uv run python scripts/build_openimages_set.py --meta data/openimages/meta --out data/openimages/eval --per-group 60` |
| `data/test-photos/` | A small labelled set from any folder of frontal portraits, for a quick look. | `uv run python scripts/make_test_photos.py --source data/london/neutral_front --out data/test-photos --count 100` |
| `data/stichprobe/` | Executed copies of `notebooks/stichprobe.ipynb` (HTML), when exported. | the notebook |

**Consent.** The London Set (DeBruine & Jones 2017, CC BY 4.0) was collected with consent for use in lab-based and web-based studies. Neither the photos nor anything made from them (crops, variants, per-photo results, screenshots) may be committed, uploaded or shown to anyone outside such use. The scripts write only into this folder.

**Expected counts** after `reproduce.sh`: 408 photos in `london/` (102 × 4 views), 2,856 in `london-eval/` plus `manifest.csv`, 240 in `openimages/eval/`. The numbers in `../PASSPORT_VALIDATOR.md` come from `scripts/eval_report.py --set data/london-eval --split test`.
