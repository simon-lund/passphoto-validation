#!/usr/bin/env bash
# From a clone to the evaluation numbers the docs claim, in one run. Safe to run again: every step skips
# what is already there.
#
#   ./reproduce.sh                   install, the London Set, the evaluation set, the pipeline's results, the report
#   ./reproduce.sh --with-openimages also the 240 real photos from Open Images (needs Google's metadata; see the script)
#   ./reproduce.sh --with-baseline   also the Decisions-API baseline (baseline/; needs OPENAI_API_KEY in baseline/.env, about $2)
#   ./reproduce.sh --skip-install    the environment and OFIQ are already there
#
# What it takes, on a desktop CPU: the OFIQ build about 15 minutes (once), the London download 136 MB, the
# evaluation set a few minutes, the pipeline over 2,856 photos about 17 minutes with 4 workers.
# The photos stay in data/ and never enter git (data/README.md).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
WITH_OPENIMAGES=0; WITH_BASELINE=0; SKIP_INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --with-openimages) WITH_OPENIMAGES=1 ;;
    --with-baseline) WITH_BASELINE=1 ;;
    --skip-install) SKIP_INSTALL=1 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "reproduce.sh: unknown option $arg (try --help)" >&2; exit 2 ;;
  esac
done
say() { printf '\n==> %s\n' "$*"; }
export GLOG_minloglevel=2

# 1. The environment and OFIQ
if [ "$SKIP_INSTALL" = 0 ]; then
  say "Install (./install.sh)"
  ./install.sh
fi
OFIQ_DIR="$(grep '^OFIQ_CONFIG_DIR=' .env | cut -d= -f2-)"
[ -n "$OFIQ_DIR" ] || { echo "reproduce.sh: .env has no OFIQ_CONFIG_DIR; run ./install.sh first" >&2; exit 1; }
# Several workers need OFIQ limited to 3 threads each, or they starve one another
export OFIQ_CONFIG_FILE=ofiq_config_eval.jaxn
[ -f "$OFIQ_DIR/ofiq_config_eval.jaxn" ] || sed 's/"ort_intra_threads": 0/"ort_intra_threads": 3/' "$OFIQ_DIR/ofiq_config.jaxn" > "$OFIQ_DIR/ofiq_config_eval.jaxn"

# 2. The London Set and the evaluation set built from it
say "The London Set (figshare, MD5-checked)"
uv run --inexact python scripts/download_london_set.py --out data/london
if [ -f data/london-eval/manifest.csv ]; then
  say "Evaluation set found: data/london-eval ($(wc -l < data/london-eval/manifest.csv) lines in manifest.csv)"
else
  say "Building the evaluation set (data/london-eval)"
  uv run --inexact python scripts/build_eval_set.py --source data/london --out data/london-eval
fi

# 3. The pipeline over every photo, then the report the docs quote
say "Running the pipeline over the evaluation set (about 17 minutes with 4 workers)"
uv run --inexact python scripts/run_eval.py --set data/london-eval --workers 4
say "Per rule and per check, test split (compare with PASSPORT_VALIDATOR.md, 'How well it works')"
uv run --inexact --extra test python scripts/eval_report.py --set data/london-eval --split test

# 4. Optional: the real photos
if [ "$WITH_OPENIMAGES" = 1 ]; then
  say "Open Images (data/openimages)"
  uv run --inexact python scripts/build_openimages_set.py --meta data/openimages/meta --out data/openimages/eval --per-group 60
  uv run --inexact python scripts/run_eval.py --set data/openimages/eval --workers 4
  uv run --inexact --extra test python scripts/score_labels.py --set data/openimages/eval
fi

# 5. Optional: the baseline
if [ "$WITH_BASELINE" = 1 ]; then
  say "The Decisions-API baseline (baseline/)"
  (cd baseline && uv run passphoto-decisions batch ../data/london-eval --yes)
  uv run --inexact --extra test python scripts/eval_report.py --set data/london-eval --split test --results results_decisions.jsonl --compare results.jsonl
fi

say "Done. The notebooks read the same files: uv run --extra test jupyter lab notebooks/evaluation.ipynb"
