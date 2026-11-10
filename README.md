# passphoto-validation

Checks a photo against the German ID photo rules (the Fotomustertafel) and says which rules it passes, which it fails and why. 39 checks feed 36 rules. Most face measurements come from [OFIQ](https://github.com/BSI-OFIQ/OFIQ-Project), the BSI's implementation of ISO/IEC 29794-5, the rest from MediaPipe and OpenCV. The answer comes as JSON or as a self-contained HTML report, with a proposed 35×45 mm crop when the framing is off.

![The HTML report for a sample photo: the verdict, the rules to fix with the reason for each, and the planned 35×45 crop that would make the photo valid](docs/report.png)

The Jupyter notebooks are the UI: `notebooks/pipeline_overview.ipynb` walks through the pipeline on the sample photos, `notebooks/evaluation.ipynb` shows how every check does on about 2,850 labelled photos and why the thresholds are where they are, `notebooks/stichprobe.ipynb` draws a random sample of those photos and shows each with its label and the validator's verdict, to check the claims by eye, `notebooks/api.ipynb` talks to the API. [docs/architecture.md](docs/architecture.md) is the one-page map of the pipeline, with a diagram; [PASSPORT_VALIDATOR.md](PASSPORT_VALIDATOR.md) has the list of checks, the report format and the measured accuracy.

## Install

Linux on x86_64 (OFIQ's build script also knows arm64 Linux; not tried here). The OFIQ build needs git, Python 3 with venv, a C++ toolchain and unzip; the Python side needs [uv](https://docs.astral.sh/uv/). These are the only steps that need root:

```bash
sudo apt-get install git build-essential python3-venv unzip
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and run the install script:

```bash
git clone git@github.com:simon-lund/passphoto-validation.git && cd passphoto-validation
```

```bash
./install.sh
```

The script

1. creates the Python environment (`uv sync --extra test`; Python 3.12 via `.python-version`, since MediaPipe has no wheels for newer versions),
2. clones OFIQ v1.2.1 into `../OFIQ-Project`, next to this repo (set `OFIQ_PROJECT_DIR` to use a checkout elsewhere),
3. builds OFIQ and its Python wheel with its own conan and cmake in `../OFIQ-Project/.conan-venv`. This takes about 15 minutes and downloads OFIQ's models (430 MB) from eu-LISA. OFIQ's conan profile would install missing system packages with sudo; the script switches it to reporting them, so if the build stops with a list of packages, install those and run the script again,
4. installs the OFIQ wheel into the environment,
5. writes `OFIQ_CONFIG_DIR=…/OFIQ-Project/data` to `.env` (gitignored). The CLI, the API and the tests read it when started from the repo root; the notebooks look for `../OFIQ-Project` themselves,
6. runs the validator once on a sample photo.

It is safe to run again, finished steps are skipped. Run it again after `git pull` as well: a plain `uv sync` removes the OFIQ wheel, `uv sync --inexact` (which the script uses) keeps it.

Options: `./install.sh --siglip` also installs the SigLIP second opinions on the eyes (PyTorch CPU; a 3.3 GB model downloads from Hugging Face on first use, to `HF_HOME`). `./install.sh --without-ofiq` skips OFIQ; the OFIQ checks then report `skipped`, the rules they cover end as `skipped`, and no photo is valid.

## Run

The CLI, with every check and its measurements:

```bash
uv run passport-validator validate tests/test-images/9.jpg -v
```

The API on http://127.0.0.1:9600 (docs at `/docs`); POST a photo to `/v1/check`, with `Accept: text/html` for the HTML report:

```bash
uv run passport-validator-api
```

The notebooks:

```bash
uv run --extra test jupyter lab notebooks/pipeline_overview.ipynb
```

The tests (the OFIQ ones run when `.env` points at OFIQ, otherwise they are skipped):

```bash
uv run --extra test pytest
```

## Thresholds

Every warn and fail threshold lives in one table, `src/passphoto_validation/core/thresholds.py`, each with a note on where the number came from. To run with other numbers without editing the code, write every parameter of every check to a file, keep the lines to change and pass the file back:

```bash
uv run passport-validator thresholds > thresholds.toml
```

```bash
uv run passport-validator validate tests/test-images/9.jpg --thresholds thresholds.toml
```

The API reads the same file from `THRESHOLDS_FILE` (or `.env`) and lists what it runs with at `/v1/thresholds`. A file that names a check or a parameter that does not exist, or gives a wrong type, stops the command or the server with a message, so a typo never silently keeps the default. The file is applied last, so its `[resolution]` values win over `MIN_RES_WIDTH` and the other resolution settings of the environment.

## Evaluation data

The labelled photo sets behind the evaluation notebooks are not in the repo. The [Face Research Lab London Set](https://doi.org/10.6084/m9.figshare.5047666) (102 people photographed on a grey backdrop, neutral and smiling, front and 3/4 views; CC BY 4.0, but its subjects consented to use in studies, so neither it nor anything made from it may be committed or redistributed) downloads in one step, and two more build and run the evaluation set: about 2,850 photos, the compliant crop of each person, their real smiling and 3/4-view photos, and 24 synthetic variants drawn into each compliant crop, 22 of which break exactly one rule and 2 of which (a 4° roll, clear glasses) must still pass. Even ids are the test half, odd ids the half the thresholds were set on.

```bash
uv run python scripts/download_london_set.py --out ../datasets/frl-london
```

```bash
uv run python scripts/build_eval_set.py --source ../datasets/frl-london --out ../datasets/frl-london-eval
```

```bash
uv run python scripts/run_eval.py --set ../datasets/frl-london-eval
```

For a quick set of test photos, `scripts/make_test_photos.py` takes any folder of frontal portraits (the London Set's `neutral_front`, or your own), crops each to a compliant photo and draws one defect into each copy, with a manifest that says which rule each photo breaks:

```bash
uv run python scripts/make_test_photos.py --source ../datasets/frl-london/neutral_front --out ../datasets/test-photos --count 100
```

`notebooks/stichprobe.ipynb` draws a random sample from a set (`LONDON_SET` points it at one, by default the evaluation set), runs the validator on every sampled photo and shows it with its label and the verdict, so the numbers in the evaluation can be checked by eye.

The 240 real photos behind `notebooks/real_photos.ipynb` come from Open Images (Flickr, CC BY 2.0) and are built by `scripts/build_openimages_set.py` from Google's metadata; `credits.csv` next to them lists each photo's author and licence.

## Licence

EUPL 1.2. The code started as a fork of eduTAP's `edutap.image_api` by Philipp Auersperg-Castell and Alexander Loechel (EUPL 1.2); the sample photos `tests/test-images/1.jpg` to `9.jpg` come from there.
