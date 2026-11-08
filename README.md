# passphoto-validation

Checks a photo against the German ID photo rules (the Fotomustertafel) and says which rules it passes, which it fails and why. 38 checks feed 36 rules. Most face measurements come from [OFIQ](https://github.com/BSI-OFIQ/OFIQ-Project), the BSI's implementation of ISO/IEC 29794-5, the rest from MediaPipe and OpenCV. The answer comes as JSON or as a self-contained HTML report, with a proposed 35×45 mm crop when the framing is off.

The Jupyter notebooks are the UI: `notebooks/pipeline_overview.ipynb` walks through the pipeline on the sample photos, `notebooks/evaluation.ipynb` shows how every check does on about 2,200 labelled photos and why the thresholds are where they are, `notebooks/api.ipynb` talks to the API. [PASSPORT_VALIDATOR.md](PASSPORT_VALIDATOR.md) has the architecture, the list of checks, the report format and the measured accuracy.

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

## Evaluation data

The labelled photo sets behind `notebooks/evaluation.ipynb` and `notebooks/real_photos.ipynb` (the Face Research Lab London Set and 240 Flickr photos from Open Images) are not in the repo. `scripts/build_eval_set.py` and `scripts/build_openimages_set.py` build them from the originals, `scripts/run_eval.py` runs the validator over a set and writes the results the notebooks read.

## Not maintained

`Dockerfile`, `docker-compose.yaml` and `traefik.yaml` are left from the eduTAP service this started as. They install neither OFIQ nor the current entry points.

## Licence

EUPL 1.2. The sample photos `tests/test-images/1.jpg` to `9.jpg` come from eduTAP's `edutap.image_api`.
