# Passport Photo Validator - Modular Architecture

## Quick start

```bash
./install.sh
```

The script creates the Python environment with uv, builds **OFIQ** once next to this repo and writes its location to `.env`; the prerequisites, what it does step by step and its options are in the [README](README.md). OFIQ is the BSI's open-source implementation of ISO/IEC 29794-5 (face image quality), co-maintained by eu-LISA, and most face checks come from it. Without it the OFIQ checks report `skipped` with the reason, the rules they cover end as `skipped`, and no photo is valid.

Optionally, **SigLIP** SO400M (Google, Apache 2.0) adds two warn-only second opinions on the eyes: reflections on glasses and gaze (`./install.sh --siglip`). It needs PyTorch (CPU) and a 3.3 GB model, which downloads from Hugging Face on first use (to `HF_HOME`, by default `~/.cache/huggingface`). It looks at the eyes OFIQ found, so it needs OFIQ too; without either, its checks report `skipped` with the reason. Loading takes about 25 s and 3.5 GB of memory; each photo takes about a second longer. `SIGLIP_THREADS` limits its threads when several validators run in parallel.

See the whole pipeline at work on the sample photos (per rule and per check, plus altered copies as sanity checks):

```bash
uv run --extra test jupyter lab notebooks/pipeline_overview.ipynb
```

`notebooks/evaluation.ipynb` shows how every check and rule does on about 2,850 labelled photos, and why the thresholds are where they are. `notebooks/stichprobe.ipynb` draws a random sample of those photos, runs the validator on each and shows it with its label and the verdict, to check the numbers by eye. `notebooks/visualize_checks2.ipynb` shows the internals of the background checks. `notebooks/api.ipynb` sends photos to the API and shows both answers, the JSON and the HTML report. Other entry points:

```bash
uv run passport-validator validate tests/test-images/9.jpg -v
```

```bash
uv run passport-validator-api
```

```bash
uv run --extra test pytest
```

## Overview

The passport photo validator has been refactored into a modular architecture with clear separation of concerns:

```
src/passphoto_validation/
├── core/                    # Business logic (pure)
│   ├── __init__.py
│   └── validator.py         # Validation engine
├── cli.py                   # CLI interface (Typer)
├── check_api.py             # API interface (FastAPI)
└── ...
```

## Architecture

### 1. **Core Module** (`core/`)
Pure business logic with zero dependencies on CLI or API frameworks.

**Components:**
- `PassportPhotoValidator` - Main validation engine
- `ValidationIssue` - Data class for individual issues
- `PassportPhotoReport` - Comprehensive validation report

**Features:**
- German ID photo rules (Fotomustertafel), measured largely with ISO/IEC 29794-5 (OFIQ)
- 39 checks, each reported with status, reasons and measurements
- Detailed measurements and scoring
- Suggested crop regions for fixes

### 2. **CLI Interface** (`cli.py`)
Beautiful command-line interface using **Typer** and **Rich**.

**Commands:**

#### `validate` - Single image validation
```bash
passport-validator validate <image_path> [OPTIONS]

Options:
  --output, -o PATH        Save JSON report to file
  --confidence, -c FLOAT   Face detection confidence (0.0-1.0, default: 0.5)
  --verbose, -v           Show detailed measurements
```

Example:
```bash
# Basic validation
passport-validator validate photo.jpg

# Save report as JSON
passport-validator validate photo.jpg -o report.json

# Verbose mode with measurements
passport-validator validate photo.jpg -v

# Adjust face detection confidence
passport-validator validate photo.jpg -c 0.7
```

#### `batch` - Multiple image validation
```bash
passport-validator batch <input_dir> [OPTIONS]

Options:
  --output, -o PATH       Directory to save JSON reports
  --pattern, -p TEXT      File pattern (default: *.jpg)
  --confidence, -c FLOAT  Face detection confidence
```

Example:
```bash
# Validate all JPGs in a directory
passport-validator batch ./photos/

# Process PNGs and save reports
passport-validator batch ./photos/ -p "*.png" -o ./reports/
```

#### `version` - Version info
```bash
passport-validator version
```

### 3. **API Interface** (`check_api.py`)
FastAPI service that uses the core validation logic. No auth: it is a prototype.

**Endpoints:**
```
POST /v1/check     multipart, photo in `file`: every check and every rule with status and reasons
GET  /v1/rules     the rule catalogue
GET  /health
```

`/v1/check` answers in JSON unless the request's `Accept` header prefers `text/html` over `application/json` (a browser's does, `*/*` doesn't). Then it returns the same report as one self-contained HTML page (`report_html.py`, `templates/report.html`): the verdict, what to fix, the photo with the detected face and the planned 35x45 crop drawn in, what that crop fixes, and every rule by group with its checks and measurements. The photo is embedded (at most 800 px), and so is the JSON answer, which the page's Copy JSON and Download JSON buttons hand out; the page can be saved, mailed or attached as it is. Both answers carry `Vary: Accept`.

The JSON answer's schema is in `/docs` and `/openapi.json` (`api_schema.py`: the models describe the answer, they don't produce it, and the tests validate real answers against them with extra keys forbidden, so they can't drift). Errors are `{"code": "...", "detail": "..."}`, also when HTML was asked for; branch on the status and `code`, never on the text. The codes: `invalid_request` (422, the multipart field is missing), `image_empty` (400), `invalid_image` (400, not an image we can read), `image_too_large` (413, over 15 MB) and `internal_error` (500, a bug in the pipeline; the traceback is in the server log, not in the answer).

**Usage:**
```bash
# Start the API server (http://127.0.0.1:9600, docs at /docs)
uv run passport-validator-api

# Test with curl
curl -F "file=@photo.jpg" http://127.0.0.1:9600/v1/check

# The HTML report, saved to report.html
curl -F "file=@photo.jpg" -H "Accept: text/html" -o report.html http://127.0.0.1:9600/v1/check
```

## Installation

`./install.sh`, see the [README](README.md).

## Usage Examples

### CLI Usage

**1. Quick validation:**
```bash
passport-validator validate tests/test-images/1.jpg
```

**Output:**
```
╭──────────────── Validation Result ────────────────╮
│ ✗ INVALID                                          │
│                                                    │
│ Overall Score: 39.0/100                           │
╰────────────────────────────────────────────────────╯

Found 8 issue(s):

  Type       Severity   Issue
 ──────────────────────────────────────────────────
  ERROR      █████████  Face is too small (39.3%)
  WARNING    ███████    Harsh shadows detected
  ...

💡 Suggestion:
  Face is too small - crop to this region
  Crop region: x=275, y=144, width=402, height=536
```

**2. Batch processing:**
```bash
passport-validator batch ./photos/ -o ./reports/
```

**3. Programmatic usage (Python):**
```python
from PIL import Image
from passphoto_validation.core import PassportPhotoValidator, ValidatorConfig
from passphoto_validation.core.utils import prepare_image

# Load image (applies EXIF rotation, returns BGR for OpenCV)
image = prepare_image(Image.open("photo.jpg"))

# Validate
validator = PassportPhotoValidator(ValidatorConfig(min_detection_confidence=0.5))
report = validator(image)

# Every rule and every check, including passes
for rule in report.rules:
    print(f"{rule.status:12} {rule.title}  {'; '.join(rule.reasons)}")
for check in report.checks:
    print(f"{check.status:8} {check.name}  {check.measurements}")

# Get measurements
print(f"Head height: {report.measurements['head_height_ratio']:.1%}")

# Export to JSON
report_dict = report.to_dict()
```

### API Usage

**1. Start server:**
```bash
uv run passport-validator-api
```

**2. Call endpoint:**
```python
import requests

url = "http://127.0.0.1:9600/v1/check"
files = {"file": open("photo.jpg", "rb")}
response = requests.post(url, files=files)
report = response.json()

print(f"Valid: {report['is_valid']}")
print(f"Score: {report['overall_score']}")
```

## Validation Checks

The warn and fail thresholds of every check live in one table, `core/thresholds.py`, with a note on where each number came from; the validator reads it when it registers its checks, and `notebooks/evaluation.ipynb` draws the same numbers as the warn and fail lines of its panels. To change a threshold, edit it there, update the snapshot in `tests/test_thresholds.py` (it pins the table, so a change is deliberate) and re-run the notebook to see the effect. Checks without an entry run with the defaults of their function; `resolution` takes its minimums from `ValidatorConfig`. A deployment can override any parameter of any check without editing the code: `passport-validator thresholds` writes them all to a TOML file, `--thresholds` or `THRESHOLDS_FILE` reads it back, and `/v1/thresholds` shows what the API runs with (see the README). Every check has a warn level and a fail level; `false` turns a level off and `enabled = false` turns a check off.

The validator runs 39 checks (`PassportPhotoValidator.check_names()`). Each ends as `pass`, `warn`, `fail`, `skipped` (it couldn't judge the photo, e.g. no face, or OFIQ isn't installed) or `error` (it crashed, which is a bug).

The German ID photo rules (Fotomustertafel) are listed in `core/rules.py`: 36 rules, each mapped to the checks that cover it. A rule takes the worst status of its checks that ran; a rule no check covers is `not_checked`, never a pass. A rule none of whose checks could run (OFIQ missing, no face landmarks) is `skipped`, and so is a rule one of whose checks crashed while none failed: a passing check doesn't make up for one whose verdict is missing.

A photo is valid (`is_valid`) only when no check reports an error, no check crashed, **and** no rule is `skipped`: a photo the pipeline couldn't judge is never valid. `skipped_rules` lists those rules; each one's `reasons` say why. A `not_checked` rule (one whose checks a thresholds file turned off) doesn't count against a photo, since no photo could pass it. Without OFIQ, every rule only OFIQ covers is skipped, so no photo is valid.

26 rules are fully covered and 10 partly; each rule's `note` says what its checks leave out. The last rules to get a check were background shadows, colour gradients, print errors such as ink stains, and retouching (skin smoothing, then coloured stickers and whole-face filters); these are the youngest checks, calibrated on synthetic defects only (see the table below), so treat their verdicts with more care than the rest.

| Source | Checks |
| --- | --- |
| OFIQ (ISO/IEC 29794-5) | `ofiq_head_yaw`, `ofiq_head_pitch`, `ofiq_head_roll` (3DDFA-V2), `ofiq_expression` (two EfficientNets and AdaBoost), `ofiq_mouth_closed`, `ofiq_eyes_open` (98 landmarks), `ofiq_eyes_visible`, `ofiq_face_occlusion`, `ofiq_mouth_occlusion` (occlusion segmentation), `ofiq_head_covering` (face parsing), `ofiq_background`, `ofiq_illumination`, `ofiq_overexposure`, `ofiq_natural_colour` |
| Our own, on OFIQ's face parsing | `glasses_tint`, `glasses_glare` |
| SigLIP SO400M zero-shot on OFIQ's eye region (optional, warn only) | `siglip_glasses_glare`, `siglip_gaze` |
| Our own | `face_detected`, `resolution`, `photo_format` (35x45), `face_size` (head 70-80% of the height), `face_centered`, `face_sharpness`, `compression` (JPEG blocks), `exposure`, `contrast`, `colour_cast` (a tint the background and the whites of the eyes share), `background_brightness`, `background_uniformity`, `background_consistency`, `background_pattern`, `glare`, `gaze`, `red_eyes` |
| Our own, on the selfie segmentation and face mesh (the youngest checks; since 2026-11-11 the person mask every background check reads also holds the hair and hats OFIQ's face parsing labels where they touch the segmented person, which the segmentation left in the background for two London Set people with voluminous hair) | `background_gradient` (lightness change across a plane fitted to the background), `background_shadow` (how much darker the band next to the person is than that plane, and how much darker the backdrop-like pixels in the rim of the person mask are, for shadows the segmentation swallowed), `print_defects` (small high-contrast marks on the background), `skin_smoothing` (cheek and forehead texture relative to the eyes, and relative to the face's gradient for a filter that smooths the eyes too; warns only), `skin_colour_patches` (patches of a colour that is not skin on the lower face, as a filter's stickers or face paint; warns only) |

The four youngest checks were calibrated on the London Set's compliant crops and on synthetic defects made from them (`scripts/build_eval_set.py`: `bg_gradient`, `bg_shadow`, `ink_stain`, `smoothed`), not on real defects. On the test half (52 people the thresholds weren't chosen on) they fail no photo that should pass; `print_defects` warns on 2% of them (stray hair the segmentation misses reads as a mark). Over both halves, one of the 306 photos that should pass fails on marks (a copy rotated by 4°, whose replicated border adds streaks), 4% warn on marks and 1% on shadows; with them, 80% of the photos that should pass get neither a fail nor a warning on any rule, against 83% before. They catch every synthetic gradient (fail) and smoothing (warn, by design), 83% of the shadows with another 10% warned, and 96% of the stains with the rest warned. The band's margin is thin: compliant crops reach a depth of 8.9 against a fail threshold of 9, and no real shadow was in the calibration. The shadows the band missed (17% of the variant) were not a threshold matter: on those photos the selfie segmentation takes the dark shadow next to the person for part of the person, and the band the check reads is then clean (with the clean photo's mask the same shadows measure 7 to 17). A tighter mask alone (confidence above 0.7 instead of 0.5) catches them but fails 3 to 4% of the good crops, whose hair and clothing then read as shadow. So the check also reads the rim of the mask (what the loose mask includes and a tight one at 0.85 does not, plus the first band outside), counting only pixels that look like a darkened backdrop: within 8 CIELAB units of the background's chroma, a local lightness deviation under 3, no darker than half the expected lightness. That `background_shadow_rim_depth` is 28 L* or more on every shadow variant of both splits and at most 4.7 on the compliant, glasses and rotated crops (a texture cap of 5 would let them reach 7.3); objects, gradients, face shadows and ink stains don't move it. The verdict follows the deeper of band and rim: every shadow variant now fails on both splits (83% before), and no good crop gained a flag. On the 240 real candid photos, most of them shot in rooms rather than against a backdrop, the band alone already failed half for shadows; the rim adds 5% fails and 1% warnings, 7 points on the 74 plain-wall photos, where the ones inspected do show a darkening behind the person, but no shadow labels exist for that set. On real candid photos (Open Images), `skin_smoothing` warns on 3%: a posterised selfie, a doll and a very smooth child's portrait.

Face filters (Snapchat-style) were tried as hand-drawn overlays on a sample photo: cat ears, a flower crown and a smoothing of the whole face passed every check; a dog muzzle, sparkles and butterflies were caught only by accident (as mouth occlusion, marks or an uneven background). Two measures were added for them, calibrated on two more synthetic variants (`face_stickers`: a pink nose and a butterfly on each cheek; `smoothed_all`: the edge-preserving filter over the whole face, eyes included). `skin_colour_patches` looks at the lower face, inside the person mask and without the lips, for patches more than 30 CIELAB chroma units from the skin's colour and at least as saturated: every London crop (plain, glasses, sunglasses, colour cast, side shadow) measures 0, the sticker variant 3% or more of the face oval, so it warns above 1.5% and, since the review of 2026-11-10, fails above 2.5% (before that it only warned). On the 230 real photos with a face, 2% warn: mirrored goggles, a clown nose, a glass and a piece of fruit at the mouth. It cannot see white or black marks (sparkles, drawn whiskers), anything on the forehead or above the head (hats, hair and real hair accessories would read the same), and strong coloured light on one side of the face reads as paint. `skin_smoothing`'s second measure, the skin's texture over the median gradient of the face, stays above 0.125 on every London crop that should pass (also darkened, low-contrast and colour-cast copies) and below 0.11 on the whole-face variant, on both splits; it warns below 0.115, except when the sharpness check has already failed the photo (a blur lowers it too). An overexposed face (blown-out skin) trips it in a quarter of the cases, as it does the texture ratio.

Some OFIQ measures aren't used because they failed on the evaluation set:
- `UnderExposurePrevention` rates well-lit dark skin as underexposed. On the London Set it scored below 100 only for Black subjects, down to 18. The `exposure` check judges the brightest parts of the photo instead.
- `Sharpness` drops from 84 to 38 when a photo is shrunk by 3%.
- `CompressionArtifacts` doesn't see JPEG quality 4 at ID-photo resolution.
- `DynamicRange` doesn't separate low-contrast photos from good ones.

The checks they replaced (BRISQUE, the solvePnP head pose, face-brightness exposure, face shadows by variance, eye and mouth ratios from the face mesh) were removed for the same reason. The old pipeline's results are in the notebook for comparison.

**How well it works** (London Set, 102 people; see `notebooks/evaluation.ipynb`, and `notebooks/stichprobe.ipynb` for a random sample of the photos with their verdicts):
- **Defects caught as fail:** 94-100% for most. The exceptions:
  - smiles: 67% fail, 13% warn;
  - colour casts: 100% since `colour_cast` (2026-11-10) reads the tint the background and the whites of the eyes share: OFIQ's natural colour alone caught 89%, scoring one cast in ten as natural. On the London backdrop the measure is easy (every cast 32 or more, every good crop 0); on real candid photos the rule fails 12% of the photos instead of 9%, and the two candid photos that were valid before now fail on a warm indoor cast;
  - low contrast: fails since 2026-11-10 (it warned only before; the review found a grey photo is not a valid photo).
- **Photos that should pass:** 96% fail no rule (the old pipeline: 11%), and 83% get no warning either. The rest are borderline head poses and a few backgrounds.

The set comes from one studio, and most defects are synthetic, so treat these numbers as an upper bound.

**Real photos with glasses, hats and closed eyes** (240 Open Images photos from Flickr, CC BY 2.0, labelled by Claude subagents; built with `scripts/build_openimages_set.py`, scored with `scripts/score_labels.py`). They are candid photos, not ID photos, so pose and gaze vary far more than in a photo booth. Caught means a warning or a fail:

| Problem | Checks | Caught | False alarms |
| --- | --- | --- | --- |
| Hat or cap | `ofiq_head_covering` | 90% | 8% |
| Sunglasses or tinted glasses | `glasses_tint`, `ofiq_eyes_visible` | 92% | 19% of clear glasses, 1% without glasses |
| Reflections on glasses | `glasses_glare` | 10% | 5% |
| Closed eyes | `ofiq_eyes_open` | 56% (40% fail) | 10% (1% fail) |
| Eyes looking away | `gaze` | 59% | 18% |
| Open mouth | `ofiq_mouth_closed` | 84% (67% fail) | 2% (1% fail) |
| Dark background | `background_brightness` | 97% (all fail) | 2% of light backgrounds (none fail) |
| Patterned background (rule) | `ofiq_background`, `background_consistency`, `background_pattern` | 80% (68% fail) | 34% of plain backgrounds (25% fail) |

The eye thresholds come from this set: `ofiq_eyes_visible` at 90/50 flagged 61% of clear glasses, and `ofiq_eyes_open` at 80/50 failed only 8% of closed eyes. A glasses frame across the eyes now warns in only 15% of cases (13 photos, most borderline). Reflections on glasses are the weak spot: the check only sees blown-out white spots. Red eyes had too few real cases to measure (2).

The background rows use a second set of labels (`vlm_background_labels.jsonl`: plain, patterned or objects, and light, medium or dark). `background_pattern` takes the median colour variance of the background's patches, so a faint pattern spread across it (a flag's emblem) fails, while hair straying into a few patches does not; it skips when fewer than 6 clean patches are visible (a third of these tight crops). Blurred objects behind the person are the weak spot: 39% of them pass `background_no_objects`, against 21% of sharp ones.

**With SigLIP enabled** (warn only, thresholds set on the `calibrate` half; `notebooks/real_photos.ipynb`, section 4):

| Rule, on Open Images | Without SigLIP | With SigLIP |
| --- | --- | --- |
| No reflections on glasses (clear glasses, 19 with reflections) | 16% caught, 6% false warnings | 37% caught, 15% false warnings |
| Eyes look into the camera (66 looking away) | 56% caught, 17% false warnings | 80% caught, 28% false warnings |

Most false warnings here fall on photos that fail other rules anyway. On the London Set, neither SigLIP check warned on any of the 306 photos that should pass, so the numbers above are unchanged. Of four zero-shot models tried (`scripts/clip_zero_shot.py`), SigLIP SO400M was the only one never worse than our checks; it misses the drawn-in reflections of the London Set, which the pixel check catches.

## Report Structure

```json
{
  "is_valid": false,
  "overall_score": 39.0,
  "issues": [
    {
      "category": "error",
      "message": "Face is too small (39.3% of image). Should be 70-80%.",
      "severity": 9
    }
  ],
  "measurements": {
    "face_height_ratio": 0.393,
    "sharpness": 45.2,
    "horizontal_offset": 0.023,
    "background_uniformity": 52.3,
    "overall_brightness": 87.4
  },
  "bbox": {
    "x": 375,
    "y": 244,
    "width": 302,
    "height": 436
  },
  "suggested_crop": {
    "x": 275,
    "y": 144,
    "width": 402,
    "height": 536,
    "reason": "Face is too small - crop to this region"
  },
  "checks": ["… one entry per check: name, status, reasons, measurements"],
  "rules": ["… one entry per rule: id, title, status, checks, coverage, reasons"],
  "skipped_rules": ["no_hat"],
  "crop": {
    "status": "proposed",
    "reason": null,
    "x": 237, "y": 0, "width": 1556, "height": 2000,
    "is_valid": true,
    "rules_after": {"photo_format": "pass", "…": "one status per rule, on the cropped photo"},
    "fixed_rules": ["photo_format"],
    "remaining_rules": []
  },
  "summary": {
    "total_issues": 8,
    "errors": 5,
    "warnings": 3
  }
}
```

**`crop`** is the fix for framing problems (`core/crop.py`). When `photo_format`, `face_size` or `face_centered` warns or fails, the validator plans a 35x45 box inside the photo that makes the head 72-79% of the height (the rule allows 70-80%), centres the face and puts it at the usual height. It keeps the head size the photo has when that is allowed. Then it runs every check again on the cropped photo. `status` is one of:

- `not_needed`: no framing check warns or fails.
- `not_possible`: no box fits inside the photo, the head can't be measured, or the crop would be below 413x531 px; `reason` says which.
- `proposed`: the box, whether the cropped photo is valid (`is_valid`), each rule's status on it (`rules_after`), the rules it fixes (`fixed_rules`) and the ones that still fail or are skipped (`remaining_rules`), which need a new photo.

Every key is present whatever the status; the ones that don't apply are `null` (`reason`, the box, `is_valid`) or empty (`rules_after`, the rule lists).

It never adds background and never changes `is_valid` of the photo as sent; it only says what a crop would give. Photos with framing problems take about twice as long, since the checks run twice; `AUTOCROP=false` turns it off (then `crop` is `null`). `suggested_crop` is the older suggestion (3:4, face at 40%) and stays for compatibility; use `crop`.

## Benefits of Modular Architecture

1. **Separation of Concerns**
   - Core logic is framework-agnostic
   - Easy to add new interfaces (GUI, mobile app, etc.)
   - Testable in isolation

2. **Reusability**
   - Use the same validator in CLI, API, or scripts
   - Import only what you need

3. **Maintainability**
   - Changes to validation logic don't affect interfaces
   - Each module has a single responsibility

4. **Extensibility**
   - Add new validation checks in `core/validator.py`
   - Add new CLI commands in `cli.py`
   - Add new API endpoints in `check_api.py`

## Development

**Run tests:**
```bash
uv run --extra test pytest
```

## Dependencies

- **Core:** `opencv-python`, `mediapipe`, `numpy`, `pillow`; OFIQ (optional, built from source by `install.sh`, see the README)
- **CLI:** `typer[all]`, `rich`
- **API:** `fastapi`, `uvicorn`, `python-multipart`

## Standards

Validates against the German ID photo rules (Fotomustertafel), using ISO/IEC 29794-5 (face image quality, via OFIQ) for most measurements.

