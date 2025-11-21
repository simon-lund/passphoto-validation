# Passport Photo Validator - Modular Architecture

## Overview

The passport photo validator has been refactored into a modular architecture with clear separation of concerns:

```
src/edutap/image_api/
├── core/                    # Business logic (pure)
│   ├── __init__.py
│   └── validator.py         # Validation engine
├── cli.py                   # CLI interface (Typer)
├── main2.py                 # API interface (FastAPI)
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
- ICAO/ISO 19794-5 standards compliance
- 13 validation checks (face detection, background, lighting, shadows, sharpness, etc.)
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

### 3. **API Interface** (`main2.py`)
FastAPI endpoint that uses the core validation logic.

**Endpoint:**
```
POST /validate-passport-photo
```

**Usage:**
```bash
# Start the API server
uvicorn edutap.image_api.main2:app --reload --host 0.0.0.0 --port 8000

# Test with curl
curl -X POST "http://localhost:8000/validate-passport-photo" \
  -F "file=@photo.jpg"
```

## Installation

```bash
# With uv (recommended)
uv pip install -e ".[dev,test,typecheck]"

# With pip
pip install -e ".[dev,test,typecheck]"
```

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
import cv2
from edutap.image_api.core import PassportPhotoValidator

# Load image
image = cv2.imread("photo.jpg")

# Validate
validator = PassportPhotoValidator(image, min_detection_confidence=0.5)
report = validator.validate()

# Check results
if report.is_valid:
    print(f"✓ Valid! Score: {report.overall_score}/100")
else:
    print(f"✗ Invalid. Found {len(report.issues)} issues:")
    for issue in report.issues:
        print(f"  - [{issue.category}] {issue.message}")

# Get measurements
print(f"Face size: {report.measurements['face_height_ratio']:.1%}")
print(f"Sharpness: {report.measurements['sharpness']:.1f}")

# Export to JSON
report_dict = report.to_dict()
```

### API Usage

**1. Start server:**
```bash
uvicorn edutap.image_api.main2:app --reload --host 0.0.0.0 --port 8000
```

**2. Call endpoint:**
```python
import requests

url = "http://localhost:8000/validate-passport-photo"
files = {"file": open("photo.jpg", "rb")}
response = requests.post(url, files=files)
report = response.json()

print(f"Valid: {report['is_valid']}")
print(f"Score: {report['overall_score']}")
```

## Validation Checks

The validator performs 13 comprehensive checks:

1. ✓ **Face Detection** - Ensures exactly one face is present
2. ✓ **Face Position** - Checks centering and vertical placement
3. ✓ **Face Size** - Validates 70-80% of image height (ICAO standard)
4. ✓ **Background** - Uniform, light-colored background
5. ✓ **Lighting** - Overall brightness and contrast
6. ✓ **Shadows** - Detects harsh shadows on face
7. ✓ **Sharpness** - Ensures image is not blurry
8. ✓ **Resolution** - Minimum 600x600 pixels
9. ✓ **Eyes Visibility** - Both eyes clearly visible
10. ✓ **Head Pose** - Looking straight at camera
11. ✓ **Expression** - Neutral expression, mouth closed
12. ✓ **Hair Obscuring** - Face fully visible
13. ✓ **Glasses Glare** - No reflections on glasses

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
  "summary": {
    "total_issues": 8,
    "errors": 5,
    "warnings": 3
  }
}
```

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
   - Add new API endpoints in `main2.py`

## Development

**Run tests:**
```bash
pytest tests/
```

**Type checking:**
```bash
mypy src/
```

**Lint:**
```bash
pre-commit run --all-files
```

## Dependencies

- **Core:** `opencv-python`, `mediapipe`, `numpy`, `pillow`
- **CLI:** `typer[all]`, `rich`
- **API:** `fastapi`, `uvicorn`, `python-multipart`

## Standards

Validates against **ICAO/ISO 19794-5** international passport photo standards.

