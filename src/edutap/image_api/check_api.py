"""HTTP API that checks a photo against the German ID photo rules.

Run with `uv run passport-validator-api` (http://127.0.0.1:9600, docs at /docs).

Every error response is `{"code": "<stable_snake_case>", "detail": "<human text>"}`. Clients branch on
the status and `code` only; the detail text may change at any time.
"""
import io
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from .core import PassportPhotoValidator
from .core.rules import GROUPS, RULES
from .core.utils import prepare_image

MAX_UPLOAD_BYTES = 15 * 1024 * 1024


class ApiError(Exception):
    """An error response with a stable machine-readable code next to the human-readable detail."""

    def __init__(self, status_code: int, code: str, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Building the validator loads the MediaPipe models, so do it once
    app.state.validator = PassportPhotoValidator()
    # MediaPipe graphs must not process two images at the same time
    app.state.validator_lock = threading.Lock()
    yield


app = FastAPI(
    title="Passport photo validation API",
    description="Checks a photo against the German ID photo rules (Fotomustertafel).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(ApiError)
async def handle_api_error(request: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status_code, content={'code': exc.code, 'detail': exc.detail})


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    fields = ', '.join('.'.join(str(part) for part in error['loc']) for error in exc.errors())
    return JSONResponse(status_code=422, content={'code': 'invalid_request', 'detail': f'Invalid request: {fields}'})


def _groups():
    return [{'id': group_id, 'title': title} for group_id, title in GROUPS.items()]


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/v1/rules')
def list_rules():
    """The rule catalogue: every rule, the checks that cover it, and how well."""
    return {'groups': _groups(), 'rules': [asdict(rule) for rule in RULES]}


@app.post('/v1/check')
def check_photo(request: Request, file: UploadFile = File(..., description='The photo (JPEG, PNG, WebP, ...)')):
    """Check one photo. Returns every check (pass, warn, fail, skipped, error) and every rule."""
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ApiError(413, 'image_too_large', f'The photo must be at most {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.')
    if not contents:
        raise ApiError(400, 'image_empty', 'The uploaded file is empty.')

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise ApiError(400, 'invalid_image', 'The uploaded file is not an image we can read.')

    image_np = prepare_image(image)
    with request.app.state.validator_lock:
        report = request.app.state.validator(image_np)

    return {
        **report.to_dict(),
        'groups': _groups(),
        'image': {'width': image_np.shape[1], 'height': image_np.shape[0]},
    }


def main():
    uvicorn.run(
        'edutap.image_api.check_api:app',
        host=os.environ.get('PASSPORT_API_HOST', '127.0.0.1'),
        port=int(os.environ.get('PASSPORT_API_PORT', '9600')),
    )


if __name__ == '__main__':
    main()
