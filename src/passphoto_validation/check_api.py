"""HTTP API that checks a photo against the German ID photo rules.

Run with `uv run passport-validator-api` (http://127.0.0.1:9600, docs at /docs).

/v1/check answers in JSON unless the request's Accept header prefers text/html, which gets the same
report as one self-contained HTML page (report_html.py) for people to read.

Every error response is `{"code": "<stable_snake_case>", "detail": "<human text>"}`, whatever the Accept
header says. Clients branch on the status and `code` only; the detail text may change at any time.
"""
import io
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from . import report_html
from .api_schema import CheckReport, ErrorResponse
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


@app.exception_handler(Exception)
async def handle_crash(request: Request, exc: Exception):
    # A bug in the pipeline. Starlette still re-raises it after this answer, so the server logs the traceback;
    # the client gets a code and no internals
    return JSONResponse(status_code=500, content={
        'code': 'internal_error',
        'detail': 'The service hit an error and could not check the photo. If it keeps happening, report it.',
    })


def _groups():
    return [{'id': group_id, 'title': title} for group_id, title in GROUPS.items()]


def prefers_html(accept: Optional[str]) -> bool:
    """Whether an Accept header asks for text/html over application/json; ties and no header mean JSON.

    Each media type takes the q of the most specific range that matches it (text/html, then text/*,
    then */*), as in RFC 9110. So `*/*` gets JSON and a browser's `text/html,...,*/*;q=0.8` gets HTML.
    """
    if not accept:
        return False
    ranges = []
    for part in accept.split(','):
        media_range, *params = (piece.strip() for piece in part.split(';'))
        q = 1.0
        for param in params:
            name, _, value = param.partition('=')
            if name.strip().lower() == 'q':
                try:
                    q = float(value)
                except ValueError:
                    q = 0.0
        ranges.append((media_range.lower(), q))

    def quality(media_type):
        main_type = media_type.split('/')[0]
        for candidate in (media_type, f'{main_type}/*', '*/*'):
            matches = [q for media_range, q in ranges if media_range == candidate]
            if matches:
                return max(matches)
        return 0.0

    html = quality('text/html')
    return html > 0 and html > quality('application/json')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/v1/rules')
def list_rules():
    """The rule catalogue: every rule, the checks that cover it, and how well."""
    return {'groups': _groups(), 'rules': [asdict(rule) for rule in RULES]}


def _error(description: str):
    return {'model': ErrorResponse, 'description': description}


# The models only describe the answers for OpenAPI; the report dict is sent as it is (see api_schema.py)
@app.post('/v1/check', responses={
    200: {'model': CheckReport, 'content': {'text/html': {}},
          'description': 'The report: JSON by default, an HTML page when the Accept header prefers text/html.'},
    400: _error('image_empty, or invalid_image: not an image we can read.'),
    413: _error(f'image_too_large: over {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.'),
    422: _error('invalid_request: the multipart field `file` is missing.'),
    500: _error('internal_error: a bug in the pipeline; the traceback is in the server log.'),
})
def check_photo(request: Request, response: Response,
                file: UploadFile = File(..., description='The photo (JPEG, PNG, WebP, ...)')):
    """Check one photo. Returns every check (pass, warn, fail, skipped, error) and every rule.

    Send `Accept: text/html` to get the report as a self-contained HTML page instead of JSON.
    """
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

    payload = {
        **report.to_dict(),
        'groups': _groups(),
        'image': {'width': image_np.shape[1], 'height': image_np.shape[0]},
    }
    # The same URL answers in two formats, so caches must key on Accept
    if prefers_html(request.headers.get('accept')):
        image_rgb = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        return HTMLResponse(report_html.render_report(payload, image_rgb, file.filename), headers={'Vary': 'Accept'})
    response.headers['Vary'] = 'Accept'
    return payload


def main():
    uvicorn.run(
        'passphoto_validation.check_api:app',
        host=os.environ.get('PASSPORT_API_HOST', '127.0.0.1'),
        port=int(os.environ.get('PASSPORT_API_PORT', '9600')),
    )


if __name__ == '__main__':
    main()
