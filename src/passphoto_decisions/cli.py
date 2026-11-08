"""Command line: check one photo, run an evaluation set, list the rules.

    uv run passphoto-decisions check photo.jpg [--json] [--dry-run]
    uv run passphoto-decisions batch <set> --yes
    uv run passphoto-decisions rules
"""
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import request as req
from .report import Thresholds, error_row, interpret, results_row
from .rules import GROUPS, RULES

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)
console = Console()

PRICE_PER_MILLION_INPUT_TOKENS = 0.10  # USD, gpt-6-luna on /v1/decisions, uncached

MODEL_OPTION = typer.Option(None, '--model', help='Decisions model (default: PASSPHOTO_MODEL or gpt-6-luna).')
DETAIL_OPTION = typer.Option(req.DETAIL, '--detail', help='Image detail level: low, high, auto or original.')
MAX_SIDE_OPTION = typer.Option(None, '--max-side', help='Shrink the photo to this many pixels on its long side before sending.')
WARN_OPTION = typer.Option(Thresholds.warn_at, '--warn-at', help='A rule warns from this probability that it is broken.')
FAIL_OPTION = typer.Option(Thresholds.fail_at, '--fail-at', help='A rule fails from this probability that it is broken.')


def make_client():
    """An OpenAI client; reads OPENAI_API_KEY from the environment or from a .env file in the working directory.
    Replaced in tests."""
    from dotenv import find_dotenv, load_dotenv
    from openai import OpenAI
    load_dotenv(find_dotenv(usecwd=True))  # the .env of the folder the command runs in, not of this file's
    if not os.environ.get('OPENAI_API_KEY'):
        raise typer.BadParameter('OPENAI_API_KEY is not set (export it, or put it in a .env file next to pyproject.toml).')
    return OpenAI(max_retries=3)


def _model(model: Optional[str]) -> str:
    return model or os.environ.get('PASSPHOTO_MODEL') or req.MODEL


def _cost(input_tokens: int) -> float:
    return input_tokens / 1_000_000 * PRICE_PER_MILLION_INPUT_TOKENS


@app.command()
def check(
    photo: Path = typer.Argument(..., exists=True, dir_okay=False, help='The photo to check.'),
    json_out: bool = typer.Option(False, '--json', help='Print the report as JSON instead of a table.'),
    dry_run: bool = typer.Option(False, '--dry-run', help='Print the request (without the image bytes) and send nothing.'),
    model: Optional[str] = MODEL_OPTION,
    detail: str = DETAIL_OPTION,
    max_side: Optional[int] = MAX_SIDE_OPTION,
    warn_at: float = WARN_OPTION,
    fail_at: float = FAIL_OPTION,
):
    """Check one photo against the 36 rules and print each rule's status and probability."""
    loaded = req.load_photo(photo, max_side)
    request = req.build_request(loaded, _model(model), detail)
    if dry_run:
        print(json.dumps(req.redacted(request), indent=2))
        return

    decision = req.send(make_client(), request)
    report = interpret(decision, loaded, Thresholds(warn_at, fail_at))
    if json_out:
        print(json.dumps(report.to_dict(), indent=2))
        return

    table = Table(title=f'{report.file}: {"valid" if report.is_valid else "not valid"}', show_lines=False)
    table.add_column('Rule')
    table.add_column('Group')
    table.add_column('Status')
    table.add_column('P(broken)', justify='right')
    styles = {'pass': 'green', 'warn': 'yellow', 'fail': 'red', 'skipped': 'dim'}
    for rule in report.rules:
        probability = '' if rule.probability is None else f'{rule.probability:.0%}'
        table.add_row(rule.id, GROUPS[rule.group], f'[{styles[rule.status]}]{rule.status}[/]', probability)
    console.print(table)
    gate = report.gate
    console.print(f'Faces: {gate.choice or "?"} ({gate.status})' + (f', {gate.reason}' if gate.reason else ''))
    console.print(f'{report.usage["input_tokens"]} input tokens, about ${_cost(report.usage["input_tokens"]):.4f}')


@app.command()
def batch(
    set_dir: Path = typer.Argument(..., exists=True, file_okay=False, help='Folder with manifest.csv and the photos.'),
    out: str = typer.Option('results_decisions.jsonl', '--out', help='Output file inside the set folder.'),
    limit: Optional[int] = typer.Option(None, '--limit', help='Only the first N photos of the manifest.'),
    workers: int = typer.Option(4, '--workers', help='Parallel requests.'),
    yes: bool = typer.Option(False, '--yes', help='Send the photos without asking.'),
    model: Optional[str] = MODEL_OPTION,
    detail: str = DETAIL_OPTION,
    max_side: Optional[int] = MAX_SIDE_OPTION,
    warn_at: float = WARN_OPTION,
    fail_at: float = FAIL_OPTION,
):
    """Send every photo of an evaluation set and write results.jsonl in passphoto-validation's format.

    Photos already in the output file are skipped, so an interrupted run continues where it stopped.
    """
    with open(set_dir / 'manifest.csv', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if limit:
        rows = rows[:limit]
    out_path = set_dir / out
    done = set()
    if out_path.exists():
        with open(out_path) as handle:
            done = {json.loads(line)['file'] for line in handle if line.strip()}
    todo = [row for row in rows if row['file'] not in done]

    console.print(f'{len(todo)} of {len(rows)} photos from {set_dir} still to send to OpenAI ({_model(model)}); '
                  f'results go to {out_path}' + (f'; {len(done)} already there' if done else ''))
    if not todo:
        return
    if not yes:
        console.print('Every photo leaves this machine. Run again with --yes to send them.')
        raise typer.Exit(code=2)

    client = make_client()
    thresholds = Thresholds(warn_at, fail_at)

    def run(row):
        try:
            loaded = req.load_photo(set_dir / row['file'], max_side)
            decision = req.send(client, req.build_request(loaded, _model(model), detail))
            return results_row(interpret(decision, loaded, thresholds), row)
        except Exception as exc:  # noqa: BLE001 - the row records the failure, the batch goes on
            return error_row(row, f'{type(exc).__name__}: {exc}')

    tokens = errors = 0
    with ThreadPoolExecutor(workers) as pool, open(out_path, 'a') as handle:
        for index, result in enumerate(pool.map(run, todo), 1):
            handle.write(json.dumps(result) + '\n')
            handle.flush()
            tokens += result['usage'].get('input_tokens', 0)
            errors += result['checks']['face_detected']['status'] == 'error'
            if index % 50 == 0 or index == len(todo):
                console.print(f'{index}/{len(todo)}  {tokens} input tokens so far, about ${_cost(tokens):.2f}')
    console.print(f'Done: {len(todo)} photos, {errors} failed requests, {tokens} input tokens (about ${_cost(tokens):.2f}) -> {out_path}')
    if errors:
        console.print('Failed photos have an error on face_detected; run again to retry them after deleting their lines.')


@app.command()
def rules():
    """List the rules and the question each one asks."""
    table = Table()
    table.add_column('Rule')
    table.add_column('Group')
    table.add_column('Question (yes = broken)')
    for rule in RULES:
        table.add_row(rule.id, GROUPS[rule.group], rule.question)
    console.print(table)


def main():
    app()


if __name__ == '__main__':
    sys.exit(main())
