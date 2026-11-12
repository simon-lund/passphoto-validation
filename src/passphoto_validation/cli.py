"""CLI for passport photo validation."""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from PIL import Image
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from passphoto_validation.core.utils import prepare_image
from passphoto_validation.core import PassportPhotoValidator, ValidatorConfig
from passphoto_validation.core.thresholds import ThresholdsFileError, to_toml

app = typer.Typer(
    name="passport-validator",
    help="Check photos against the German ID photo rules (the Fotomustertafel)",
    add_completion=False,
)
console = Console()

THRESHOLDS_OPTION = typer.Option(
    None,
    "--thresholds",
    "-t",
    exists=True,
    dir_okay=False,
    help="A TOML file that overrides check parameters (start from `passport-validator thresholds`); "
         "without it, the file THRESHOLDS_FILE names, if set",
)
CONFIDENCE_OPTION = typer.Option(
    None,
    "--confidence",
    "-c",
    min=0.0,
    max=1.0,
    help="Minimum face detection confidence (0.0-1.0); without it, MIN_DETECTION_CONFIDENCE or 0.5",
)


def _config(confidence: Optional[float] = None, thresholds: Optional[Path] = None) -> ValidatorConfig:
    """The configuration for the CLI's options. An option that is not given is left out, so the environment
    and .env decide it (THRESHOLDS_FILE, MIN_DETECTION_CONFIDENCE), as they do for the API: a value passed
    to ValidatorConfig, None included, would override them."""
    given = {}
    if confidence is not None:
        given['min_detection_confidence'] = confidence
    if thresholds is not None:
        given['thresholds_file'] = str(thresholds)
    return ValidatorConfig(**given)


def _thresholds_error(exc: ThresholdsFileError):
    """End the command with the message of a bad thresholds file. Escaped: the message names the check as
    [check], which Rich would read as markup and drop."""
    console.print(f"[red]{escape(str(exc))}[/red]")
    raise typer.Exit(2)


def _validator(confidence: Optional[float], thresholds: Optional[Path]) -> PassportPhotoValidator:
    """The validator for the CLI's options; a bad thresholds file ends the command with its message."""
    try:
        validator = PassportPhotoValidator(_config(confidence, thresholds))
    except ThresholdsFileError as exc:
        _thresholds_error(exc)
    if validator.config.thresholds_file:
        console.print(f"[dim]Thresholds file: {escape(validator.config.thresholds_file)}[/dim]")
    return validator


def load_image(image_path: Path) -> np.ndarray:
    """Load and convert image to BGR format for OpenCV."""
    try:
        image = Image.open(image_path)
        return prepare_image(image)
    except Exception as e:
        console.print(f"[red]Error loading image:[/red] {escape(str(e))}")
        raise typer.Exit(1)


@app.command()
def validate(
    image_path: Path = typer.Argument(
        ...,  # required
        exists=True,
        dir_okay=False,
        help="Path to the passport photo to validate"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save report as JSON to this file"
    ),
    confidence: Optional[float] = CONFIDENCE_OPTION,
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed measurements"
    ),
    thresholds: Optional[Path] = THRESHOLDS_OPTION,
):
    """
    Check a photo against the German ID photo rules.

    Returns a detailed report with errors, warnings, and recommendations.
    """
    console.print(f"\n[cyan]Validating:[/cyan] {escape(str(image_path))}")

    # Load image
    with console.status("[bold cyan]Loading image..."):
        image_np = load_image(image_path)

    # Validate
    with console.status("[bold cyan]Analyzing photo..."):
        validator = _validator(confidence, thresholds)
        report = validator(image_np)

    # Display results
    _display_report(report, verbose)

    # Save JSON if requested
    if output:
        report_dict = report.to_dict()
        with open(output, 'w') as f:
            json.dump(report_dict, f, indent=2)
        console.print(f"\n[green]Report saved to:[/green] {escape(str(output))}")

    # Exit with appropriate code
    sys.exit(0 if report.is_valid else 1)


def _not_checked(report) -> list:
    """The rules no check judged (their checks are turned off in the thresholds file, or none covers them).
    They never block validity, so the output has to say that they were left out."""
    return [rule for rule in report.rules if rule.status == 'not_checked']


def _display_report(report, verbose: bool):
    """Display validation report in a nice format."""

    # Header with score
    if report.is_valid:
        status = "[bold green]✓ VALID[/bold green]"
        status_color = "green"
    else:
        status = "[bold red]✗ INVALID[/bold red]"
        status_color = "red"
    not_checked = _not_checked(report)
    left_out = (f"\n[yellow]{len(not_checked)} of {len(report.rules)} rules were not checked[/yellow]"
                if not_checked else "")

    console.print(Panel(
        f"{status}\n\n[bold]Overall Score:[/bold] {report.overall_score:.1f}/100{left_out}",
        title="Validation Result",
        border_style=status_color,
        box=box.ROUNDED
    ))

    # The problems: one row per check that did not pass, with the rules it decided (report.problems)
    titles = {rule.id: rule.title for rule in report.rules}
    if report.problems:
        console.print(f"\n[bold]{len(report.problems)} problem(s):[/bold]\n")
        table = Table(show_header=True, header_style="bold", box=box.SIMPLE)
        table.add_column("Status", width=10)
        table.add_column("Problem", style="white")
        table.add_column("Rules", style="cyan")
        labels = {'fail': '[red]FAIL[/red]', 'warn': '[yellow]WARN[/yellow]', 'skipped': '[yellow]NOT JUDGED[/yellow]',
                  'error': '[magenta]CRASHED[/magenta]'}
        for problem in report.problems:
            rest = ' '.join(problem.reasons)
            rest = rest[len(problem.headline):].strip() if rest.startswith(problem.headline) else rest
            text = f"[bold]{escape(problem.headline)}[/bold]" + (f"\n{escape(rest)}" if rest else '')
            rules = '\n'.join(escape(titles.get(rule, rule)) for rule in problem.rules)
            if problem.not_judged:
                rules += ('\n[dim]not judged because of this: '
                          + ', '.join(escape(titles.get(rule, rule)) for rule in problem.not_judged) + '[/dim]')
            table.add_row(labels.get(problem.status, problem.status.upper()), text, rules)
        console.print(table)
    elif not_checked:
        passed = sum(1 for rule in report.rules if rule.status == 'pass')
        console.print(f"\n[bold green]✓ No problems in the {passed} rules that were checked.[/bold green]")
    else:
        console.print("\n[bold green]✓ Every rule passes.[/bold green]")
    if not_checked:
        console.print(f"\n[bold yellow]{len(not_checked)} rule(s) not checked[/bold yellow] "
                      f"(their checks are turned off in the thresholds file):")
        for rule in not_checked:
            console.print(f"  [dim]{escape(rule.title)}[/dim]")

    # Suggested crop
    if report.suggested_crop:
        console.print(f"\n[bold yellow]💡 Suggestion:[/bold yellow]")
        crop = report.suggested_crop
        console.print(f"  {escape(crop['reason'])}")
        console.print(f"  Crop region: x={crop['x']}, y={crop['y']}, "
                      f"width={crop['width']}, height={crop['height']}")

    # A 35x45 crop for framing problems, checked on the cropped photo
    crop = report.crop
    if crop and crop.status == 'proposed':
        verdict = "[green]valid[/green]" if crop.is_valid else f"still fails {len(crop.remaining_rules)} rule(s)"
        console.print(f"\n[bold]Crop to 35x45:[/bold] x={crop.x}, y={crop.y}, width={crop.width}, height={crop.height}")
        console.print(f"  Fixes {len(crop.fixed_rules)} rule(s); the cropped photo is {verdict}.")
    elif crop and crop.status == 'not_possible':
        console.print(f"\n[bold]Crop to 35x45:[/bold] not possible. {escape(crop.reason)}")

    # Every check, then the measurements (verbose mode)
    if verbose:
        console.print("\n[bold]Checks:[/bold]")
        for check in report.checks:
            reasons = f" {' '.join(check.reasons)}" if check.reasons else ''
            console.print(f"  {check.status:8} {check.name}{escape(reasons)}")
    if verbose and report.measurements:
        console.print("\n[bold]Measurements:[/bold]")
        for key, value in report.measurements.items():
            console.print(f"  • {key}: {escape(str(value))}")


@app.command()
def batch(
    input_dir: Path = typer.Argument(
        ...,  # required
        exists=True,
        file_okay=False,
        help="Directory containing passport photos"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Directory to save JSON reports"
    ),
    pattern: str = typer.Option(
        "*.jpg",
        "--pattern",
        "-p",
        help="File pattern to match (e.g., '*.jpg', '*.png')"
    ),
    confidence: Optional[float] = CONFIDENCE_OPTION,
    thresholds: Optional[Path] = THRESHOLDS_OPTION,
):
    """
    Check every photo in a directory against the German ID photo rules.

    Processes all images matching the pattern and generates a summary.
    """
    # Find all matching images
    image_files = list(input_dir.glob(pattern))

    if not image_files:
        console.print(f"[red]No images found matching pattern '{escape(pattern)}' in {escape(str(input_dir))}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[cyan]Found {len(image_files)} image(s) to validate[/cyan]\n")

    # Create output directory if needed
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    validator = _validator(confidence, thresholds)

    # Process each image
    results = []
    valid_count = 0

    for image_path in image_files:
        console.print(f"[cyan]Processing:[/cyan] {escape(image_path.name)}")
        image_np = load_image(image_path)

        try:
            report = validator(image_np)


            not_checked = len(_not_checked(report))
            left_out = f", {not_checked} rules not checked" if not_checked else ""
            if report.is_valid:
                console.print(f"  [green]✓ Valid (score: {report.overall_score:.1f}{left_out})[/green]")
                valid_count += 1
            else:
                console.print(f"  [red]✗ Invalid (score: {report.overall_score:.1f})[/red]")
                console.print(f"    {len([i for i in report.issues if i.category == 'error'])} errors, "
                              f"{len([i for i in report.issues if i.category == 'warning'])} warnings, "
                              f"{len(report.skipped_rules)} rules not judged{left_out}")

            results.append({
                'filename': image_path.name,
                'valid': report.is_valid,
                'score': report.overall_score,
                'issues': len(report.issues)
            })

            # Save individual report if output_dir specified
            if output_dir:
                report_path = output_dir / f"{image_path.stem}_report.json"
                with open(report_path, 'w') as f:
                    json.dump(report.to_dict(), f, indent=2)

        except Exception as e:
            console.print(f"  [red]Error: {escape(str(e))}[/red]")
            results.append({
                'filename': image_path.name,
                'valid': False,
                'score': 0,
                'error': str(e)
            })

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total: {len(image_files)}")
    console.print(f"  [green]Valid: {valid_count}[/green]")
    console.print(f"  [red]Invalid: {len(image_files) - valid_count}[/red]")

    if output_dir:
        console.print(f"\n[green]Reports saved to:[/green] {escape(str(output_dir))}")


@app.command()
def thresholds(thresholds: Optional[Path] = THRESHOLDS_OPTION):
    """Print every parameter of every check with its current value, as a TOML thresholds file.

    Save it, keep the lines to change, and pass the file back with --thresholds (or set THRESHOLDS_FILE,
    which the CLI, the API and the review page all read). The values are the ones in effect, so with
    THRESHOLDS_FILE set they include that file's. warn_... and fail_... are the levels, false turns one off;
    enabled = false turns a check off.
    """
    try:
        config = _config(thresholds=thresholds)
        checks = PassportPhotoValidator.registered_checks(config)
        disabled = PassportPhotoValidator.disabled_checks(config)
    except ThresholdsFileError as exc:
        _thresholds_error(exc)
    typer.echo(to_toml(checks, disabled))


@app.command()
def version():
    """Show version information."""
    from importlib.metadata import version as installed_version
    console.print(f"[bold]passphoto-validation[/bold] {installed_version('passphoto-validation')}")
    console.print("Rules: the German Fotomustertafel, measured largely with ISO/IEC 29794-5 (OFIQ)")
