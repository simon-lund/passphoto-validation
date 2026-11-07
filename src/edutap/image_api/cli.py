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
from rich.panel import Panel
from rich.table import Table

from edutap.image_api.core.utils import prepare_image
from edutap.image_api.core import PassportPhotoValidator, ValidatorConfig

app = typer.Typer(
    name="passport-validator",
    help="Validate passport photos against ICAO/ISO 19794-5 standards",
    add_completion=False,
)
console = Console()


def load_image(image_path: Path) -> np.ndarray:
    """Load and convert image to BGR format for OpenCV."""
    try:
        image = Image.open(image_path)
        return prepare_image(image)
    except Exception as e:
        console.print(f"[red]Error loading image:[/red] {e}")
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
    confidence: float = typer.Option(
        0.5,
        "--confidence",
        "-c",
        min=0.0,
        max=1.0,
        help="Minimum face detection confidence (0.0-1.0)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed measurements"
    ),
):
    """
    Validate a passport photo against ICAO standards.

    Returns a detailed report with errors, warnings, and recommendations.
    """
    console.print(f"\n[cyan]Validating:[/cyan] {image_path}")

    # Load image
    with console.status("[bold cyan]Loading image..."):
        image_np = load_image(image_path)

    # Validate
    with console.status("[bold cyan]Analyzing photo..."):
        validator = PassportPhotoValidator(ValidatorConfig(min_detection_confidence=confidence))
        report = validator(image_np)

    # Display results
    _display_report(report, verbose)

    # Save JSON if requested
    if output:
        report_dict = report.to_dict()
        with open(output, 'w') as f:
            json.dump(report_dict, f, indent=2)
        console.print(f"\n[green]Report saved to:[/green] {output}")

    # Exit with appropriate code
    sys.exit(0 if report.is_valid else 1)


def _display_report(report, verbose: bool):
    """Display validation report in a nice format."""

    # Header with score
    if report.is_valid:
        status = "[bold green]✓ VALID[/bold green]"
        status_color = "green"
    else:
        status = "[bold red]✗ INVALID[/bold red]"
        status_color = "red"

    console.print(Panel(
        f"{status}\n\n[bold]Overall Score:[/bold] {report.overall_score:.1f}/100",
        title="Validation Result",
        border_style=status_color,
        box=box.ROUNDED
    ))

    # Issues table
    if report.issues:
        console.print(f"\n[bold]Found {len(report.issues)} issue(s):[/bold]\n")

        table = Table(show_header=True, header_style="bold", box=box.SIMPLE)
        table.add_column("Type", style="cyan", width=8)
        table.add_column("Severity", justify="center", width=8)
        table.add_column("Issue", style="white")

        # Sort by severity (highest first)
        sorted_issues = sorted(report.issues, key=lambda x: x.severity, reverse=True)

        for issue in sorted_issues:
            if issue.category == 'error':
                type_str = "[red]ERROR[/red]"
            elif issue.category == 'warning':
                type_str = "[yellow]WARNING[/yellow]"
            else:
                type_str = "[blue]INFO[/blue]"

            severity_bar = "█" * issue.severity
            table.add_row(type_str, severity_bar, issue.message)

        console.print(table)
    else:
        console.print("\n[bold green]✓ No issues found![/bold green]")

    # Rules that could not be judged keep a photo from being valid
    skipped = [rule for rule in report.rules if rule.id in report.skipped_rules]
    if skipped:
        console.print(f"\n[bold yellow]{len(skipped)} rule(s) could not be checked:[/bold yellow]")
        for rule in skipped:
            console.print(f"  {rule.title}: {rule.reasons[0] if rule.reasons else 'no check ran'}")

    # Suggested crop
    if report.suggested_crop:
        console.print(f"\n[bold yellow]💡 Suggestion:[/bold yellow]")
        crop = report.suggested_crop
        console.print(f"  {crop['reason']}")
        console.print(f"  Crop region: x={crop['x']}, y={crop['y']}, "
                      f"width={crop['width']}, height={crop['height']}")

    # A 35x45 crop for framing problems, checked on the cropped photo
    crop = report.crop
    if crop and crop.status == 'proposed':
        verdict = "[green]valid[/green]" if crop.is_valid else f"still fails {len(crop.remaining_rules)} rule(s)"
        console.print(f"\n[bold]Crop to 35x45:[/bold] x={crop.x}, y={crop.y}, width={crop.width}, height={crop.height}")
        console.print(f"  Fixes {len(crop.fixed_rules)} rule(s); the cropped photo is {verdict}.")
    elif crop and crop.status == 'not_possible':
        console.print(f"\n[bold]Crop to 35x45:[/bold] not possible. {crop.reason}")

    # Measurements (verbose mode)
    if verbose and report.measurements:
        console.print("\n[bold]Measurements:[/bold]")
        for key, value in report.measurements.items():
            console.print(f"  • {key}: {value}")


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
    confidence: float = typer.Option(
        0.5,
        "--confidence",
        "-c",
        min=0.0,
        max=1.0,
        help="Minimum face detection confidence"
    ),
):
    """
    Validate multiple passport photos in a directory against ICAO standards.

    Processes all images matching the pattern and generates a summary.
    """
    # Find all matching images
    image_files = list(input_dir.glob(pattern))

    if not image_files:
        console.print(f"[red]No images found matching pattern '{pattern}' in {input_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[cyan]Found {len(image_files)} image(s) to validate[/cyan]\n")

    # Create output directory if needed
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    validator = PassportPhotoValidator(ValidatorConfig(min_detection_confidence=confidence))

    # Process each image
    results = []
    valid_count = 0

    for image_path in image_files:
        console.print(f"[cyan]Processing:[/cyan] {image_path.name}")
        image_np = load_image(image_path)

        try:
            report = validator(image_np)


            if report.is_valid:
                console.print(f"  [green]✓ Valid (score: {report.overall_score:.1f})[/green]")
                valid_count += 1
            else:
                console.print(f"  [red]✗ Invalid (score: {report.overall_score:.1f})[/red]")
                console.print(f"    {len([i for i in report.issues if i.category == 'error'])} errors, "
                              f"{len([i for i in report.issues if i.category == 'warning'])} warnings, "
                              f"{len(report.skipped_rules)} rules not checked")

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
            console.print(f"  [red]Error: {e}[/red]")
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
        console.print(f"\n[green]Reports saved to:[/green] {output_dir}")


@app.command()
def version():
    """Show version information."""
    console.print("[bold]Passport Photo Validator[/bold]")
    console.print("Standards: ICAO/ISO 19794-5")
    console.print("Version: 1.0.0")
