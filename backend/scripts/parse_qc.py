"""Parse quality report for one document.

A bad parse breaks everything downstream silently: two-column pages whose lines get
interleaved still produce text, still embed, and still return confident answers. This
script makes the parse inspectable before anything is indexed.

    python scripts/parse_qc.py --arxiv 1810.04805
    python scripts/parse_qc.py --file data/documents/uploads/paper.pdf

Reported: block counts per type, the section outline, page coverage, and blocks that
look wrong (reading order jumping back a page, paragraphs too short to be real).
A visual bounding-box overlay is the next step, once real papers have been run through.
"""

import json
from collections import Counter
from pathlib import Path

import typer

from abstractrag.core.config import get_settings
from abstractrag.core.logging import setup_logging
from abstractrag.rag.ingestion.base import SourceInput
from abstractrag.rag.ingestion.resolver import SourceResolver
from abstractrag.rag.models import BlockType, ParsedDocument

app = typer.Typer(help="Inspect how a document was parsed.", no_args_is_help=True)

# Below this, a "paragraph" is usually a fragment left over from a layout mistake.
SHORT_PARAGRAPH_CHARS = 40


@app.command()
def main(
    arxiv: str = typer.Option(None, "--arxiv", help="arXiv ID to download and parse"),
    file: Path = typer.Option(None, "--file", exists=True, help="Local PDF to parse"),
    wiki: str = typer.Option(None, "--wiki", help="Wikipedia article title or URL"),
    json_out: Path = typer.Option(None, "--json", help="Also write the report as JSON"),
) -> None:
    setup_logging()
    settings = get_settings()
    figures_dir = settings.ingestion.resolved_storage_dir() / "figures"
    resolver = SourceResolver(settings.ingestion, settings.llm.base_url, figures_dir)
    document = resolver.resolve(
        SourceInput(
            arxiv_id=arxiv,
            wikipedia=wiki,
            file_name=file.name if file else None,
            file_bytes=file.read_bytes() if file else None,
        )
    )

    report = build_report(document)
    _print_report(document, report)
    if json_out:
        json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"\nwrote {json_out}")


def build_report(document: ParsedDocument) -> dict:
    pages = [block.page for block in document.blocks if block.page is not None]
    return {
        "title": document.title,
        "origin": document.origin,
        "blocks": len(document.blocks),
        "block_types": {
            block_type.value: count
            for block_type, count in Counter(
                block.block_type for block in document.blocks
            ).items()
        },
        "pages": {"first": min(pages), "last": max(pages), "seen": len(set(pages))}
        if pages
        else None,
        "sections": _outline(document),
        "warnings": _warnings(document),
    }


def _outline(document: ParsedDocument) -> list[str]:
    outline: list[str] = []
    for block in document.blocks:
        path = " > ".join(block.section_path)
        if path and (not outline or outline[-1] != path):
            outline.append(path)
    return outline


def _warnings(document: ParsedDocument) -> list[str]:
    warnings: list[str] = []
    previous_page = 0

    for position, block in enumerate(document.blocks):
        if block.page is not None:
            if block.page < previous_page:
                warnings.append(
                    f"block {position} jumps back to page {block.page} after {previous_page} "
                    "(reading order may be wrong)"
                )
            previous_page = block.page

        if (
            block.block_type is BlockType.PARAGRAPH
            and len(block.text) < SHORT_PARAGRAPH_CHARS
        ):
            warnings.append(f"block {position} is only {len(block.text)} chars: {block.text!r}")

    if not any(block.section_path for block in document.blocks):
        warnings.append("no section headings detected - section-aware chunking will do nothing")

    return warnings


def _print_report(document: ParsedDocument, report: dict) -> None:
    typer.echo(f"{document.title}\n{document.origin}\n")
    typer.echo(f"blocks: {report['blocks']}")
    for block_type, count in sorted(report["block_types"].items()):
        typer.echo(f"  {block_type:<10} {count}")

    if report["pages"]:
        pages = report["pages"]
        typer.echo(f"pages: {pages['first']}-{pages['last']} ({pages['seen']} with content)")

    typer.echo(f"\nsections ({len(report['sections'])}):")
    for section in report["sections"][:40]:
        typer.echo(f"  {section}")

    warnings = report["warnings"]
    typer.echo(f"\nwarnings ({len(warnings)}):")
    for warning in warnings[:20]:
        typer.echo(f"  ! {warning}")
    if len(warnings) > 20:
        typer.echo(f"  ... {len(warnings) - 20} more")


if __name__ == "__main__":
    app()
