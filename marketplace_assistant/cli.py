import typer
from pathlib import Path
from marketplace_assistant.loaders.loader import load_csv_to_duckdb, load_file_to_duckdb
from marketplace_assistant.parsers.financial import import_and_normalize_csv

app = typer.Typer()


@app.command("import-csv")
def import_csv(path: str, table: str = "income_statement"):
    """Import a CSV into the DuckDB database as `table`."""
    p = Path(path)
    if not p.exists():
        typer.echo(f"File not found: {p}")
        raise typer.Exit(code=1)
    # Use loader that supports CSV and Excel files
    load_file_to_duckdb(str(p), table)
    typer.echo(f"Imported {p} into table {table}")


@app.command("normalize-csv")
def normalize_csv(path: str, kind: str = "income_statement", quarter: str = None, overwrite: bool = False):
    """Normalize a Marketplace CSV and store it in DuckDB as a normalized table.

    `kind` can be: income_statement, balance_sheet, cash_flow (aliases allowed).
    """
    p = Path(path)
    if not p.exists():
        typer.echo(f"File not found: {p}")
        raise typer.Exit(code=1)
    try:
        norm = import_and_normalize_csv(str(p), kind=kind, quarter=quarter, overwrite=overwrite)
        typer.echo(f"Normalized {p} into table '{kind}' (rows: {len(norm)})")
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
