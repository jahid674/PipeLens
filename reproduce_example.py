"""Execute the PipeLens example notebook from a clean command line."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "artifacts" / "example.executed.ipynb"


def execute_notebook(output: Path, timeout: int) -> None:
    notebook = nbformat.read(ROOT / "example.ipynb", as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    execute_notebook(args.output.resolve(), args.timeout)
    print(f"Executed notebook written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
