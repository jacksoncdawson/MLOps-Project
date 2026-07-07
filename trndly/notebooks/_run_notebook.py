"""Programmatic notebook runner.

We don't ship `jupyter`/`nbclient` in the project venv, but the cleaning
notebooks are pure pandas/numpy/matplotlib — runnable as a flat script.

This loader extracts every code cell from a target notebook and execs them in
a single shared namespace. Matplotlib cells are skipped (we don't need plots
for a headless re-run). The notebook's relative-path constants (`DATA_DIR`,
`OUT_DIR`) work as long as we set CWD to the notebook's directory before exec.

Usage:

    python EDA/_run_notebook.py EDA/0_clean_historical.ipynb

The script writes nothing back to the notebook. Outputs are produced by the
notebook's own persist cells (e.g. `transactions.parquet`, `lookup.csv`).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def run_notebook(nb_path: str, skip_visual: bool = True) -> None:
    nb_path = Path(nb_path).resolve()
    nb = json.loads(nb_path.read_text())

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"running {nb_path.name}: {len(code_cells)} code cells", flush=True)

    cwd_before = os.getcwd()
    os.chdir(nb_path.parent)

    ns: dict = {"__name__": "__main__"}
    started = time.time()

    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if not src.strip():
            continue
        if skip_visual and ("matplotlib" in src or "plt." in src):
            print(f"  [{i:>2}] skip (visual)", flush=True)
            continue
        head = src.strip().splitlines()[0][:70]
        t0 = time.time()
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [{i:>2}] FAIL after {time.time() - t0:.1f}s: {head}", flush=True)
            os.chdir(cwd_before)
            raise
        print(f"  [{i:>2}] ok  ({time.time() - t0:5.1f}s)  {head}", flush=True)

    os.chdir(cwd_before)
    print(f"done in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python _run_notebook.py <notebook.ipynb>", file=sys.stderr)
        sys.exit(2)
    run_notebook(sys.argv[1])
