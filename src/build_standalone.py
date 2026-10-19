"""Concatenate the standalone page: shell + weights + runtime.

Produces one self-contained HTML file with the network embedded, suitable
for publishing anywhere that serves static files.

Usage:
    python src/export_web.py && python src/build_standalone.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHELL = ROOT / "web" / "standalone" / "shell.html"
RUNTIME = ROOT / "web" / "standalone" / "runtime.js"
WEIGHTS = ROOT / "web" / "weights.js"
OUT = ROOT / "web" / "digit-reader.html"


def main():
    for f in (SHELL, RUNTIME, WEIGHTS):
        if not f.exists():
            raise SystemExit(f"missing {f} -- run src/export_web.py first")

    # The shell ends with an open <script>; weights and runtime go inside it.
    page = "".join([
        SHELL.read_text(),
        WEIGHTS.read_text(),
        RUNTIME.read_text(),
        "</script>\n",
    ])
    OUT.write_text(page)
    print(f"wrote {OUT}  ({len(page) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
