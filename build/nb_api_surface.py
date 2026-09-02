#!/usr/bin/env python3
"""Extract the API surface of a set of notebooks.

The recitation handouts are scoped from what the assignments actually call,
not from a topic list.  Run this against an assignment's *student* notebooks
before writing or revising the handout that supports it:

    python build/nb_api_surface.py ../assignments/A2/A2_part*.ipynb

It prints, in order: the imports, the attribute calls per library, the bare
built-ins that matter, and the section headings.  Anything a handout teaches
that does not appear here is a candidate for cutting.
"""
import json
import re
import sys
from collections import Counter

LIBS = ("np", "plt", "torch", "nn", "F", "pd", "sklearn", "scipy", "optim",
        "sio", "torchvision", "sns")
CALL = re.compile(r"\b(" + "|".join(LIBS) + r")\.([A-Za-z_][A-Za-z_0-9.]*)")
IMPORT = re.compile(r"^\s*(?:import|from)\s+\S+")
HEADING = re.compile(r"^#{1,4}\s+(.*)")


def cells(path):
    nb = json.load(open(path))
    for c in nb["cells"]:
        yield c["cell_type"], "".join(c["source"])


def main(paths):
    imports, calls, headings = set(), Counter(), []
    for p in paths:
        for kind, src in cells(p):
            if kind == "code":
                for line in src.splitlines():
                    if IMPORT.match(line):
                        imports.add(line.strip())
                for lib, attr in CALL.findall(src):
                    calls[f"{lib}.{attr.split('(')[0]}"] += 1
            else:
                for line in src.splitlines():
                    m = HEADING.match(line)
                    if m:
                        headings.append(m.group(1).strip("* "))

    print("=" * 60, "\nIMPORTS\n" + "=" * 60)
    for i in sorted(imports):
        print(" ", i)

    print("\n" + "=" * 60, "\nAPI CALLS  (count, name)\n" + "=" * 60)
    by_lib = {}
    for name, n in calls.items():
        by_lib.setdefault(name.split(".")[0], []).append((n, name))
    for lib in sorted(by_lib):
        print(f"\n-- {lib} --")
        for n, name in sorted(by_lib[lib], reverse=True):
            print(f"  {n:4d}  {name}")

    print("\n" + "=" * 60, "\nHEADINGS\n" + "=" * 60)
    for h in headings:
        print(" ", h)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
