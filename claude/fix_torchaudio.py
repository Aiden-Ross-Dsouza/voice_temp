"""
fix_torchaudio.py
-----------------
Fixes the meta-tensor crash in torchaudio that prevents IndicF5 from loading:
    RuntimeError: Tensor.item() cannot be called on meta tensors

Run once from your project folder:
    python claude/fix_torchaudio.py
"""

import glob
import importlib
import inspect
import os
import pathlib
import shutil
import sys


def find_functional_py():
    """Return path to torchaudio/functional/functional.py (the installed .py source)."""
    import torchaudio.functional
    p = pathlib.Path(inspect.getfile(torchaudio.functional))
    print(f"torchaudio functional module : {p}")
    # inspect.getfile may return the .pyc — we want the .py
    if p.suffix == ".pyc":
        # .pyc lives at  __pycache__/functional.cpython-310.pyc
        # .py  lives at  ../functional.py
        py = p.parent.parent / (p.stem.split(".")[0] + ".py")
        if py.exists():
            p = py
    print(f"Source .py file              : {p}")
    return p


def show_lines(path, centre, radius=4):
    lines = path.read_text(encoding="utf-8").splitlines()
    lo = max(0, centre - radius - 1)
    hi = min(len(lines), centre + radius)
    print(f"\n--- {path} (lines {lo+1}–{hi}) ---")
    for i, l in enumerate(lines[lo:hi], lo + 1):
        marker = ">>>" if i == centre else "   "
        print(f"{marker} {i:4d}  {l}")
    print()


def patch_functional_py(path: pathlib.Path):
    txt = path.read_text(encoding="utf-8")
    lines = txt.splitlines(keepends=True)

    NEEDLE   = "if (fb.max(dim=0).values == 0.0).any():"
    PATCHED  = "if not fb.is_meta and (fb.max(dim=0).values == 0.0).any():"

    # Check current state
    for i, l in enumerate(lines, 1):
        stripped = l.strip()
        if stripped == PATCHED:
            print(f"Line {i}: already patched correctly.")
            show_lines(path, i)
            return True
        if stripped == NEEDLE:
            print(f"Line {i}: found unpatched line — patching now.")
            show_lines(path, i)
            lines[i-1] = l.replace(NEEDLE, PATCHED, 1)
            path.write_text("".join(lines), encoding="utf-8")
            print("Patch written.")
            show_lines(path, i)
            return True

    # Neither found — print surrounding area for diagnosis
    print("ERROR: Neither patched nor unpatched line found.")
    print("Searching for 'fb.max' to show context...")
    for i, l in enumerate(lines, 1):
        if "fb.max" in l or "melscale_fbanks" in l:
            show_lines(path, i)
    return False


def nuke_pyc(path: pathlib.Path):
    """Delete all .pyc files for this module so Python recompiles from the .py."""
    stem = path.stem  # e.g. "functional"
    cache_dir = path.parent / "__pycache__"
    deleted = []
    if cache_dir.is_dir():
        for pyc in cache_dir.glob(f"{stem}*.pyc"):
            pyc.unlink()
            deleted.append(pyc)
    if deleted:
        print(f"Deleted {len(deleted)} .pyc file(s):")
        for f in deleted:
            print(f"  {f}")
    else:
        print("No .pyc files found (or already clean).")
    return deleted


def verify(path: pathlib.Path):
    """Reload torchaudio and confirm the patch is live."""
    # Force reimport
    mods_to_remove = [k for k in sys.modules if k.startswith("torchaudio")]
    for m in mods_to_remove:
        del sys.modules[m]

    import torchaudio.functional as taf_fresh
    import inspect as _ins
    src = pathlib.Path(_ins.getfile(taf_fresh))
    txt = src.read_text(encoding="utf-8")
    PATCHED = "if not fb.is_meta and (fb.max(dim=0).values == 0.0).any():"
    NEEDLE  = "if (fb.max(dim=0).values == 0.0).any():"
    if PATCHED in txt:
        print("\nVERIFICATION PASSED: patch is live in the loaded module.")
        return True
    elif NEEDLE in txt:
        print("\nVERIFICATION FAILED: original unpatched line still present.")
        print("This usually means Python loaded a .pyc — check __pycache__ manually.")
        return False
    else:
        print("\nVERIFICATION WARNING: neither line found after reload.")
        return False


def main():
    print("=" * 60)
    print("IndicF5 torchaudio meta-tensor fix")
    print("=" * 60)

    path = find_functional_py()

    if not path.exists():
        print(f"ERROR: {path} does not exist.")
        sys.exit(1)

    patched = patch_functional_py(path)
    if not patched:
        sys.exit(1)

    print("\nClearing .pyc cache...")
    nuke_pyc(path)

    print("\nVerifying patch is live...")
    ok = verify(path)

    if ok:
        print("\nAll done. Run your pipeline now:")
        print("  python claude/indic_f5_voice_clone.py --ref_audio claude/lady.wav "
              "--text_file claude/text.txt --language hi --hf_token <your_token>")
    else:
        print("\nPatch written but verification failed.")
        print("Try running: python -B claude/indic_f5_voice_clone.py ...")
        print("(The -B flag tells Python to ignore .pyc files entirely.)")

    print("=" * 60)


if __name__ == "__main__":
    main()