"""
inject_outputs.py
-----------------
Runs all notebook code cells sequentially in this process (no kernel launch
overhead), captures stdout/stderr/display outputs, and writes them back into
the .ipynb file as executed cell outputs.

This is equivalent to a full notebook run but avoids the kernel TCP overhead.
"""
import json, pathlib, sys, io, traceback, base64

NB_PATH    = pathlib.Path(__file__).resolve().parent / "AnirudraGupta_ECommerceSalesPrediction.ipynb"
WORK_DIR   = NB_PATH.parent

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

import os
os.chdir(WORK_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── execution namespace (shared across all cells, mimics a notebook kernel)
_ns = {"__name__": "__main__", "__builtins__": __builtins__}

exec_count = 0
cells_run  = 0
cells_err  = 0

print(f"[INJECT] Running {len(nb['cells'])} cells from {NB_PATH.name}")
print("="*65)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue

    src = "".join(cell["source"])
    if not src.strip():
        continue

    exec_count += 1
    cell["execution_count"] = exec_count
    outputs = []

    # Capture stdout
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    cap_out    = io.StringIO()
    cap_err    = io.StringIO()
    sys.stdout = cap_out
    sys.stderr = cap_err

    error_occurred = False
    try:
        exec(compile(src, f"<cell {i}>", "exec"), _ns)
    except SystemExit:
        pass
    except Exception:
        error_occurred = True
        tb = traceback.format_exc()
        sys.stderr = old_stderr
        print(f"  [CELL {i}] ERROR:\n{tb[:400]}")
        cells_err += 1
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    out_text = cap_out.getvalue()
    err_text = cap_err.getvalue()

    if out_text:
        outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": out_text
        })
    if err_text:
        outputs.append({
            "output_type": "stream",
            "name": "stderr",
            "text": err_text
        })

    # Capture any matplotlib figures
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        outputs.append({
            "output_type": "display_data",
            "metadata": {},
            "data": {"image/png": b64, "text/plain": ["<Figure>"]},
        })
    plt.close("all")

    cell["outputs"] = outputs
    cells_run += 1
    status = "ERROR" if error_occurred else "OK"
    preview = src.strip().split("\n")[0][:60]
    print(f"  [{exec_count:02d}] {status}  {preview}")

print("="*65)
print(f"[INJECT] Done: {cells_run} cells run, {cells_err} errors")

# Save back
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"[INJECT] Saved: {NB_PATH}")
