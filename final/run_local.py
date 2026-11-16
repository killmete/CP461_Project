"""Local smoke-test harness for the CP461 Colab notebook.

Runs the notebook's code cells in ONE Python namespace (like a Colab kernel), from the sandbox folder
final/_localwork, so the project's real data/runs/outputs are never written to. Optionally appends extra
.py section files after the notebook cells.

Usage (from the project root, with the local venv):
  final/.venv/Scripts/python.exe final/run_local.py --nb Litter_EfficientSAM_TACO.ipynb --upto 28 \
      --set MAX_TEST_IMAGES=20 --append final/sections/e6_prompt_study.py

  --nb       notebook to run (default: the v1 notebook)
  --upto N   run code cells with index <= N (cell indices as in final/notebook_v1_code.py)
  --skip     comma-separated cell indices to skip
  --set      CONFIG overrides applied right after the CONFIG cell (python literals), repeatable
  --append   extra .py files executed after the cells, in order, repeatable

Defaults always applied: LAUNCH_DEMO=False, SAVE_TO_DRIVE=False, TIMING_IMAGES=5.
Figures go to final/_localwork/litter_esam/outputs/figures (plt.show is a no-op).
"""
import argparse, json, os, re, subprocess, sys, time, traceback
from pathlib import Path

if __name__ == "__main__" and os.environ.get("PYTHONUTF8") != "1":          # Colab is UTF-8 everywhere; make Windows match (open(), print)
    os.environ["PYTHONUTF8"] = "1"
    sys.exit(subprocess.run([sys.executable, *sys.argv]).returncode)

# Windows: DataLoader workers re-import this file as __mp_main__ → only run the harness in the real main process
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    SANDBOX = ROOT / "final" / "_localwork"

    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", default=str(ROOT / "Litter_EfficientSAM_TACO.ipynb"))
    ap.add_argument("--upto", type=int, default=10**9)
    ap.add_argument("--skip", default="")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--append", action="append", default=[])
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: plt.close("all")

    def display(*objs, **kw):
        for o in objs:
            try:
                print(o.to_string() if hasattr(o, "to_string") else o)
            except Exception:
                print(o)

    nb = json.loads(Path(args.nb).read_text(encoding="utf-8"))
    skip = {int(s) for s in args.skip.split(",") if s.strip()}
    overrides = {"LAUNCH_DEMO": "False", "SAVE_TO_DRIVE": "False", "TIMING_IMAGES": "5"}
    for kv in args.set:
        k, v = kv.split("=", 1); overrides[k.strip()] = v.strip()

    os.chdir(SANDBOX)                 # notebook uses BASE = Path.cwd() off Colab → WORK = _localwork/litter_esam
    sys.argv = [sys.argv[0]]
    ns = {"__name__": "__main__", "display": display}

    def run(src, label):
        # drop IPython magics / shell lines (packages are pre-installed in the venv)
        src = "\n".join(l for l in src.splitlines() if not re.match(r"\s*[%!]", l))
        t0 = time.time()
        print(f"\n========== {label} ==========", flush=True)
        try:
            exec(compile(src, label, "exec"), ns)
        except Exception:
            traceback.print_exc()
            print(f"\n!!!!!!!!!! FAILED in {label} !!!!!!!!!!")
            sys.exit(1)
        print(f"---------- {label} ok ({time.time() - t0:.1f}s)", flush=True)

    for i, c in enumerate(nb["cells"]):
        if i > args.upto:
            break
        if c["cell_type"] != "code" or i in skip:
            continue
        src = "".join(c["source"])
        run(src, f"cell {i}")
        if "PROJECT_NAME" in src and "=" in src and "CONFIG" in src:      # the CONFIG cell → apply overrides
            run("\n".join(f"{k} = {v}" for k, v in overrides.items()), "overrides " + str(overrides))

    for f in args.append:
        run(Path(ROOT / f if not Path(f).is_absolute() else f).read_text(encoding="utf-8"), f)
    print("\nALL DONE")
