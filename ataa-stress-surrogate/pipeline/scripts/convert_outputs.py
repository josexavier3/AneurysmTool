"""Stage 2a -- stream-convert the ~23 GB text outputs to a float32 .npy memmap.

Reads the file ONCE with pandas' C parser in chunks (constant memory), into a memmapped
array of shape (n_rows, 16248) = (n_rows, 1354*12): S in the first 8124 columns, C in the
last 8124. Output ~5.2 GB under the mandatory run root (gitignored).

Usage:
    python convert_outputs.py [--rows N]   # --rows: convert only first N rows (for testing)
"""
import os, sys, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

OUT_WIDTH = C.N_NODES * 12          # 16248
N_ROWS_DEFAULT = 79626              # matches sim_inputs_coupled_extended.txt
DATA_CACHE = C.DATA_CACHE           # <work-root>/runs/<run-id>/data
CHUNK = 2000                        # rows per chunk (~130 MB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=0, help="convert only first N rows (0=all)")
    args = ap.parse_args()
    C.init_run()

    src = C.OUTPUTS_FULL
    if not os.path.exists(src):
        sys.exit(f"NOT FOUND: {src}")

    os.makedirs(DATA_CACHE, exist_ok=True)
    n = args.rows or N_ROWS_DEFAULT
    dst = Path(DATA_CACHE) / f"sim_outputs_{'sub' if args.rows else 'full'}.npy"
    temporary = dst.with_suffix(".partial.npy")
    if temporary.exists():
        temporary.unlink()
    print(f"Converting up to {n} rows x {OUT_WIDTH} cols -> {dst}", flush=True)

    arr = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(n, OUT_WIDTH))
    reader = pd.read_csv(src, header=None, sep=",", dtype=np.float32,
                         chunksize=CHUNK, nrows=(args.rows or None), engine="c")
    written, t0 = 0, time.time()
    for chunk in reader:
        b = chunk.to_numpy(dtype=np.float32, copy=False)
        if b.shape[1] != OUT_WIDTH:
            del arr
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Row width {b.shape[1]} != expected {OUT_WIDTH}")
        end = written + b.shape[0]
        if end > n:
            del arr
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Source contains more than the expected {n} rows")
        arr[written:end] = b
        written = end
        el = time.time() - t0
        print(f"  {written}/{n}  ({el:.0f}s, {written/max(el,1):.0f} rows/s, "
              f"ETA {(n-written)/max(written/max(el,1),1):.0f}s)", flush=True)
        if args.rows and written >= args.rows:
            break
    arr.flush()
    del arr
    if written != n:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Conversion wrote {written} rows, expected {n}; partial output removed"
        )
    os.replace(temporary, dst)
    print(f"DONE: {written} rows in {time.time()-t0:.0f}s -> {dst}", flush=True)


if __name__ == "__main__":
    main()
