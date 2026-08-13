# Running the example on Windows with WSL

This is a step-by-step for the machine the pipeline was developed on: Windows, with WSL available.
It covers only `examples/phantom`, which is the part that runs without the study data.

Two things make this easier than running the pipeline proper:

- **No solvers.** The example stops after stage 2, so neither `svFSI` nor `svOneDSolver` is
  involved, and none of the `wsl bash -lc` path translation the other stages use is exercised.
- **No dataset, and no `config/local.py`.** The phantom carries its own configuration. If you are
  asked for `config/local.py` at any point, something is wrong — see [Troubleshooting](#troubleshooting).

You therefore have a free choice of where to run it, and the two options differ in what has
actually been tested.

| | Environment file | Tested |
|---|---|---|
| **Inside WSL** | `environment.yml` | yes, on Linux — the same thing WSL runs |
| **Native Windows** | `environment-windows-study.yml` | yes, by R. Valente, August 2026 |

Neither needs the other. Pick one. Both have now produced identical results, down to the last
printed decimal of every phase — see [What you should see](#what-you-should-see).

> **Note, 13 August 2026.** Two changes landed that day, after the run recorded below. First
> `make_phantom.py` gained a sixtieth check — stage 0a's inputs — with `run_stage0a.py` alongside it.
> Then stage 0b was made to run: both control planes moved to stations that satisfy its conventions,
> `make_phantom.py` went to 62 checks, and `run_stage0b.py` was added. The stage-2 accuracy figures
> moved with them, and stage 2 now has two sets, depending on whether stage 0b has run.
>
> **Everything below is the record of R. Valente's run of the earlier state and is left as written.**
> It is a measurement, not a specification, so it has not been edited to match. The 59/59 count and
> every accuracy figure in it predate both changes. Re-run both platforms and re-record this page
> before citing it as current; the numbers to expect are in
> [the main README](README.md#what-is-verified).

---

## A. Inside WSL

### 1. Put the repository on the Linux filesystem

This matters more than it sounds. If the repository sits under `/mnt/c/...` — which is where
GitHub Desktop puts it — every file WSL reads or writes crosses the Windows/Linux filesystem
boundary, and this example writes about 680 MB. Expect it to be several times slower there.

```bash
# in a WSL (Ubuntu) shell
cd ~
git clone https://github.com/josexavier3/AneurysmTool.git
cd AneurysmTool/aorta-mesh-morphing
```

If you would rather keep working from the copy GitHub Desktop already manages, that works too: from
WSL it appears under `/mnt/c/...`, and you would `cd` to it there. Just expect it to take longer.

### 2. Conda, inside WSL

The Windows Anaconda installation is not usable from WSL; it needs its own. If `conda` is not on
the path in the WSL shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh        # accept the defaults, then reopen the shell
```

### 3. Create the environment

Use `environment.yml`, **not** `environment-windows-study.yml`. The latter is a Windows export:
234 of
its 242 dependencies carry Windows build strings and it will not solve under WSL. `environment.yml`
pins the same package *versions* without the build strings.

```bash
conda env create -f environment.yml            # creates env "d_view"
conda activate d_view
```

Check it landed where it should:

```bash
python -c "import pyvista, open3d, numpy; print(pyvista.__version__, open3d.__version__, numpy.__version__)"
# expected: 0.46.5 0.19.0 2.2.6
```

### 4. Run it

```bash
cd examples/phantom
python make_phantom.py        # builds the phantom, verifying as it goes
python run_stage0a.py         # picks the control planes headlessly, then scores them
python run_stage0b.py         # tracks the wall motion, then scores it
python run_stage2.py          # runs the morphing on it, then scores the result
```

All four are ordinary scripts: no notebook server, no display, no arguments. Run them in that
order — stage 0b writes the `dispm/` that stage 2 then reads.

---

## B. Native Windows

Same commands, from an Anaconda Prompt, with the environment built from the Windows export:

```bat
conda env create -f environment-windows-study.yml
conda activate d_view
cd examples\phantom
python make_phantom.py
python run_stage0a.py
python run_stage0b.py
python run_stage2.py
```

This was run in August 2026 and produced output identical to Linux's: 59/59 checks, and every
figure in the accuracy table the same to the last printed decimal, including the per-phase columns.
Two platforms, two Python builds, two BLAS libraries, the same numbers.

---

## What you should see

`make_phantom.py` prints every check as it passes, in four groups, and exits non-zero if any fails.
Each group ends with its count — 9/9, 14/14, 14/14 and 23/23, sixty in all:

```
Phase 4 — SimVascular-shaped outputs
  [ok ] load_case("PHANTOM") returns without config/local.py: 24 keys, cycle 0.8571 s
  ...
  [ok ] ground truth peaks at the systolic phase: max 9.45 mm at 30%

23/23 checks passed
```

`run_stage0a.py` then prints four more, scoring `select_cut.py`'s own inlet and arch planes against
the analytic ones — see [Stage 0a in the main README](README.md#stage-0a) for what the numbers mean.

`run_stage2.py` executes the code cells of `wall_def_temp_def.ipynb` in order — it does not
reimplement anything — and ends with the comparison against the analytic field the phantom was
generated from:

```
peak true displacement          9.451 mm
median error, phases 0-80%      0.300 mm
mean error, phases 0-80%        1.110 mm (11.7 % of peak)
95th percentile                 4.663 mm
maximum local error            27.917 mm

By region, at peak systole (phase 30%) — the two motions are not recovered alike:
region                 nodes   scale    cos  median err  mean |true|
ascending + arch      116294   0.442  0.689       2.081        3.715
descending             47642   0.796  0.853       0.010        0.288
```

**These numbers came out the same under Linux and under native Windows.** That is worth more than
either run alone: it means they are a property of the method and the input, not of one machine's
floating-point library. If yours differ in the second decimal, or the node counts per region differ
at all, something is wrong and it is worth chasing.

The 0.442 value is a documented limitation of this reduced example: it runs without the `dispm/`
control points that Stage 0b supplies for an acquired case. `README.md` in this directory explains
what the result does and does not establish.

---

## What it costs

Measured on Linux, 24 cores. Under WSL, expect the same order; on `/mnt/c`, longer.

| | `make_phantom.py` | `run_stage2.py` |
|---|---|---|
| Wall-clock | 4 min 37 s | 2 min 39 s |
| Peak memory | 4.9 GB | 2.3 GB |
| Writes | 198 MB | 484 MB |

Peak memory is the number to check before starting: the generator holds the ten phase surfaces and
the ground-truth array at once, and 4.9 GB is a real requirement, not a high-water mark you can
ignore. Disk needs 700 MB free.

Everything it writes goes to `examples/phantom/data/`, which is git-ignored — you will not be
committing 680 MB by accident. Delete it whenever you like; it regenerates.

---

## Troubleshooting

**`conda env create` fails with `ResolvePackageNotFound` or Windows build strings under WSL.**
You used `environment-windows-study.yml`. Use `environment.yml`.

**`ImportError: config/local.py not found`.** The phantom needs no local configuration, so this
should not happen. It means `config/__init__.py` is the older version that raised on a missing
`local.py` rather than deferring — check you are on a `main` that contains the phantom commit.

**VTK or PyVista complains about a display, or `libGL.so.1` is missing.** `run_stage2.py` sets
`PYVISTA_OFF_SCREEN` and renders without a window, but VTK still wants the GL libraries present.
On a bare WSL image:

```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglx-mesa0 libxrender1
```

Windows 11's WSLg provides a display and usually makes this unnecessary.

**It is very slow.** Check where the repository is. `/mnt/c/...` is the usual reason.

**`--check-reproducible` and byte-for-byte comparison.** `python make_phantom.py --check-reproducible`
rebuilds everything a second time and compares. It compares two runs *on the same machine*, which is
what it is for. Do not compare the file hashes against a Linux run: `cuts_posit.txt` is written as
text, so Windows gives it CRLF line endings and the hashes differ for that reason alone.

---

## Scope boundary

This guide covers the deterministic Stage-0a, Stage-0b and Stage-2 phantom only. It does not request
or use a study-case `cuts_posit.txt`, private ring selections or clinical `dispm/` files. The
`dispm/` this example produces is written by Stage 0b from the phantom's own analytic surfaces, and
demonstrates that stage on those inputs rather than on acquired cine-CTA — see
[Limitations](README.md#limitations).
