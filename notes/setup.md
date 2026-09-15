# Setup notes

## Python

`evogym==2.0.0` publishes wheels for `>=3.7,<3.11` only. The sandbox default is Python 3.11.15,
where `pip install evogym` fails with *"No matching distribution found"*. `/usr/bin/python3.10`
(3.10.20) is present, so the venv was rebuilt on 3.10:

```
/usr/bin/python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**No C++ build was needed** — the manylinux wheel `evogym-2.0.0-cp310-cp310-manylinux_2_28_x86_64.whl`
installed directly. Simulator reports itself as *Evolution Gym Simulator v2.2.5*.

### setuptools pin

`evogym/envs/base.py` does `import pkg_resources`, which setuptools removed in 81. The venv's
bootstrap setuptools was 84.0.0, giving `ModuleNotFoundError: No module named 'pkg_resources'` on
`from evogym.envs import *`. Fixed by pinning `setuptools==80.10.2`.

## Ground-truth data (`mertan-a/morphology-fitness-landscape`)

The code in that repo clones fine anonymously, but the data lives in **git LFS**
(`results/compressed_data.tar.gz`, 331,387,185 bytes). Two routes failed and one worked — see
`notes/blockers.md` for the failures.

**What worked:** GitHub's public LFS media host, which serves LFS content for public repos over
ordinary HTTPS without credentials:

```
curl -sSL -o compressed_data.tar.gz \
  https://media.githubusercontent.com/media/mertan-a/morphology-fitness-landscape/master/results/compressed_data.tar.gz
```

Note the branch is **`master`**, not `main` (a `main` URL 404s).

**Integrity verified.** The download's SHA-256 matches the `oid` recorded in the repo's LFS
pointer file exactly, and the byte size matches:

```
pointer oid : d3df2f8d2ba5ce2673c79601170edf1589e4c5edd32e7b5dc9ed5b1c621613c9   size 331387185
download    : d3df2f8d2ba5ce2673c79601170edf1589e4c5edd32e7b5dc9ed5b1c621613c9   size 331387185
```

Extracting yields:

| file | size |
|---|---|
| `results/raw_results.npy` | 3,216,229,035 B (3.0 GB) |
| `results/updated_results.pkl` | 31,290,048 B (30 MB) |

Both are gitignored (`data/mc-landscape/`).

## Environment config of the ground truth

Read from the upstream repo's own `simulator.py` / `main.py`; see `notes/dataset.md` for the full
account and for how our probe environment is matched to it.
