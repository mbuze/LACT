# LACT

**LAmmps Continuation Techniques** — a Python wrapper around [LAMMPS](https://www.lammps.org/) implementing pseudo-arclength continuation for atomistic systems.

LACT traces equilibrium solution paths as a scalar continuation parameter (e.g. applied strain or force) varies, including through **fold points** (limit points / spinodal instabilities) where standard load-stepping fails. It uses a Newton–Krylov solver on an extended system with a pseudo-arclength constraint, so both stable and unstable equilibria can be followed in a single run.

## Features

- Pseudo-arclength continuation with adaptive step-size control
- Quasi-static loading (incremental parameter + LAMMPS energy minimisation)
- Eigenvector-following for saddle-point (transition state) tracking
- Turn detection and optional exit callbacks
- Checkpointing and restart
- MPI-parallel via mpi4py (optional; falls back to serial)

## Installation

### Prerequisites

- Python ≥ 3.8
- A working [LAMMPS](https://docs.lammps.org/Build.html) build with the Python interface enabled
- MPI library (e.g. OpenMPI) if using the LAMMPS parallel build

### Install LACT

```bash
pip install numpy scipy matscipy ase matplotlib
pip install .            # from the repository root
```

If LAMMPS was built locally, install its Python bindings too, e.g.:

```bash
pip install /path/to/lammps/python
# or from a wheel:
pip install /path/to/lammps/build/lammps-*.whl
```

## Quick start

```python
from lammps import lammps
from LACT import atom_cont_system

# 1. Set up a LAMMPS simulation (pair style, atoms, computes, etc.)
lmp = lammps(cmdargs=["-screen", "none"])
lmp.commands_string("""
    units         metal
    ...
    compute forces all property/atom fx fy fz
    compute ids    all property/atom id
""")

# 2. Define how the continuation parameter changes the simulation
def update_command(mu):
    """Return a LAMMPS command string that applies parameter value mu."""
    return f"change_box all x final 0.0 {L0 + mu} units box"

# 3. Create the continuation system
system = atom_cont_system(lmp, update_command)

# 4. Seed the path with a quasi-static ramp
system.quasi_static_run(mu_start=0.0, increment=0.01, n_iter=10)

# 5. Trace the solution path through folds
system.continuation_run(
    n_iter=200,
    ds_default=0.5,    # initial arclength step
    ds_smallest=1e-3,  # abort if step shrinks below this
    ds_largest=2.0,    # cap on step growth
)

# 6. Inspect results
for Y in system.data["Y_s"]:
    positions = Y[:-1].reshape(-1, 3)  # atomic deviations from reference
    mu = Y[-1]                          # continuation parameter value
```

The two LAMMPS computes (`forces` and `ids`) shown above are **required** by LACT.

## Examples

Interactive [marimo](https://marimo.io) notebooks live in `examples/`:

| Notebook | Description |
|----------|-------------|
| `demo1_lj_dimer.py` | Two LJ atoms pulled apart — traces the force–extension S-curve through the spinodal fold |
| `demo2_lj_chain.py` | Six-atom Morse chain with disordered bond strengths — shows which bond snaps first |

Run with:

```bash
pip install marimo
marimo run examples/demo1_lj_dimer.py
```

## Parallel runs

LACT detects `mpi4py` at import and falls back to serial if unavailable.
For parallel execution, pass the MPI communicator when creating the system:

```python
from mpi4py import MPI
system = atom_cont_system(lmp, update_command, comm=MPI.COMM_WORLD)
```

Run with `mpirun -np N python your_script.py`.

## Citation

If you use LACT in your research, please cite it:

> M. Buze and F. Birks, *LACT (LAMMPS Continuation Techniques)*, https://github.com/mbuze/LACT

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## License

MIT — see [LICENSE](LICENSE).
