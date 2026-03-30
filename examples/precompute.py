#!/usr/bin/env python
"""Pre-compute LAMMPS results for all demo notebooks and save as .json files.

Run this with a working LAMMPS installation:

    python examples/precompute.py

The resulting files in examples/data/ are loaded by the notebooks when
LAMMPS is unavailable (e.g. in a Pyodide/WASM export).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lammps import lammps  # noqa: E402
from LACT import atom_cont_system  # noqa: E402
from LACT.precomputed import save_system  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ── Demo 1: LJ dimer ─────────────────────────────────────────────────────────

def demo1():
    print("Demo 1: LJ dimer")

    def make_dimer():
        _lmp = lammps(cmdargs=["-screen", "none"])
        _lmp.commands_string("""
            units         lj
            dimension     3
            boundary      f f f
            atom_style    atomic
            atom_modify   map yes
            region box block -50.0 50.0 -50.0 50.0 -50.0 50.0
            create_box 1 box
            mass 1 1.0
            create_atoms 1 single 0.0 0.0 0.0
            create_atoms 1 single 1.12246 0.0 0.0
            reset_atom_ids sort yes
            pair_style lj/cut 3.0
            pair_coeff 1 1 1.0 1.0 3.0
            group fixed id 1
            group mobile id 2
            fix freeze fixed setforce 0.0 0.0 0.0
            fix pull mobile addforce 0.0 0.0 0.0
            fix_modify pull energy yes
            compute forces all property/atom fx fy fz
            compute ids all property/atom id
        """)

        def update_command(force):
            return f"""
            unfix pull
            fix pull mobile addforce {force} 0.0 0.0
            fix_modify pull energy yes
            """

        return atom_cont_system(_lmp, update_command)

    # Quasi-static
    qs_sys = make_dimer()
    qs_sys.quasi_static_run(0.0, 0.1, 30, verbose=False)
    save_system(qs_sys, os.path.join(DATA_DIR, "demo1_qs.json"))
    print(f"  qs: {len(qs_sys.data['Y_s'])} points")

    # Continuation
    cont_sys = make_dimer()
    cont_sys.quasi_static_run(0.0, 0.5, 5, verbose=False)
    cont_sys.continuation_run(
        n_iter=100, ds_default=0.01, ds_smallest=0.001, ds_largest=1.0,
        verbose=False, checkpoint_freq=0, cont_target=0.0, target_tol=0.05,
    )
    save_system(cont_sys, os.path.join(DATA_DIR, "demo1_cont.json"))
    print(f"  cont: {len(cont_sys.data['Y_s'])} points")


# ── Demo 2: Morse chain ──────────────────────────────────────────────────────

def demo2():
    print("Demo 2: Morse chain")

    n_atoms = 6
    alpha = 6.0
    r0 = 1.12
    bond_info = [
        ((1, 2), 0.85), ((2, 3), 0.90), ((3, 4), 0.95),
        ((4, 5), 1.00), ((5, 6), 1.05),
    ]

    def make_chain():
        _lmp = lammps(cmdargs=["-screen", "none"])
        _lmp.commands_string(f"""
            units         lj
            dimension     3
            boundary      f f f
            atom_style    atomic
            atom_modify   map yes
            region box block -50.0 50.0 -50.0 50.0 -50.0 50.0
            create_box {n_atoms} box
            mass * 1.0
            create_atoms 1 single {0 * r0:.6f} 0.0 0.0
            create_atoms 2 single {1 * r0:.6f} 0.0 0.0
            create_atoms 3 single {2 * r0:.6f} 0.0 0.0
            create_atoms 4 single {3 * r0:.6f} 0.0 0.0
            create_atoms 5 single {4 * r0:.6f} 0.0 0.0
            create_atoms 6 single {5 * r0:.6f} 0.0 0.0
            reset_atom_ids sort yes
            pair_style morse 5.0
            pair_coeff * * 0.001 {alpha} {r0} 0.001
            pair_coeff 1 2 {bond_info[0][1]} {alpha} {r0} 5.0
            pair_coeff 2 3 {bond_info[1][1]} {alpha} {r0} 5.0
            pair_coeff 3 4 {bond_info[2][1]} {alpha} {r0} 5.0
            pair_coeff 4 5 {bond_info[3][1]} {alpha} {r0} 5.0
            pair_coeff 5 6 {bond_info[4][1]} {alpha} {r0} 5.0
            group left_end id 1
            group right_end id 6
            fix freeze left_end setforce 0.0 0.0 0.0
            fix pull right_end addforce 0.0 0.0 0.0
            fix_modify pull energy yes
            compute forces all property/atom fx fy fz
            compute ids all property/atom id
        """)

        def update_command(force):
            return f"""
            unfix pull
            fix pull right_end addforce {force} 0.0 0.0
            fix_modify pull energy yes
            """

        return atom_cont_system(_lmp, update_command)

    # Quasi-static
    qs_sys = make_chain()
    try:
        qs_sys.quasi_static_run(0.0, 0.02, 200, verbose=False)
    except Exception:
        pass
    save_system(qs_sys, os.path.join(DATA_DIR, "demo2_qs.json"))
    print(f"  qs: {len(qs_sys.data['Y_s'])} points")

    # Continuation
    cont_sys = make_chain()
    cont_sys.quasi_static_run(0.0, 0.5, 5, verbose=False)
    cont_sys.continuation_run(
        n_iter=200, ds_default=0.01, ds_smallest=0.001, ds_largest=0.05,
        verbose=False, checkpoint_freq=0, cont_target=0.5, target_tol=0.1,
    )
    save_system(cont_sys, os.path.join(DATA_DIR, "demo2_cont.json"))
    print(f"  cont: {len(cont_sys.data['Y_s'])} points")


# ── Demo 3: Crystal buckling ─────────────────────────────────────────────────

def demo3():
    print("Demo 3: Crystal buckling")

    NX = 4

    def make_crystal():
        _lmp = lammps(cmdargs=["-screen", "none"])
        _lmp.commands_string(f"""
            units         lj
            dimension     2
            boundary      p p p
            atom_style    atomic
            atom_modify   map yes
            lattice       hex 0.9165
            region        box block 0 {NX} 0 {NX} -0.5 0.5
            create_box    1 box
            create_atoms  1 box
            mass          1 1.0
            pair_style    lj/cut 2.5
            pair_coeff    1 1 1.0 1.0 2.5
            fix boxrelax all box/relax iso 0.0 vmax 0.001
            minimize 0 1e-12 50000 50000
            unfix boxrelax
            compute forces all property/atom fx fy fz
            compute ids all property/atom id
            run 0
        """)
        _box = _lmp.extract_box()
        _xlo, _xhi = _box[0][0], _box[1][0]
        _Lx = _xhi - _xlo

        def _update(strain):
            _dx = _Lx * strain / 2
            return f"change_box all x final {_xlo+_dx} {_xhi-_dx} units box"

        return atom_cont_system(_lmp, _update), _lmp

    def seed_continuation(qs_Ys, idx, reverse=False):
        new_sys, _lmp = make_crystal()
        new_sys.quasi_static_run(0.0, 0.0, 1, verbose=False)
        Y_a = qs_Ys[idx].copy()
        Y_b = qs_Ys[idx + 1].copy()
        if reverse:
            new_sys.data["Y_s"] = [Y_b, Y_a]
        else:
            new_sys.data["Y_s"] = [Y_a, Y_b]
        new_sys.data["ds_s"] = []
        return new_sys

    # Quasi-static
    strain_max = 0.16
    K = 1000
    sys, sys_lmp = make_crystal()
    increment = strain_max / K
    sys.quasi_static_run(0.0, increment, K + 1, verbose=False)
    sys.compute_energies()
    save_system(sys, os.path.join(DATA_DIR, "demo3_qs.json"))
    print(f"  qs: {len(sys.data['Y_s'])} points")

    qs_Ys = sys.data["Y_s"]

    # Continuation A: seeded near end, reverse
    sys_a = seed_continuation(qs_Ys, -6, reverse=True)
    sys_a.continuation_run(
        n_iter=200, ds_default=0.01, ds_smallest=0.001, ds_largest=0.05,
        verbose=False, checkpoint_freq=0, cont_target=0.16, target_tol=0.001,
    )
    sys_a.data["Y_s"] = sys_a.data["Y_s"][:-2]
    sys_a.compute_energies()
    save_system(sys_a, os.path.join(DATA_DIR, "demo3_cont_a.json"))
    print(f"  cont_a: {len(sys_a.data['Y_s'])} points")

    # Continuation B: seeded at idx 750, forward
    sys_b = seed_continuation(qs_Ys, 750, reverse=False)
    sys_b.continuation_run(
        n_iter=100, ds_default=0.001, ds_smallest=0.001, ds_largest=0.005,
        verbose=False, checkpoint_freq=0, cont_target=0.16, target_tol=0.005,
    )
    sys_b.compute_energies()
    save_system(sys_b, os.path.join(DATA_DIR, "demo3_cont_b.json"))
    print(f"  cont_b: {len(sys_b.data['Y_s'])} points")


# ── Demo 4: Vacancy migration ────────────────────────────────────────────────

def demo4():
    print("Demo 4: Vacancy migration")

    N = 4
    a = 2.0 ** 1.5
    n_atoms = 255
    seed = 12345

    def make_crystal():
        _rng = np.random.default_rng(seed)
        _lmp = lammps(cmdargs=["-screen", "none"])
        _lmp.commands_string(f"""
            units         metal
            lattice       fcc {a}
            region        cell block 0 {N} 0 {N} 0 {N} units lattice
            atom_modify   map yes
            create_box    2 cell
            create_atoms  1 region cell
            pair_style    lj/smooth/linear 6.0
            pair_coeff    1 1 2.0 2.0 6.0
            pair_coeff    2 2 2.0 2.0 6.0
            pair_coeff    1 2 2.0 2.0 6.0
            mass          * 1.0
            region vac block {N//2} {N//2 + 0.25} {N//2} {N//2 + 0.25} {N//2} {N//2 + 0.25} units lattice
            group  vac region vac
            delete_atoms group vac
            compute forces all property/atom fx fy fz
            compute ids all property/atom id
        """)
        _types = _lmp.numpy.extract_atom("type")
        _types[_rng.uniform(size=_types.size) < 0.2] = 2

        def update_command(mu):
            return f"""
            pair_coeff 1 2 2.0 {2.0 * mu} 6.0
            """

        _sys = atom_cont_system(_lmp, update_command)
        return _sys, _lmp

    def seed_continuation(qs_Ys, idx, reverse=False):
        new_sys, _lmp = make_crystal()
        new_sys.quasi_static_run(1.0, 0.0, 1, verbose=False)
        Y_a = qs_Ys[idx].copy()
        Y_b = qs_Ys[idx + 1].copy()
        if reverse:
            new_sys.data["Y_s"] = [Y_b, Y_a]
        else:
            new_sys.data["Y_s"] = [Y_a, Y_b]
        new_sys.data["ds_s"] = []
        return new_sys

    # Quasi-static
    qs_sys, qs_lmp = make_crystal()
    qs_sys.quasi_static_run(1.0, 0.001, 200, verbose=False)
    qs_sys.compute_energies()
    save_system(qs_sys, os.path.join(DATA_DIR, "demo4_qs.json"))
    print(f"  qs: {len(qs_sys.data['Y_s'])} points")

    qs_Ys = qs_sys.data["Y_s"]

    # Continuation A: forward from idx 120
    sys_a = seed_continuation(qs_Ys, 120, reverse=False)
    sys_a.continuation_run(
        n_iter=200, ds_default=0.01, ds_smallest=1e-5, ds_largest=2.0,
        cont_target=0.95, verbose=False, checkpoint_freq=0,
    )
    sys_a.compute_energies()
    save_system(sys_a, os.path.join(DATA_DIR, "demo4_cont_a.json"))
    print(f"  cont_a: {len(sys_a.data['Y_s'])} points")

    # Continuation B: reverse from idx 135
    sys_b = seed_continuation(qs_Ys, 135, reverse=True)
    sys_b.continuation_run(
        n_iter=200, ds_default=0.01, ds_smallest=1e-5, ds_largest=2.0,
        cont_target=0.95, verbose=False, checkpoint_freq=0,
    )
    sys_b.compute_energies()
    save_system(sys_b, os.path.join(DATA_DIR, "demo4_cont_b.json"))
    print(f"  cont_b: {len(sys_b.data['Y_s'])} points")


if __name__ == "__main__":
    demo1()
    demo2()
    demo3()
    demo4()
    print("Done. Data files written to examples/data/")
