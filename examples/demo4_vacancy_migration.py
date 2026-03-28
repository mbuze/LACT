import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Demo 4: 3D LJ crystal with vacancy — vacancy migration

    A 3D FCC Lennard-Jones crystal (4×4×4 unit cells, 255 atoms after
    removing one vacancy) contains two randomly assigned atomic species
    (80% type 1, 20% type 2). The **cross-species interaction parameter**
    $\sigma_{12} = 2\mu$ is used as the continuation parameter.

    At $\mu = 1$ both species interact identically. As $\mu$ increases,
    the size mismatch between species drives instabilities — the vacancy
    migrates and the lattice rearranges. Quasi-static loading captures
    these jumps as discontinuities, while **arclength continuation**
    traces the full solution path through the folds.

    In this notebook we use LACT to trace these instabilities numerically.

    As a result, we are able to capture the simple saddle corresponding to vacancy migration, but also a more complicated rearrangement involving three atoms only.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We first load the necessary dependencies and define some helper functions,
    which include creating a LAMMPS instance, passing on the simulation setup
    and the update command associated with the continuation parameter
    ($\sigma_{12} = 2\mu$):
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    try:
        from LACT.precomputed import PrecomputedSystem
    except ImportError:
        import sys as _sys, io as _io

        class PrecomputedSystem:
            def __init__(self, path_or_url):
                if "pyodide" in _sys.modules:
                    from pyodide.http import open_url
                    _d = np.load(_io.BytesIO(open_url(path_or_url).read()), allow_pickle=False)
                else:
                    _d = np.load(path_or_url, allow_pickle=False)
                self.natoms = int(_d["natoms"])
                self.U_0 = _d["U_0"]
                self.data = {"Y_s": list(_d["Y_s"]), "ds_s": list(_d["ds_s"]) if "ds_s" in _d else []}
                if "energies" in _d:
                    self.data["energies"] = list(_d["energies"])

    try:
        from lammps import lammps
        from LACT import atom_cont_system
        HAVE_LAMMPS = True
    except ImportError:
        lammps = None
        atom_cont_system = None
        HAVE_LAMMPS = False

    return HAVE_LAMMPS, PrecomputedSystem, atom_cont_system, go, lammps, mo, np, plt


@app.cell
def _(HAVE_LAMMPS, atom_cont_system, lammps, np):
    N = 4
    a = 2.0 ** 1.5  # FCC lattice constant
    n_atoms = 255    # 4^3 * 4 - 1 vacancy
    seed = 12345

    if HAVE_LAMMPS:
        def make_crystal():
            _rng = np.random.default_rng(seed)
            _lmp = lammps(cmdargs=["-screen", "none"])
            _lmp.commands_string(
                f"""
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
                """
            )

            # Randomly assign ~20% of atoms as type 2 (fixed seed)
            _types = _lmp.numpy.extract_atom("type")
            _types[_rng.uniform(size=_types.size) < 0.2] = 2

            def update_command(mu):
                return f"""
                pair_coeff 1 2 2.0 {2.0 * mu} 6.0
                """

            _sys = atom_cont_system(_lmp, update_command)
            return _sys, _lmp
    else:
        make_crystal = None

    return make_crystal, n_atoms


@app.cell
def _(n_atoms, np):
    def get_crystal_data(sys_obj):
        """Extract mu values and displacement norms from solution data."""
        _mus, _disp_norms, _max_disps = [], [], []
        for _Y in sys_obj.data["Y_s"]:
            _mus.append(_Y[-1])
            _dev = _Y[:-1].reshape(n_atoms, 3)
            _norm = np.linalg.norm(_dev)
            _max_d = np.max(np.linalg.norm(_dev, axis=1))
            _disp_norms.append(_norm)
            _max_disps.append(_max_d)
        return np.array(_mus), np.array(_disp_norms), np.array(_max_disps)

    return (get_crystal_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quasi-static loading

    We ramp $\mu$ from 1.0 in steps of 0.001 for 200 steps
    ($\mu \in [1.0, 1.2]$). Instabilities appear as discontinuous jumps
    in the energy and displacement norm.
    """)
    return


@app.cell
def _(HAVE_LAMMPS, PrecomputedSystem, get_crystal_data, make_crystal, np):
    if HAVE_LAMMPS:
        qs_sys, qs_lmp = make_crystal()
        qs_sys.quasi_static_run(1.0, 0.001, 200, verbose=False)
        qs_sys.compute_energies()
    else:
        qs_sys = PrecomputedSystem("data/demo4_qs.npz")
    qs_mus, qs_disp_norms, qs_max_disps = get_crystal_data(qs_sys)
    qs_energies = np.array(qs_sys.data["energies"])
    return qs_disp_norms, qs_energies, qs_mus, qs_sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Energy and displacement vs $\mu$ (quasi-static only)

    These plots help identify the instabilities where continuation
    branches should be seeded.
    """)
    return


@app.cell(hide_code=True)
def _(plt, qs_disp_norms, qs_energies, qs_mus):
    _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    _ax1.plot(qs_mus, qs_energies, "-", lw=1.5, color="C1")
    _ax1.set_ylabel("Potential energy")
    _ax1.set_title("Quasi-static: energy vs $\\mu$")
    _ax1.grid(True, alpha=0.3)

    _ax2.plot(qs_mus, qs_disp_norms, "-", lw=1.5, color="C1")
    _ax2.set_xlabel(r"$\mu$")
    _ax2.set_ylabel("Total displacement norm")
    _ax2.set_title("Quasi-static: displacement norm vs $\\mu$")
    _ax2.grid(True, alpha=0.3)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arclength continuation

    We use `seed_continuation` to create fresh systems seeded from
    two consecutive quasi-static points near the instabilities.
    Adjust the indices below based on the QS energy plot above.
    """)
    return


@app.function
def seed_continuation(make_crystal, qs_Ys, idx, reverse=False):
    import numpy as np
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


@app.cell
def _(HAVE_LAMMPS, PrecomputedSystem, make_crystal, qs_sys):
    if HAVE_LAMMPS:
        # --- Continuation A: seed near first instability, forward ---
        # Adjust idx_a based on the QS energy plot
        idx_a = 120
        sys_a = seed_continuation(make_crystal, qs_sys.data["Y_s"], idx_a, reverse=False)
        sys_a.continuation_run(
            n_iter=200,
            ds_default=0.01,
            ds_smallest=1e-5,
            ds_largest=2.0,
            cont_target=0.95,
            verbose=False,
            checkpoint_freq=0,
        )
        sys_a.compute_energies()
    else:
        sys_a = PrecomputedSystem("data/demo4_cont_a.npz")
    return (sys_a,)


@app.cell
def _(HAVE_LAMMPS, PrecomputedSystem, make_crystal, qs_sys):
    if HAVE_LAMMPS:
        # --- Continuation B: seed near first instability, reverse ---
        # Adjust idx_b based on the QS energy plot
        idx_b = 135
        sys_b = seed_continuation(make_crystal, qs_sys.data["Y_s"], idx_b, reverse=True)
        sys_b.continuation_run(
            n_iter=200,
            ds_default=0.01,
            ds_smallest=1e-5,
            ds_largest=2.0,
            cont_target=0.95,
            verbose=False,
            checkpoint_freq=0,
        )
        sys_b.compute_energies()
    else:
        sys_b = PrecomputedSystem("data/demo4_cont_b.npz")
    return (sys_b,)


@app.cell
def _(np, qs_sys, sys_a, sys_b):
    branch_styles = [
        ("QS",              qs_sys, "-", "C1", 3.0),
        ("Continuation A",  sys_a,  "-", "C0", 1.5),
        ("Continuation B",  sys_b,  "-", "C2", 1.5),
    ]

    def get_branch_energies(sys_obj):
        _mus = np.array([Y[-1] for Y in sys_obj.data["Y_s"]])
        _energies = np.array(sys_obj.data["energies"])
        return _mus, _energies

    return branch_styles, get_branch_energies


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Energy diagram (all branches)
    """)
    return


@app.cell(hide_code=True)
def _(branch_styles, get_branch_energies, plt):
    _fig, _ax = plt.subplots(figsize=(10, 5))
    for _label, _sys, _fmt, _color, _lw in branch_styles:
        _s, _e = get_branch_energies(_sys)
        _ax.plot(_s, _e, _fmt, lw=_lw, color=_color, label=_label)
    _ax.set_xlabel(r"$\mu$")
    _ax.set_ylabel("Potential energy")
    _ax.set_title("Energy diagram")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Displacement norm diagram (all branches)
    """)
    return


@app.cell(hide_code=True)
def _(branch_styles, get_crystal_data, plt):
    _fig, _ax = plt.subplots(figsize=(10, 5))
    for _label, _sys, _fmt, _color, _lw in branch_styles:
        _s, _d, _ = get_crystal_data(_sys)
        _ax.plot(_s, _d, _fmt, lw=_lw, color=_color, label=_label)
    _ax.set_xlabel(r"$\mu$")
    _ax.set_ylabel("Total displacement norm")
    _ax.set_title("Displacement norm vs $\\mu$")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Representative configurations

    3D scatter plots of the crystal at key points: the reference state,
    a configuration from Continuation A, and one from Continuation B.
    Atoms are coloured by displacement magnitude. We pick the entries which correspond to $\mu \approx 1$ and reveal the single atom rearrangement (Continuation A) and three atom rearrangement (Continuation B).
    """)
    return


@app.cell(hide_code=True)
def _(go, mo, n_atoms, np, qs_sys, sys_a, sys_b):
    _snapshots = [
        (qs_sys, 0,  "Reference (QS, $\\mu$ = {:.3f})"),
        (sys_a,  49, "Continuation A, idx 49 ($\\mu$ = {:.3f})"),
        (sys_b,  36, "Continuation B, idx 36 ($\\mu$ = {:.3f})"),
    ]

    _figs = []
    for _sys, _idx, _title_fmt in _snapshots:
        _Y = _sys.data["Y_s"][_idx]
        _U0 = _sys.U_0
        _dev = _Y[:-1].reshape(n_atoms, 3)
        _pos = _U0 + _dev
        _mu = _Y[-1]
        _dmag = np.linalg.norm(_dev, axis=1)

        _fig = go.Figure(data=[go.Scatter3d(
            x=_pos[:, 0], y=_pos[:, 1], z=_pos[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=_dmag,
                colorscale="Viridis",
                colorbar=dict(title="Disp."),
                cmin=0,
                cmax=max(np.max(_dmag), 0.01),
            ),
        )])
        _fig.update_layout(
            title=_title_fmt.format(_mu),
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=40, b=0),
            height=400,
        )
        _figs.append(_fig)

    mo.hstack([mo.ui.plotly(_f) for _f in _figs])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive explorer

    Pick a system and slide along the solution curve. The left panel
    shows the 3D atomic configuration (coloured by displacement or
    species); the right panel shows the energy curve with a red dot
    marking the current point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sys_dropdown = mo.ui.dropdown(
        options={"QS": "qs", "Continuation A": "cont_a", "Continuation B": "cont_b"},
        value="QS",
        label="System",
    )
    sys_dropdown
    return (sys_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    color_dropdown = mo.ui.dropdown(
        options={"Displacement magnitude": "disp", "Species": "species"},
        value="Displacement magnitude",
        label="Colour by",
    )
    color_dropdown
    return (color_dropdown,)


@app.cell(hide_code=True)
def _(cont_sys_map, mo, sys_dropdown):
    _active = cont_sys_map[sys_dropdown.value]
    _n_pts = len(_active.data["Y_s"])
    idx_slider = mo.ui.slider(
        start=0, stop=_n_pts - 1, step=1, value=0,
        label="Index along curve",
    )
    idx_slider
    return (idx_slider,)


@app.cell
def _(qs_sys, sys_a, sys_b):
    cont_sys_map = {"qs": qs_sys, "cont_a": sys_a, "cont_b": sys_b}
    return (cont_sys_map,)


@app.cell(hide_code=True)
def _(
    branch_styles,
    color_dropdown,
    cont_sys_map,
    get_branch_energies,
    go,
    idx_slider,
    mo,
    n_atoms,
    np,
    plt,
    sys_dropdown,
):
    _active = cont_sys_map[sys_dropdown.value]
    _idx = idx_slider.value
    _Y = _active.data["Y_s"][_idx]
    _U0 = _active.U_0
    _dev = _Y[:-1].reshape(n_atoms, 3)
    _pos = _U0 + _dev
    _mu = _Y[-1]

    # ── Left: 3D configuration (plotly) ──
    if color_dropdown.value == "disp":
        _dmag = np.linalg.norm(_dev, axis=1)
        _marker = dict(
            size=4,
            color=_dmag,
            colorscale="Viridis",
            colorbar=dict(title="Disp.", len=0.6),
            cmin=0,
            cmax=max(np.max(_dmag), 0.01),
        )
    else:
        # Species colouring — need to read types from LAMMPS
        _rng = np.random.default_rng(12345)
        _type_colors = np.ones(n_atoms)
        _type_colors[_rng.uniform(size=n_atoms) < 0.2] = 2
        _marker = dict(
            size=4,
            color=_type_colors,
            colorscale=[[0, "steelblue"], [0.5, "steelblue"], [0.5, "salmon"], [1, "salmon"]],
            cmin=1,
            cmax=2,
            colorbar=dict(title="Type", tickvals=[1, 2], ticktext=["1", "2"], len=0.6),
        )

    _fig3d = go.Figure(data=[go.Scatter3d(
        x=_pos[:, 0], y=_pos[:, 1], z=_pos[:, 2],
        mode="markers",
        marker=_marker,
    )])
    _fig3d.update_layout(
        title=f"$\\mu$ = {_mu:.4f}",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        width=500,
    )

    # ── Right: energy curve (matplotlib) ──
    _fig2d, _ax = plt.subplots(figsize=(6, 4.5))
    for _label, _sys, _fmt, _color, _lw in branch_styles:
        _s, _e = get_branch_energies(_sys)
        _ax.plot(_s, _e, _fmt, lw=_lw * 0.7, color=_color, alpha=0.5, label=_label)

    # Current point
    _cur_mus, _cur_e = get_branch_energies(_active)
    _ax.plot(_cur_mus[_idx], _cur_e[_idx], "o", ms=12, color="red",
             zorder=10, label="Current point")

    _ax.set_xlabel(r"$\mu$")
    _ax.set_ylabel("Potential energy")
    _ax.set_title("Energy diagram")
    _ax.legend(fontsize=7)
    _ax.grid(True, alpha=0.3)
    _fig2d.tight_layout()

    mo.hstack([mo.ui.plotly(_fig3d), _fig2d])
    return


if __name__ == "__main__":
    app.run()
