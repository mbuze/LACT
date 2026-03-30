# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "marimo-precompute>=0.2.3",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Demo 3: Subcritical pitchfork in a 2D crystal

    A 2D hexagonal Lennard-Jones crystal is compressed uniaxially.
    At a critical strain (~14.5 %) the symmetric (uniformly compressed) equilibrium
    becomes unstable and two **buckled** configurations emerge — a
    **pitchfork bifurcation**.

    The buckling mode is a vortex-like in-plane rotation of atoms about
    the centre of a lattice triangle. The two buckled branches correspond
    to opposite rotation senses. The bifurcation is **subcritical**: the
    buckled branches first extend *backward* in strain (with a fold/turning
    point) before continuing forward — creating a hysteresis region.

    In this notebook we will use LACT to capture this phenomenon by tracing three solution curves:

    | Curve | Description |
    |-------|-------------|
    | **QS** | Quasi-static loading starting from undeformed state |
    | **Continuation A** | Seeded just before the end of QS, continued backward — traces the buckled branch |
    | **Continuation B** | Seeded well before bifurcation, continued forward past it on the symmetric (unstable) branch |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We first load the necessary dependencies and some helper functions:
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from marimo_precompute import persistent_cache, prefetch_all
    try:
        from LACT.precomputed import PrecomputedSystem
    except ImportError:
        class PrecomputedSystem:
            def __init__(self, d):
                self.natoms = int(d["natoms"])
                self.U_0 = np.asarray(d["U_0"])
                self.data = {"Y_s": [np.asarray(y) for y in d["Y_s"]], "ds_s": list(d.get("ds_s", []))}
                if "energies" in d:
                    self.data["energies"] = list(d["energies"])

    return PrecomputedSystem, mo, np, persistent_cache, prefetch_all, plt


@app.cell
async def _(prefetch_all):
    await prefetch_all()
    return


@app.cell(hide_code=True)
def _(np, sys):
    def _find_triangle_atoms(U0):
        """3 atoms forming the central triangle (to the right of domain mean)."""
        _rc_mean = U0[:, :2].mean(axis=0)
        _dists = np.linalg.norm(U0[:, :2] - _rc_mean, axis=1)
        _idx4 = np.argsort(_dists)[:4]
        _xs = U0[_idx4, 0]
        return _idx4[np.argsort(_xs)[1:]]

    tri_idx = _find_triangle_atoms(sys.U_0)

    def circulation(sys_obj):
        """Compute circulation of displacement field about the tracking triangle centroid."""
        _U0 = sys_obj.U_0
        _rc0 = _U0[tri_idx, :2].mean(axis=0)
        strains, circ = [], []
        for Y in sys_obj.data["Y_s"]:
            strains.append(Y[-1])
            _dev = Y[:-1].reshape(-1, 3)
            _pos = _U0 + _dev
            _rc = _pos[tri_idx, :2].mean(axis=0)
            _rx = _pos[:, 0] - _rc[0]
            _ry = _pos[:, 1] - _rc[1]
            _ux = _dev[:, 0] - (_rc[0] - _rc0[0])
            _uy = _dev[:, 1] - (_rc[1] - _rc0[1])
            circ.append(np.sum(_rx * _uy - _ry * _ux))
        return np.array(strains), np.array(circ)

    return circulation, tri_idx


@app.cell(hide_code=True)
def _(np, sys, sys_a, sys_b):
    branch_styles = [
        ("QS",              sys,   "-", "C1", 6.0),
        ("Continuation A",  sys_a, "-", "C0", 1.5),
        ("Continuation B",  sys_b, "-", "C2", 2.5),
    ]

    def get_branch_data(sys_obj):
        strains = np.array([Y[-1] for Y in sys_obj.data["Y_s"]])
        energies = np.array(sys_obj.data["energies"])
        return strains, energies

    return branch_styles, get_branch_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then create a LAMMPS instance and pass on the simulation setup, together with the update command associated with the continuation parameter (uniaxial compression):
    """)
    return


@app.cell
def _():
    make_crystal = None
    try:
        from lammps import lammps
        from LACT import atom_cont_system

        _NX = 4

        def make_crystal():
            """Create a 2D hex LJ crystal, relaxed to zero stress."""
            _lmp = lammps(cmdargs=["-screen", "none"])
            _lmp.commands_string(
                f"""
                units         lj
                dimension     2
                boundary      p p p
                atom_style    atomic
                atom_modify   map yes
                lattice       hex 0.9165
                region        box block 0 {_NX} 0 {_NX} -0.5 0.5
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
                """
            )
            _box = _lmp.extract_box()
            _xlo, _xhi = _box[0][0], _box[1][0]
            _Lx = _xhi - _xlo

            def _update(strain):
                _dx = _Lx * strain / 2
                return f"change_box all x final {_xlo+_dx} {_xhi-_dx} units box"

            return atom_cont_system(_lmp, _update), _lmp
    except ImportError:
        pass

    return (make_crystal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because in this system we expect (at least) two meaningful continuation curves, we will need the ability to spawn a new continuation routine:
    """)
    return


@app.function
def seed_continuation(make_crystal, qs_Ys, idx, reverse=False):
    """Create a fresh system seeded for continuation from two consecutive QS points.

    Parameters
    ----------
    make_crystal : callable
        Factory that returns (atom_cont_system, lmp).
    qs_Ys : list of arrays
        The quasi-static solution list from the original QS run.
    idx : int
        Index of the *first* of the two consecutive points.
        Uses qs_Ys[idx] and qs_Ys[idx + 1].
    reverse : bool
        If False, the tangent points from idx -> idx+1 (increasing strain).
        If True, the tangent points from idx+1 -> idx (decreasing strain).

    Returns
    -------
    new_sys : atom_cont_system
        A fresh system with data["Y_s"] seeded and ready for
        new_sys.continuation_run(...).
    """
    import numpy as np
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


@app.cell
def _():
    strain_max = 0.16
    K = 1000
    strain_plot_min = 0.12  # only plot from this strain onward
    return K, strain_max, strain_plot_min


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now trace the quasi-static curve:
    """)
    return


@app.cell
def _(PrecomputedSystem, K, make_crystal, persistent_cache, strain_max):
    def _run_qs():
        _s, _lmp = make_crystal()
        _increment = strain_max / K
        _s.quasi_static_run(0.0, _increment, K + 1, verbose=False)
        _s.compute_energies()
        return {
            "natoms": _s.natoms, "U_0": _s.U_0,
            "Y_s": _s.data["Y_s"], "energies": _s.data["energies"],
        }
    with persistent_cache(name="demo3_qs"):
        qs_data = _run_qs()
    sys = PrecomputedSystem(qs_data)
    qs_Ys = sys.data["Y_s"]
    return qs_Ys, sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And let us plot the energy curve:
    """)
    return


@app.cell(hide_code=True)
def _(branch_styles, get_branch_data, plt, strain_plot_min):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _label, _sys, _fmt, _color, _lw in branch_styles[:1]:
        _s, _e = get_branch_data(_sys)
        _mask = _s >= strain_plot_min
        _ax.plot(_s[_mask], _e[_mask], _fmt, lw=_lw,
                 color=_color, label=_label)
    _ax.set_xlabel("Uniaxial strain")
    _ax.set_ylabel("Energy per atom")
    _ax.set_title("Energy diagram")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We observe that the QS curve admits two energy drops, related to passing a bifurcation point.

    The first smaller drop appears to correspond to an even more complicated bifurcating behaviour, which we do not pursue in this demo.

    We instead focus on the second larger drop and show that by seeding a reverse continuation run after the drop we are able to trace out the pitchfork bifurcation. This is done as follows:
    """)
    return


@app.cell(hide_code=True)
def _(PrecomputedSystem, make_crystal, persistent_cache, qs_Ys):
    def _run_cont_a():
        _s = seed_continuation(make_crystal, qs_Ys, -6, reverse=True)
        _s.continuation_run(
            n_iter=200,
            ds_default=0.01,
            ds_smallest=0.001,
            ds_largest=0.05,
            verbose=False,
            checkpoint_freq=0,
            cont_target=0.16,
            target_tol=0.001,
        )
        _s.data["Y_s"] = _s.data["Y_s"][:-2]
        _s.compute_energies()
        return {
            "natoms": _s.natoms, "U_0": _s.U_0,
            "Y_s": _s.data["Y_s"], "energies": _s.data["energies"],
        }
    with persistent_cache(name="demo3_cont_a"):
        cont_a_data = _run_cont_a()
    sys_a = PrecomputedSystem(cont_a_data)
    return (sys_a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we also start another continuation curve before the first drop to show that the symmetric compressed state remains an equilibrium configuration (but changes from a local minimum to a saddle point):
    """)
    return


@app.cell(hide_code=True)
def _(PrecomputedSystem, make_crystal, persistent_cache, qs_Ys):
    def _run_cont_b():
        _s = seed_continuation(make_crystal, qs_Ys, 750, reverse=False)
        _s.continuation_run(
            n_iter=100,
            ds_default=0.001,
            ds_smallest=0.001,
            ds_largest=0.005,
            verbose=False,
            checkpoint_freq=0,
            cont_target=0.16,
            target_tol=0.005,
        )
        _s.compute_energies()
        return {
            "natoms": _s.natoms, "U_0": _s.U_0,
            "Y_s": _s.data["Y_s"], "energies": _s.data["energies"],
        }
    with persistent_cache(name="demo3_cont_b"):
        cont_b_data = _run_cont_b()
    sys_b = PrecomputedSystem(cont_b_data)
    return (sys_b,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Energy vs strain

    We can now plot the three curves together:
    """)
    return


@app.cell(hide_code=True)
def _(branch_styles, get_branch_data, plt, strain_plot_min):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _label, _sys, _fmt, _color, _lw in branch_styles:
        _s, _e = get_branch_data(_sys)
        _mask = _s >= strain_plot_min
        _ax.plot(_s[_mask], _e[_mask], _fmt, lw=_lw,
                 color=_color, label=_label)
    _ax.set_xlabel("Uniaxial strain")
    _ax.set_ylabel("Energy per atom")
    _ax.set_title("Energy diagram")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The buckled branch (blue) exhibits interesting behaviour: it seems to admit a genuine fold point followed by turning in on itself, because we start tracing the buckle occurring in the other orientation. This will be properly illuminated in the bifurcation diagram below. The symmetric branch (green) is continued past the
    bifurcation using LACT's arclength solver, something that a QS approach cannot do.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bifurcation diagram

    The **circulation** $\Gamma = \sum_j (\mathbf{r}_j - \mathbf{r}_c) \times \mathbf{u}_j \cdot \hat{z}$
    measures the net rotational displacement about the vortex centre
    (tracked as the centroid of a central lattice triangle in the deformed
    configuration). It is zero on the symmetric branch and takes opposite
    signs for the two vortex orientations — making the subcritical
    pitchfork structure clearly visible.
    """)
    return


@app.cell(hide_code=True)
def _(branch_styles, circulation, plt, strain_plot_min):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _label, _sys, _fmt, _color, _lw in branch_styles:
        _s, _c = circulation(_sys)
        _mask = _s >= strain_plot_min
        _ax.plot(_s[_mask], _c[_mask], _fmt, lw=_lw,
                 color=_color, label=_label)
    _ax.set_xlabel("Uniaxial strain")
    _ax.set_ylabel("Circulation $\\Gamma$")
    _ax.set_title("Bifurcation diagram (subcritical pitchfork)")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Representative configurations

    A selection of configurations at key points along the solution curves.
    Red arrows show displacements from the undeformed reference. The orange
    cross marks the vortex centre (deformed triangle centroid).
    """)
    return


@app.cell(hide_code=True)
def _(np, plt, sys, sys_a, tri_idx):
    _snapshots = [
        (sys,   0,                              "Undeformed"),
        (sys,   len(sys.data["Y_s"]) // 2,     "Symmetric (mid)"),
        (sys,   len(sys.data["Y_s"]) - 1,      "Symmetric (max strain)"),
        (sys_a, len(sys_a.data["Y_s"]) // 4,   "Buckled A (early)"),
        (sys_a, len(sys_a.data["Y_s"]) // 2,   "Buckled A (mid)"),
        (sys_a, len(sys_a.data["Y_s"]) - 1,    "Buckled A (late)"),
    ]

    _ncols = 3
    _nrows = 2
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(5 * _ncols, 5 * _nrows))
    _axes_flat = _axes.flatten()

    for _ax, (_sys, _idx, _title) in zip(_axes_flat, _snapshots):
        _n = _sys.natoms
        _U0 = _sys.U_0
        _Y = _sys.data["Y_s"][_idx]
        _dev = _Y[:-1].reshape(_n, 3)
        _pos = _U0 + _dev

        _ax.scatter(_pos[:, 0], _pos[:, 1], s=80, c="C0", edgecolors="k", lw=0.5)

        _rc = _pos[tri_idx, :2].mean(axis=0)
        _ax.scatter(_rc[0], _rc[1], s=120, c="orange", marker="x", zorder=5, linewidths=2)

        for _j in range(_n):
            _d = _pos[_j, :2] - _U0[_j, :2]
            if np.linalg.norm(_d) > 1e-3:
                _ax.annotate(
                    "", xy=(_pos[_j, 0], _pos[_j, 1]),
                    xytext=(_U0[_j, 0], _U0[_j, 1]),
                    arrowprops=dict(arrowstyle="->", color="r", lw=0.8),
                )

        _ax.set_title(f"{_title}\n(strain = {_Y[-1]:.4f})")
        _ax.set_aspect("equal")
        _ax.grid(True, alpha=0.2)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive explorer

    Pick a branch and slide along the solution curve. The left panel shows
    the atomic configuration; the right panel shows the bifurcation diagram
    with a red dot marking the current point.
    """)
    return


@app.cell(hide_code=True)
def _(mo, sys, sys_a, sys_b):
    _systems = {
        "QS": sys,
        "Continuation A": sys_a,
        "Continuation B": sys_b,
    }
    sys_picker = mo.ui.dropdown(
        options=_systems,
        label="System",
        value="QS",
    )
    sys_picker
    return (sys_picker,)


@app.cell(hide_code=True)
def _(mo, np, strain_plot_min, sys, sys_picker):
    _sel = sys_picker.value
    _n_pts = len(_sel.data["Y_s"])
    # For QS system, start slider at strain_plot_min region
    if _sel is sys:
        _strains = np.array([Y[-1] for Y in _sel.data["Y_s"]])
        _start = int(np.searchsorted(_strains, strain_plot_min))
    else:
        _start = 0
    idx_slider = mo.ui.slider(
        start=_start, stop=_n_pts - 1, step=1, value=_start,
        label="Index along curve",
    )
    idx_slider
    return (idx_slider,)


@app.cell(hide_code=True)
def _(
    branch_styles,
    circulation,
    idx_slider,
    np,
    plt,
    strain_plot_min,
    sys_picker,
    tri_idx,
):
    _sys = sys_picker.value
    _idx = idx_slider.value
    _n = _sys.natoms
    _U0 = _sys.U_0
    _Y = _sys.data["Y_s"][_idx]
    _dev = _Y[:-1].reshape(_n, 3)
    _pos = _U0 + _dev

    _fig, (_ax_l, _ax_r) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: configuration ──
    _ax_l.scatter(_pos[:, 0], _pos[:, 1], s=80, c="C0", edgecolors="k", lw=0.5)

    _rc = _pos[tri_idx, :2].mean(axis=0)
    _ax_l.scatter(_rc[0], _rc[1], s=120, c="orange", marker="x", zorder=5, linewidths=2)

    for _j in range(_n):
        _d = _pos[_j, :2] - _U0[_j, :2]
        if np.linalg.norm(_d) > 1e-3:
            _ax_l.annotate(
                "", xy=(_pos[_j, 0], _pos[_j, 1]),
                xytext=(_U0[_j, 0], _U0[_j, 1]),
                arrowprops=dict(arrowstyle="->", color="r", lw=0.8),
            )

    _ax_l.set_title(f"Configuration — strain = {_Y[-1]:.4f}")
    _ax_l.set_aspect("equal")
    _ax_l.grid(True, alpha=0.2)

    # ── Right: bifurcation diagram with current point ──
    for _label, _s_obj, _fmt, _color, _lw in branch_styles:
        _s, _c = circulation(_s_obj)
        _mask = _s >= strain_plot_min
        _ax_r.plot(_s[_mask], _c[_mask], _fmt, lw=_lw * 0.7,
                   color=_color, alpha=0.5, label=_label)

    # Current point
    _cur_s, _cur_c = circulation(_sys)
    _ax_r.plot(_cur_s[_idx], _cur_c[_idx], "o", ms=12, color="red",
               zorder=10, label="Current point")

    _ax_r.set_xlabel("Uniaxial strain")
    _ax_r.set_ylabel("Circulation $\\Gamma$")
    _ax_r.set_title("Bifurcation diagram")
    _ax_r.legend(fontsize=8)
    _ax_r.grid(True, alpha=0.3)

    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
