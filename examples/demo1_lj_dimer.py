import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Demo 1: LJ dimer — tracing a force-extension fold

    Two Lennard-Jones atoms are pulled apart by fixing one atom and applying a force to the other.
    The force-extension curve has a **fold point** bifurcation at the inflection
    point of the LJ potential ($r \approx 1.245\,\sigma$). Beyond this
    point, the bond snaps and so any configuration where the two atoms are sufficiently apart is a degenerate equilibrium. Standard quasi-static loading can capture the snapping, but is unable to trace the unstable equilibrium.

    **Arclength continuation** parameterises the solution path by its
    arc length rather than by the applied force, letting it smoothly
    traverse the fold and trace the full S-curve onto the unstable
    segment.

    Given the simplicity of the system, we actually have an analytical solution tracing the stable and unstable equilibria.

    In this notebook, we will use LACT to trace it numerically using a LAMMPS instance and LACT continuation wrapper.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We first load the necessary dependencies and define some helper functions, which include creating a LAMMPS instance, passing on the simulation setup and the update command associated with the continuation parameter (magnitude of the force applied to an atom):
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        from LACT.precomputed import PrecomputedSystem
        load_json = None
    except ImportError:
        import json as _json

        async def load_json(url):
            from pyodide.http import pyfetch
            from js import self as _js_self
            # Worker runs from assets/ subdir, go up to site root
            base = str(_js_self.location.href).rsplit("/", 2)[0] + "/"
            resp = await pyfetch(base + url)
            return _json.loads(await resp.string())

        class PrecomputedSystem:
            def __init__(self, d):
                self.natoms = int(d["natoms"])
                self.U_0 = np.asarray(d["U_0"])
                self.data = {"Y_s": [np.asarray(y) for y in d["Y_s"]], "ds_s": list(d.get("ds_s", []))}
                if "energies" in d:
                    self.data["energies"] = list(d["energies"])

    try:
        from lammps import lammps
        from LACT import atom_cont_system
        HAVE_LAMMPS = True
    except ImportError:
        lammps = None
        atom_cont_system = None
        HAVE_LAMMPS = False

    return HAVE_LAMMPS, PrecomputedSystem, atom_cont_system, lammps, load_json, mo, np, plt


@app.cell
def _(HAVE_LAMMPS, atom_cont_system, lammps):
    if HAVE_LAMMPS:
        def make_dimer():
            _lmp = lammps(cmdargs=["-screen", "none"])
            _lmp.commands_string(
                """
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
                """
            )

            def update_command(force):
                return f"""
                unfix pull
                fix pull mobile addforce {force} 0.0 0.0
                fix_modify pull energy yes
                """

            return atom_cont_system(_lmp, update_command)
    else:
        make_dimer = None

    return (make_dimer,)


@app.cell
def _(np):
    def get_separation(sys_obj):
        """Extract separations and forces from a dimer system's solution data."""
        _U0 = sys_obj.U_0
        _seps, _forces = [], []
        for _Y in sys_obj.data["Y_s"]:
            _pos = _Y[:-1].reshape(-1, 3)
            _r = (_U0[1, 0] + _pos[1, 0]) - (_U0[0, 0] + _pos[0, 0])
            _seps.append(_r)
            _forces.append(_Y[-1])
        return np.array(_seps), np.array(_forces)

    # Analytical LJ force-extension curve
    r_anal = np.linspace(1.1, 2.5, 500)
    F_anal = 24.0 * (1.0 / r_anal**7 - 2.0 / r_anal**13)
    return F_anal, get_separation, r_anal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quasi-static loading (for comparison)

    First we try the naive approach: ramp the applied force in small steps
    from 0 to 3.0 using LAMMPS energy minimisation at each step.
    This works up to the fold ($F \approx 2.40$) and then the bond snaps and the atom flies to the pair-potential cutoff, landing in the degenerate equilibrium state.
    """)
    return


@app.cell
async def _(HAVE_LAMMPS, PrecomputedSystem, load_json, get_separation, make_dimer):
    if HAVE_LAMMPS:
        qs_sys = make_dimer()
        qs_sys.quasi_static_run(0.0, 0.1, 30, verbose=False)
    else:
        qs_sys = PrecomputedSystem(await load_json("data/demo1_qs.json"))
    qs_seps, qs_forces = get_separation(qs_sys)
    return qs_forces, qs_seps, qs_sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arclength continuation

    Now we use the continuation routine in LACT. A short quasi-static ramp seeds the path with a few points, then `continuation_run` takes over and traces smoothly through
    the fold and back to zero force on the unstable branch.
    """)
    return


@app.cell
async def _(HAVE_LAMMPS, PrecomputedSystem, load_json, get_separation, make_dimer):
    if HAVE_LAMMPS:
        cont_sys = make_dimer()
        cont_sys.quasi_static_run(0.0, 0.5, 5, verbose=False)
        n_qs = len(cont_sys.data["Y_s"])

        cont_sys.continuation_run(
            n_iter=100,
            ds_default=0.01,
            ds_smallest=0.001,
            ds_largest=1.0,
            verbose=False,
            checkpoint_freq=0,
            cont_target=0.0,
            target_tol=0.05,
        )
    else:
        cont_sys = PrecomputedSystem(await load_json("data/demo1_cont.json"))
        n_qs = 5

    cont_seps, cont_forces = get_separation(cont_sys)
    return cont_forces, cont_seps, cont_sys, n_qs


@app.cell(hide_code=True)
def _(F_anal, cont_forces, cont_seps, n_qs, plt, qs_forces, qs_seps, r_anal):
    _fig, _ax = plt.subplots(figsize=(7, 5))

    _ax.plot(F_anal, r_anal, "-", lw=5, color="C1", label="Analytical LJ")
    _ax.plot(qs_forces, qs_seps, "s", ms=5, color="C2", alpha=0.7,
             label="Quasi-static (load-stepping)")
    _ax.plot(cont_forces[n_qs:], cont_seps[n_qs:],
             "-", lw=1.5, color="C0", label="Arclength continuation")

    _i_max = F_anal.argmax()
    _ax.plot(F_anal[_i_max], r_anal[_i_max], "r*", ms=12, zorder=5,
             label="Fold (spinodal)")

    _ax.set_xlabel(r"Applied force $F$")
    _ax.set_ylabel(r"Separation $r / \sigma$")
    _ax.set_title("Force-extension curve: LJ dimer pull")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What happened at the fold?

    The analytical curve (orange) shows that the applied force required to
    hold two LJ atoms at separation $r$ has a **maximum** at the inflection
    point of the potential ($r^* \approx 1.245\,\sigma$,
    $F_{\max} \approx 2.40\,\varepsilon/\sigma$).

    For applied forces below that threshold, we have two possible equilibria: a stable one and an unstable one. At the bifurcation point they collide and beyond it, the bond snaps and atoms become separate beyond their interaction radius.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Representative configurations

    The dimer at key points along the continuation curve: near equilibrium,
    at the fold, and on the unstable (repulsive) branch. The arrow shows
    the applied force direction and magnitude.
    """)
    return


@app.cell(hide_code=True)
def _(cont_forces, cont_seps, cont_sys, n_qs, np, plt):
    _n_cont = len(cont_sys.data["Y_s"])
    _i_fold = n_qs + np.argmax(cont_forces[n_qs:])
    _snapshots = [
        (0,              "Equilibrium (F=0)"),
        (_i_fold,        "At the fold"),
        (_n_cont - 1,    "Unstable branch"),
    ]

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 3))
    _U0 = cont_sys.U_0
    _max_force = np.max(np.abs(cont_forces))

    for _ax, (_idx, _title) in zip(_axes, _snapshots):
        _Y = cont_sys.data["Y_s"][_idx]
        _dev = _Y[:-1].reshape(-1, 3)
        _pos = _U0 + _dev
        _F = _Y[-1]
        _r = cont_seps[_idx]

        # Atoms
        _ax.scatter(_pos[:, 0], [0, 0], s=300, c=["grey", "C0"],
                    edgecolors="k", lw=1, zorder=5)
        _ax.annotate("fixed", (_pos[0, 0], -0.15), ha="center", fontsize=8, color="grey")
        _ax.annotate("mobile", (_pos[1, 0], -0.15), ha="center", fontsize=8, color="C0")

        # Force arrow
        if abs(_F) > 0.01:
            _arrow_len = 0.3 * _F / _max_force
            _ax.annotate("", xy=(_pos[1, 0] + _arrow_len, 0),
                         xytext=(_pos[1, 0], 0),
                         arrowprops=dict(arrowstyle="->", color="red", lw=2))

        # Bond line
        _ax.plot([_pos[0, 0], _pos[1, 0]], [0, 0], "k--", lw=0.8, alpha=0.5)

        _ax.set_title(f"{_title}\n$r = {_r:.3f}\\sigma$, $F = {_F:.2f}$")
        _ax.set_xlim(_pos[0, 0] - 0.3, _pos[1, 0] + 0.5)
        _ax.set_ylim(-0.3, 0.3)
        _ax.set_aspect("equal")
        _ax.set_yticks([])
        _ax.grid(True, alpha=0.2)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive explorer

    Slide along the continuation curve. The left panel shows the dimer
    configuration; the right panel shows the force-extension curve with a
    red dot marking the current point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sys_dropdown = mo.ui.dropdown(
        options={"Quasi-static": "qs", "Continuation": "cont"},
        value="Continuation",
        label="System",
    )
    sys_dropdown
    return (sys_dropdown,)


@app.cell(hide_code=True)
def _(cont_sys, mo, qs_sys, sys_dropdown):
    _systems = {"qs": qs_sys, "cont": cont_sys}
    _n_pts = len(_systems[sys_dropdown.value].data["Y_s"])
    idx_slider = mo.ui.slider(
        start=0, stop=_n_pts - 1, step=1, value=0,
        label="Index along curve",
    )
    idx_slider
    return (idx_slider,)


@app.cell(hide_code=True)
def _(
    F_anal,
    cont_forces,
    cont_seps,
    cont_sys,
    get_separation,
    idx_slider,
    n_qs,
    np,
    plt,
    qs_forces,
    qs_seps,
    qs_sys,
    r_anal,
    sys_dropdown,
):
    _systems = {"qs": qs_sys, "cont": cont_sys}
    _active = _systems[sys_dropdown.value]
    _idx = idx_slider.value
    _Y = _active.data["Y_s"][_idx]
    _U0 = _active.U_0
    _dev = _Y[:-1].reshape(-1, 3)
    _pos = _U0 + _dev
    _F = _Y[-1]
    _act_seps, _ = get_separation(_active)
    _r = _act_seps[_idx]
    _max_force = max(np.max(np.abs(cont_forces)), np.max(np.abs(qs_forces)))

    _fig, (_ax_l, _ax_r) = plt.subplots(1, 2, figsize=(14, 5),
                                         gridspec_kw={"width_ratios": [1, 1.5]})

    # ── Left: dimer configuration ──
    _ax_l.scatter(_pos[:, 0], [0, 0], s=400, c=["grey", "C0"],
                  edgecolors="k", lw=1, zorder=5)
    _ax_l.annotate("fixed", (_pos[0, 0], -0.2), ha="center", fontsize=9, color="grey")
    _ax_l.annotate("mobile", (_pos[1, 0], -0.2), ha="center", fontsize=9, color="C0")

    if abs(_F) > 0.01:
        _arrow_len = 0.4 * _F / _max_force
        _ax_l.annotate("", xy=(_pos[1, 0] + _arrow_len, 0),
                        xytext=(_pos[1, 0], 0),
                        arrowprops=dict(arrowstyle="->", color="red", lw=2.5))

    _ax_l.plot([_pos[0, 0], _pos[1, 0]], [0, 0], "k--", lw=0.8, alpha=0.5)
    _ax_l.set_title(f"$r = {_r:.3f}\\sigma$, $F = {_F:.2f}$")
    _ax_l.set_xlim(_pos[0, 0] - 0.4, max(_pos[1, 0] + 0.6, 2.0))
    _ax_l.set_ylim(-0.4, 0.4)
    _ax_l.set_aspect("equal")
    _ax_l.set_yticks([])
    _ax_l.grid(True, alpha=0.2)

    # ── Right: force-extension with current point ──
    _ax_r.plot(F_anal, r_anal, "-", lw=6, color="C1", label="Analytical LJ")
    _ax_r.plot(qs_forces, qs_seps, "s", ms=4, color="C2", alpha=0.5,
               label="Quasi-static")
    _ax_r.plot(cont_forces[n_qs:], cont_seps[n_qs:],
               "-", lw=1.2, color="C0", alpha=0.5, label="Continuation")

    _ax_r.plot(_F, _r, "o", ms=12, color="red", zorder=10, label="Current point")

    _ax_r.set_xlabel(r"Applied force $F$")
    _ax_r.set_ylabel(r"Separation $r / \sigma$")
    _ax_r.set_title("Force-extension curve")
    _ax_r.legend(fontsize=8)
    _ax_r.grid(True, alpha=0.3)

    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
