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
    # Demo 2: Morse chain — which bond snaps?

    A chain of 6 atoms connected by Morse bonds is pulled apart by a
    force applied to the right end. Each nearest-neighbour bond has a
    different well depth $D$, giving each a different maximum sustainable
    tension $F_{\max} = D\alpha/2$. The **weakest bond** (smallest $D$)
    reaches its fold first and snaps.

    In this notebook we use LACT to trace the force–extension S-curve
    through the fold, revealing the full unstable branch that
    quasi static loading cannot reach. The bond-length plot confirms that only
    the weakest bond snaps — the others remain near equilibrium.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We first load the necessary dependencies and define some helper functions, which include creating a LAMMPS instance, passing on the simulation setup and the update command associated with the continuation parameter (magnitude of the force applied to the atom on the right):
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


@app.cell
def _():
    n_atoms = 6
    alpha = 6.0       # Morse width parameter
    r0 = 1.12         # Morse equilibrium distance
    # Well depths per bond — ordered weakest to strongest
    bond_info = [
        ((1, 2), 0.85),  # weakest — snaps first
        ((2, 3), 0.90),
        ((3, 4), 0.95),
        ((4, 5), 1.00),
        ((5, 6), 1.05),  # strongest
    ]

    make_chain = None
    try:
        from lammps import lammps
        from LACT import atom_cont_system

        def make_chain():
            _lmp = lammps(cmdargs=["-screen", "none"])
            _lmp.commands_string(
                f"""
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
                """
            )

            def update_command(force):
                return f"""
                unfix pull
                fix pull right_end addforce {force} 0.0 0.0
                fix_modify pull energy yes
                """

            return atom_cont_system(_lmp, update_command)
    except ImportError:
        pass

    return alpha, bond_info, make_chain, n_atoms, r0


@app.cell
def _(alpha, n_atoms, np, r0):
    def get_chain_data(sys_obj):
        """Extract extensions, forces, and bond lengths from a chain system."""
        _U0 = sys_obj.U_0
        _extensions, _forces, _bonds = [], [], []
        for _Y in sys_obj.data["Y_s"]:
            _pos = _U0 + _Y[:-1].reshape(n_atoms, 3)
            _extensions.append(_pos[-1, 0] - _U0[-1, 0])
            _forces.append(_Y[-1])
            _bonds.append([_pos[i + 1, 0] - _pos[i, 0] for i in range(n_atoms - 1)])
        return np.array(_extensions), np.array(_forces), np.array(_bonds)

    # Analytical fold distance and fold forces per bond
    r_fold = r0 + np.log(2) / alpha
    return get_chain_data, r_fold


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quasi-static loading (for comparison)

    First we try the naive approach: ramp the applied force in small steps
    from 0 to 4.0 using LAMMPS energy minimisation at each step.
    This works up to the fold ($F_{\max} = 0.85 \times 6/2 = 2.55$),
    where the bond snaps and the atom flies to the pair-potential cutoff.
    """)
    return


@app.cell
def _(PrecomputedSystem, get_chain_data, make_chain, persistent_cache):
    def _run_qs():
        _s = make_chain()
        try:
            _s.quasi_static_run(0.0, 0.02, 200, verbose=False)
        except Exception:
            pass  # LAMMPS crashes past the fold — keep the points we got
        return {"natoms": _s.natoms, "U_0": _s.U_0, "Y_s": _s.data["Y_s"]}
    with persistent_cache(name="demo2_qs"):
        qs_data = _run_qs()
    qs_sys = PrecomputedSystem(qs_data)
    qs_extensions, qs_forces, qs_bonds = get_chain_data(qs_sys)
    return qs_bonds, qs_extensions, qs_forces, qs_sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arclength continuation

    Now we use the continuation routine in LACT. A short quasi-static
    ramp seeds the path with a few points, then `continuation_run`
    traces smoothly through the fold and onto the unstable branch.
    """)
    return


@app.cell
def _(PrecomputedSystem, get_chain_data, make_chain, persistent_cache):
    def _run_cont():
        _s = make_chain()
        _s.quasi_static_run(0.0, 0.5, 5, verbose=False)
        _n_qs = len(_s.data["Y_s"])
        _s.continuation_run(
            n_iter=200,
            ds_default=0.01,
            ds_smallest=0.001,
            ds_largest=0.05,
            verbose=False,
            checkpoint_freq=0,
            cont_target=0.5,
            target_tol=0.1,
        )
        return {
            "natoms": _s.natoms, "U_0": _s.U_0,
            "Y_s": _s.data["Y_s"], "n_qs": _n_qs,
        }
    with persistent_cache(name="demo2_cont"):
        cont_data = _run_cont()
    cont_sys = PrecomputedSystem(cont_data)
    n_qs = cont_data["n_qs"]
    cont_extensions, cont_forces, cont_bonds = get_chain_data(cont_sys)
    return cont_bonds, cont_extensions, cont_forces, cont_sys, n_qs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can compare the two approaches visually:
    """)
    return


@app.cell(hide_code=True)
def _(cont_extensions, cont_forces, n_qs, plt, qs_extensions, qs_forces):
    _fig, _ax = plt.subplots(figsize=(7, 5))

    _ax.plot(qs_forces, qs_extensions, "s", ms=5, color="C2", alpha=0.7,
             label="Quasi-static (load-stepping)")
    _ax.plot(cont_forces[n_qs:], cont_extensions[n_qs:],
             "-", lw=1.5, color="C0", label="Arclength continuation")

    _ax.set_xlabel("Applied force $F$")
    _ax.set_ylabel("Extension (displacement of atom 6)")
    _ax.set_title("Force–extension through the first fold")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(
    alpha,
    bond_info,
    cont_bonds,
    cont_forces,
    n_qs,
    plt,
    qs_bonds,
    qs_forces,
    r0,
    r_fold,
):
    _fig, (_ax_top, _ax_bot) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for _idx, ((_i, _j), _D, ) in enumerate(bond_info):
        _c = f"C{_idx}"
        _Fmax = _D * alpha / 2
        _ax_top.plot(qs_forces, qs_bonds[:, _idx],
                     "s", ms=4, color=_c, alpha=0.5)
        _ax_top.plot(cont_forces[n_qs:], cont_bonds[n_qs:, _idx],
                     "-", lw=1.5, color=_c,
                     label=f"Bond {_i}–{_j} ($D$={_D:.2f}, $F_{{max}}$={_Fmax:.2f})")

    _ax_top.axhline(r_fold, color="k", ls="--", lw=0.8, alpha=0.5,
                     label=f"Fold at $r^*$={r_fold:.3f}")
    _ax_top.set_ylabel(r"Bond length $r$")
    _ax_top.set_title("Bond lengths vs. applied force")
    _ax_top.legend(fontsize=7, loc="upper left")
    _ax_top.grid(True, alpha=0.3)

    for _idx, ((_i, _j), _D) in enumerate(bond_info):
        _c = f"C{_idx}"
        _ax_bot.plot(qs_forces, qs_bonds[:, _idx] - r0,
                     "s", ms=4, color=_c, alpha=0.5)
        _ax_bot.plot(cont_forces[n_qs:], cont_bonds[n_qs:, _idx] - r0,
                     "-", lw=1.5, color=_c, label=f"Bond {_i}–{_j}")
    _ax_bot.set_xlabel("Applied force $F$")
    _ax_bot.set_ylabel(r"Bond stretch $r - r_0$")
    _ax_bot.set_title("Bond stretch: weakest bond absorbs most deformation")
    _ax_bot.legend(fontsize=7)
    _ax_bot.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpreting the results

    **Force–extension curve:** The green squares (quasi-static
    loading) follow the stable branch up to the fold at
    $F_{\max} = 2.55$, where the LAMMPS minimiser crashes — an atom
    is lost. The blue line (arclength continuation) traces smoothly
    through the fold onto the unstable branch (decreasing force).

    **Bond lengths:** Under increasing applied force, all bonds stretch,
    but the weakest bond ($D=0.85$) stretches the most. At the fold, it
    passes the inflection point $r^*$ and begins to snap — its length
    increases rapidly while the other bonds contract (since the total
    force is now decreasing). This confirms that **disorder determines
    which bond breaks**: the weakest link in the chain.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Representative configurations

    The chain at key points along the continuation curve: near equilibrium,
    at the fold, and on the unstable branch. The arrow shows the applied
    force on the rightmost atom.
    """)
    return


@app.cell(hide_code=True)
def _(cont_extensions, cont_forces, cont_sys, n_atoms, n_qs, np, plt):
    _n_cont = len(cont_sys.data["Y_s"])
    _i_fold = n_qs + np.argmax(cont_forces[n_qs:])
    _snapshots = [
        (0,              "Equilibrium (F=0)"),
        (_i_fold,        "At the fold"),
        (_n_cont - 1,    "Unstable branch"),
    ]

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 3.5))
    _U0 = cont_sys.U_0
    _max_force = np.max(np.abs(cont_forces))

    for _ax, (_idx, _title) in zip(_axes, _snapshots):
        _Y = cont_sys.data["Y_s"][_idx]
        _dev = _Y[:-1].reshape(n_atoms, 3)
        _pos = _U0 + _dev
        _F = _Y[-1]
        _ext = cont_extensions[_idx]

        # Atoms: fixed = grey, others = C0
        _colors = ["grey"] + ["C0"] * (n_atoms - 1)
        _ax.scatter(_pos[:, 0], [0] * n_atoms, s=250, c=_colors,
                    edgecolors="k", lw=0.8, zorder=5)
        _ax.annotate("fixed", (_pos[0, 0], -0.2), ha="center", fontsize=7, color="grey")
        _ax.annotate("pulled", (_pos[-1, 0], -0.2), ha="center", fontsize=7, color="C0")

        # Bond lines
        for _i in range(n_atoms - 1):
            _ax.plot([_pos[_i, 0], _pos[_i + 1, 0]], [0, 0],
                     "k--", lw=0.6, alpha=0.4)

        # Force arrow
        if abs(_F) > 0.01:
            _arrow_len = 0.4 * _F / _max_force
            _ax.annotate("", xy=(_pos[-1, 0] + _arrow_len, 0),
                         xytext=(_pos[-1, 0], 0),
                         arrowprops=dict(arrowstyle="->", color="red", lw=2))

        _ax.set_title(f"{_title}\next = {_ext:.3f}, $F = {_F:.2f}$")
        _ax.set_xlim(_pos[0, 0] - 0.4, _pos[-1, 0] + 0.8)
        _ax.set_ylim(-0.4, 0.4)
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

    Pick a system and slide along the solution curve. The left panel
    shows the chain configuration; the right panel shows the
    force–extension curve with a red dot marking the current point.
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
    cont_extensions,
    cont_forces,
    cont_sys,
    get_chain_data,
    idx_slider,
    n_atoms,
    n_qs,
    np,
    plt,
    qs_extensions,
    qs_forces,
    qs_sys,
    sys_dropdown,
):
    _systems = {"qs": qs_sys, "cont": cont_sys}
    _active = _systems[sys_dropdown.value]
    _idx = idx_slider.value
    _Y = _active.data["Y_s"][_idx]
    _U0 = _active.U_0
    _dev = _Y[:-1].reshape(n_atoms, 3)
    _pos = _U0 + _dev
    _F = _Y[-1]
    _act_ext, _, _ = get_chain_data(_active)
    _ext = _act_ext[_idx]
    _max_force = max(np.max(np.abs(cont_forces)), np.max(np.abs(qs_forces)))

    _fig, (_ax_l, _ax_r) = plt.subplots(1, 2, figsize=(14, 5),
                                         gridspec_kw={"width_ratios": [1, 1.5]})

    # ── Left: chain configuration ──
    _colors = ["grey"] + ["C0"] * (n_atoms - 1)
    _ax_l.scatter(_pos[:, 0], [0] * n_atoms, s=350, c=_colors,
                  edgecolors="k", lw=0.8, zorder=5)
    _ax_l.annotate("fixed", (_pos[0, 0], -0.25), ha="center", fontsize=8, color="grey")
    _ax_l.annotate("pulled", (_pos[-1, 0], -0.25), ha="center", fontsize=8, color="C0")

    for _i in range(n_atoms - 1):
        _ax_l.plot([_pos[_i, 0], _pos[_i + 1, 0]], [0, 0],
                   "k--", lw=0.6, alpha=0.4)

    if abs(_F) > 0.01:
        _arrow_len = 0.4 * _F / _max_force
        _ax_l.annotate("", xy=(_pos[-1, 0] + _arrow_len, 0),
                        xytext=(_pos[-1, 0], 0),
                        arrowprops=dict(arrowstyle="->", color="red", lw=2.5))

    _ax_l.set_title(f"ext = {_ext:.3f}, $F = {_F:.2f}$")
    _ax_l.set_xlim(_pos[0, 0] - 0.4, _pos[-1, 0] + 0.8)
    _ax_l.set_ylim(-0.5, 0.5)
    _ax_l.set_aspect("equal")
    _ax_l.set_yticks([])
    _ax_l.grid(True, alpha=0.2)

    # ── Right: force-extension with current point ──
    _ax_r.plot(qs_forces, qs_extensions, "s", ms=4, color="C2", alpha=0.5,
               label="Quasi-static")
    _ax_r.plot(cont_forces[n_qs:], cont_extensions[n_qs:],
               "-", lw=1.2, color="C0", alpha=0.5, label="Continuation")

    _ax_r.plot(_F, _ext, "o", ms=12, color="red", zorder=10, label="Current point")

    _ax_r.set_xlabel("Applied force $F$")
    _ax_r.set_ylabel("Extension (displacement of atom 6)")
    _ax_r.set_title("Force–extension curve")
    _ax_r.legend(fontsize=8)
    _ax_r.grid(True, alpha=0.3)

    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
