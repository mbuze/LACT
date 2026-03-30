"""Lightweight stand-in for atom_cont_system using cached data."""

import numpy as np


class PrecomputedSystem:
    """Wraps a dict (from ``mo.persistent_cache``) into the same interface
    that the visualization cells expect from ``atom_cont_system``.

    Attributes: ``natoms``, ``U_0``, ``data["Y_s"]``, ``data["energies"]``,
    ``data["ds_s"]``.
    """

    def __init__(self, d):
        self.natoms = int(d["natoms"])
        self.U_0 = np.asarray(d["U_0"])
        self.data = {
            "Y_s": [np.asarray(y) for y in d["Y_s"]],
            "ds_s": list(d["ds_s"]) if "ds_s" in d else [],
        }
        if "energies" in d:
            self.data["energies"] = list(d["energies"])
