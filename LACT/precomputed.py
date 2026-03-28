"""Drop-in replacement for atom_cont_system that loads pre-computed data."""

import json
import numpy as np


class PrecomputedSystem:
    """Lightweight stand-in for ``atom_cont_system`` using serialized data.

    Exposes the same attributes that the visualization cells in the example
    notebooks read: ``natoms``, ``U_0``, and ``data`` (with keys ``"Y_s"``,
    ``"energies"``, ``"ds_s"``).

    Parameters
    ----------
    d : dict
        Parsed JSON data with keys ``"natoms"``, ``"U_0"``, ``"Y_s"``,
        and optionally ``"energies"`` and ``"ds_s"``.
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


def save_system(sys_obj, path):
    """Serialize an ``atom_cont_system`` to a JSON file."""
    d = {
        "natoms": int(sys_obj.natoms),
        "U_0": np.round(sys_obj.U_0, 12).tolist(),
        "Y_s": [np.round(y, 12).tolist() for y in sys_obj.data["Y_s"]],
    }
    if sys_obj.data.get("ds_s"):
        d["ds_s"] = [round(v, 12) for v in sys_obj.data["ds_s"]]
    if sys_obj.data.get("energies"):
        d["energies"] = [round(v, 12) for v in sys_obj.data["energies"]]
    with open(path, "w") as f:
        json.dump(d, f)
