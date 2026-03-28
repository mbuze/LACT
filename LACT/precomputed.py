"""Drop-in replacement for atom_cont_system that loads pre-computed data."""

import sys
import io
import numpy as np


def _load_npz(path_or_url):
    """Load an .npz file from a local path or URL (Pyodide-compatible)."""
    if "pyodide" in sys.modules:
        from pyodide.http import open_url  # noqa: F811

        buf = open_url(path_or_url)
        return np.load(io.BytesIO(buf.read()), allow_pickle=False)
    return np.load(path_or_url, allow_pickle=False)


class PrecomputedSystem:
    """Lightweight stand-in for ``atom_cont_system`` using serialized data.

    Exposes the same attributes that the visualization cells in the example
    notebooks read: ``natoms``, ``U_0``, and ``data`` (with keys ``"Y_s"``,
    ``"energies"``, ``"ds_s"``).
    """

    def __init__(self, path_or_url):
        d = _load_npz(path_or_url)
        self.natoms = int(d["natoms"])
        self.U_0 = d["U_0"]
        self.data = {
            "Y_s": list(d["Y_s"]),
            "ds_s": list(d["ds_s"]) if "ds_s" in d else [],
        }
        if "energies" in d:
            self.data["energies"] = list(d["energies"])


def save_system(sys_obj, path):
    """Serialize an ``atom_cont_system`` to an ``.npz`` file."""
    arrays = {
        "natoms": np.array(sys_obj.natoms),
        "U_0": sys_obj.U_0,
        "Y_s": np.array(sys_obj.data["Y_s"]),
    }
    if sys_obj.data.get("ds_s"):
        arrays["ds_s"] = np.array(sys_obj.data["ds_s"])
    if sys_obj.data.get("energies"):
        arrays["energies"] = np.array(sys_obj.data["energies"])
    np.savez_compressed(path, **arrays)
