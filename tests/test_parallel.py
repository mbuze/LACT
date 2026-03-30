# run a quasistatic simulation in parallel and in serial and test the results
from LACT import atom_cont_system

import numpy as np
import matplotlib.pyplot as plt

from lammps import lammps, LMP_STYLE_ATOM, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY
from ctypes import c_double, c_int

import scipy
from scipy import optimize
import mpi4py.MPI as MPI

comm_single = MPI.COMM_SELF
comm_world = MPI.COMM_WORLD

def quasi_static_run(comm):
    lmp_initialise = '''
    # ------------------------ INITIALIZATION ----------------------------
    processors    * * *
    units         metal
    dimension    3
    boundary    p    p    p
    atom_style   atomic
    atom_modify map yes


    box        tilt large
    #--------------------------- LAMMPS Data File--------------------------
    #read_data    input_data/proper_thick_domain.lmp
    #read_data    input_data/thicker_domain.lmp
    read_data    input_data/4b_step.lmp
    change_box    all triclinic
    reset_atom_ids sort yes

    #--------------
    pair_style    eam/alloy
    pair_coeff    * * input_data/Cu_mishin1.eam.alloy Cu

    #-------------------Various continuation commands----------------------
    compute forces all property/atom fx fy fz
    compute ids all property/atom id
    compute x_check all property/atom x y z
    #atom_modify map yes
    '''

    lmp = lammps(comm=comm, cmdargs=['-screen', 'none'])

    lmp.commands_string(lmp_initialise)


    def change_box_command(limits,direction="x"):
        command = f"""
        change_box all {direction} final {limits[0]} {limits[1]} units box
        """
        return command

    box_size = lmp.extract_box()
    def update_command_box_x(x):
        direction = "x"
        i = 0 if direction=="x" else 1 if direction=="y" else 2
        return change_box_command([box_size[0][i]+x,box_size[1][i]-x],direction=direction)

    system = atom_cont_system(lmp,update_command_box_x, comm=comm)

    print(system.natoms)

    system.quasi_static_run(0.0, 0.1, 2, verbose=True)

    return system


def continuation_run(system):
    ds_default = 2e0
    ds_smallest = 1e-3
    ds_largest = 4e0

    system.continuation_run(5,
                        ds_default = ds_default,
                        ds_smallest = ds_smallest,
                        ds_largest = ds_largest,
                        verbose=True)
    
    return system

print("Running in serial")
print("--------Quasi-static run test--------")
system_serial = quasi_static_run(comm_single)
Y_s_serial_quas = system_serial.data["Y_s"][-1]

print("--------Continuation run test--------")
system_serial = continuation_run(system_serial)
Y_s_serial_cont = system_serial.data["Y_s"][-1]



print("Running in parallel")
print("--------Quasi-static run test--------")
system_parallel = quasi_static_run(comm_world)
Y_s_parallel_quas = system_parallel.data["Y_s"][-1]

print("--------Continuation run test--------")
system_parallel = continuation_run(system_parallel)
Y_s_parallel_cont = system_parallel.data["Y_s"][-1]

assert np.allclose(Y_s_serial_quas, Y_s_parallel_quas), "Quasi-static run test failed"

assert np.allclose(Y_s_serial_cont, Y_s_parallel_cont, atol=1e-6), "Continuation run test failed"

print("---------PASS---------")

