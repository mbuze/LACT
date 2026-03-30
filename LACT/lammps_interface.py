import numpy as np

from lammps import lammps, LMP_STYLE_ATOM, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY
from ctypes import c_double, c_int
import warnings

import scipy
from scipy import optimize
from scipy.sparse.linalg import LinearOperator, eigsh
from .utils import *

from matscipy.optimize import ode12r
from ase.optimize.sciopt import OptimizerConvergenceError

import matplotlib.pyplot as plt

try:
    from mpi4py import MPI
    parallel = True
except ImportError:
    print("mpi4py not found, serial runs only!")
    parallel = False
import os

class atom_cont_system:
    """
    An atomistic continuation system.
    """
    def __init__(self,
                 lmp,
                 update_command,
                 comm=None):
        self.lmp = lmp
        self.natoms = lmp.extract_global("natoms")
        self.ref_X = np.reshape(
            np.array(lmp.gather_atoms("x", 1, 3)),
            (self.natoms, 3),
            ).copy()
        self.data = {
             "Y_s": [],
            "ds_s": [],
        }
        self.initial_step = 0
        self.overrule_ds = None
        self.change_cont_param = lambda x : update_command(x)
        self.bond_changes = None
        self.on_saddle=False
        if parallel and comm is not None:
            rank = comm.Get_rank()
            size = comm.Get_size()
            self.rank = rank
            self.size = size
            self.comm = comm
        else:
            self.rank = 0
            self.size = 1
            self.comm = None

    def set_u0_and_μ0(self, U_0, mu_0): #image_arr_0
        """set initial states for continuation,
        U_0 is the initial atomic positions (N, 3) and 
        mu_0 is the initial continuation parameter"""
        if self.rank == 0:
            print("setting initial states...")
        self.U_0 = U_0
        self.μ_0 = mu_0
        self.image_arr_0 = np.zeros((self.natoms,3)) 
        #zeros because U_0 is the reference

    def reset_atoms_and_μ(self):
        """reset atom positions in LAMMPS to U_0 and μ to μ_0"""
        # if self.rank == 0:
        #     print("resetting atoms and μ...")
        self.lmp.commands_string(self.change_cont_param(self.μ_0))
        self.update_lammps_positions(self.U_0, self.image_arr_0)
        self.lmp.command('set group all image 0 0 0') #reset all the image IDS to 0

    def get_positions_from_lammps(self):
        X = np.reshape(
            np.array(self.lmp.gather_atoms("x", 1, 3)),
            (self.natoms, 3),
        ).copy()
        images = np.reshape(np.array(self.lmp.gather_atoms("image",0,3)),
                (self.natoms,3)).copy()
        self.image_arr = images
        cell = self.lmp.extract_box()
        boxlo = cell[0]
        boxhi = cell[1]
        xy = cell[2]
        yz = cell[3]
        xz = cell[4]
        a_vec = np.array([boxhi[0]-boxlo[0],0,0])
        b_vec = np.array([xy,boxhi[1]-boxlo[1],0])
        c_vec = np.array([xz,yz,boxhi[2]-boxlo[2]])
        vecs =  [a_vec,b_vec,c_vec]
        im_summed = np.sum(np.abs(images),axis=1)
        ids = np.where(im_summed > 0)[0]

        for i in ids:
            for j in range(3):
                X[i,:] += images[i,j]*vecs[j]
     
        return X, images
    
    def update_lammps_positions(self,X,image_arr):
        """update atom positions in LAMMPS to X,
           wrapping according to image"""
        cell = self.lmp.extract_box()
        boxlo = cell[0]
        boxhi = cell[1]
        xy = cell[2]
        yz = cell[3]
        xz = cell[4]
        a_vec = np.array([boxhi[0]-boxlo[0],0,0])
        b_vec = np.array([xy,boxhi[1]-boxlo[1],0])
        c_vec = np.array([xz,yz,boxhi[2]-boxlo[2]])
        vecs =  [a_vec,b_vec,c_vec]
        im_summed = np.sum(np.abs(image_arr),axis=1)
        ids = np.where(im_summed > 0)[0]
        for i in ids:
            for j in range(3):
                X[i,:] -= image_arr[i,j]*vecs[j]

        X_c = ((len(X.flatten()))*c_double)(*X.flatten())
        self.lmp.scatter_atoms("x", 1, 3, X_c)
        
    def add_correction_to_positions(self, Y):
        """add the correction to the atom positions"""
        # if self.rank == 0:
        #     print("adding correction...")
        X, image_arr = self.get_positions_from_lammps()
        X += Y[:-1].reshape(self.natoms,3)
        self.update_lammps_positions(X, image_arr)

    
    
    def minimize_to_saddle(self,μ,maxiter,ftol=1e-5):

        def evf_gradient(x,μ):
            """
            Compute modified gradient for eigenvector-following.
            Used for relaxing to unstable branch
            
            Parameters
            ----------
            x : ndarray
                flattened atomic positions (Y[:-1])
            """
            xcpy = x.flatten().copy()
            Y = np.append(xcpy,μ)
            g = -self.get_force_vector(Y)
            # if self.rank == 0:
            #     print("iteration max force component",np.max(np.abs(g)))
            self.temp_Y = Y
            self.prev_max_force = np.max(np.abs(g)) 

            vmin = neg_eigvec        
            g_par = np.dot(g, vmin) * vmin
            g_perp = g - g_par
            
            # Flip sign along lowest mode
            g_mod = g_perp - g_par
            return -g_mod
        
        Yin = self.data["Y_s"][-1].copy()
        Yin[-1] = μ
        self.temp_Y = Yin.copy()
        converged = False
        self.prev_max_force = 100
        print(f"{self.rank}:here3")
        if self.rank == 0:
            print("Starting ode12r minimisation onto saddle...")
        first_neg_eigvec = None
        for i in range(int(maxiter/40)):
            x0 = self.temp_Y[:-1]
            try:
                neg_eigvec = self.get_smallest_eigen(self.temp_Y)
                if first_neg_eigvec is not None:
                    if np.dot(first_neg_eigvec,neg_eigvec)<0:
                        neg_eigvec = -neg_eigvec
                    else:
                        first_neg_eigvec = neg_eigvec 
                res_x = ode12r(evf_gradient,x0,args=(μ,),verbose=0,fmax=ftol,steps=40)
                converged = True
                if self.rank == 0:
                    print(f"Optimisation successful after {i+1} eigensolves and {(i*50)+res_x[1]} ode12r iterations")
                _Y_final = np.append(res_x[0].copy(),μ)
            except OptimizerConvergenceError:
                if self.rank == 0:
                    print("Largest force component:", self.prev_max_force)
                continue
            break
        
        
        if converged:
            return _Y_final, converged
        else:
            return Yin, converged

        
    def quasi_static_run(self,μ_start,increment,n_iter,verbose=False,reset_u0=True,ftol=1e-8,on_saddle=False):
        if self.rank == 0:
            print('''
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            A quasi-static run: adjust μ and minimise in LAMMPS
            ''')
        if len(self.data["Y_s"]) > 0:
            if self.rank == 0:
                print("Warning: System contains some data already!")
        for k in range(n_iter):
            if k>0 or not reset_u0:
                #reset structure to initial state
                self.reset_atoms_and_μ()
            # increment continuation parameter
            μ = μ_start + k*increment
            # handle the on saddle case
            if on_saddle:
                # minimize using modified forces and ode12r
                maxiter = 1000
                _Y, conv = self.minimize_to_saddle(μ,maxiter,ftol=ftol)
                if not conv:
                    if self.rank == 0:
                        print(f"ode12r Failed to converge in {maxiter} steps")
                    return False
                else:
                    self.data["Y_s"] += [_Y]
                    continue

            # if not on saddle, normal minimisation
            cmd = self.change_cont_param(μ)
            self.lmp.commands_string(cmd)
            self.lmp.command('run 0')
            if k>0 or not reset_u0:
                # get positions after continuation shift
                _X_mu_only, image_arr = self.get_positions_from_lammps()
                # add previous minimisation to atom positions
                self.add_correction_to_positions(self.data["Y_s"][-1])
            
            # self.lmp.command("dump mindumpy all custom 10 min_dump.lammpstrj id type x y z ix iy iz fx fy fz")
            self.lmp.command('thermo 1')
            self.lmp.command('run 0')
            self.lmp.command(f'minimize 0 {ftol} 5000 5000')
            # self.lmp.command("undump mindumpy")
            if k == 0 and reset_u0:
                self.lmp.command('set group all image 0 0 0')
            _X, image_arr = self.get_positions_from_lammps()
            self.lmp.command('reset_timestep 0')
            if k == 0 and reset_u0:
                self.set_u0_and_μ0(_X,μ) #image_arr
                #initial atomistic corrector is 0
                _Y = np.append(np.zeros_like(_X.flatten()),μ)
            else:
                _Y = np.append((_X.flatten() - _X_mu_only.flatten()),μ)
            self.data["Y_s"] += [_Y]

            if self.rank == 0:
                print("final Y is ",_Y)
                print("abs(Y) max is ",np.max(np.abs(_Y)))
            
            if verbose:
                Ys = self.data["Y_s"]
                if self.rank == 0:
                    print("Iteration step: ",k+1," ",", Solution step: ",len(Ys)," ",", Continuation parameter: ", Ys[-1][-1])
                    print("--------------------------------------------------")
            
                
    
    def get_smallest_eigen(self,Yin,v0=None):
        #function which gets the smallest eigenvalue and vector of atomistic system
        F0 = self.get_force_vector(Yin)
        eps = 1e-5
        def Hv_matvec(v):
            v = v.flatten()
            v = np.append(v,0.0)
            Fp = self.get_force_vector((Yin + eps * v))
            # H v ≈ - (F(x+eps v) - F(x)) / eps
            return (-(Fp - F0) / eps)
        n = len(F0)
        Hlinop = LinearOperator((n, n), matvec=Hv_matvec, dtype=float)
        if self.rank == 0:
            print("entering eigensolve...")
        vals, vecs = eigsh(Hlinop, k=1, which='SA', tol=1e-6, maxiter=500)
        neg_eigval = vals[0]
        neg_eigvec = vecs[:,0]
        if self.rank==0:
            print("neg_eigval:", neg_eigval)
            print("neg_eigvec:", neg_eigvec)
        return neg_eigvec

    def pass_ext_variable_info(self,Y):
        self.reset_atoms_and_μ()
        self.lmp.commands_string(self.change_cont_param(Y[-1]))
        self.add_correction_to_positions(Y)
    
    def get_gradient_wrt_cont_param(self,idx=-1,bond_changes=None,verbose=False):
        """If we know the bond changes that the continuation event is meant to be exploring,
        we compute the gradient of the bond length with respect to the continuation parameter."""
        if bond_changes is None:
            return None

        # if we have a set of bond changes, then we compute the gradient of Y wrt cont param for each bond change
        if np.ndim(bond_changes) == 1:
            bond_changes = [bond_changes]

        bonds_on_saddle = []
        grads = []
        for bond_change in bond_changes:
            atom_1 = int(bond_change[0])
            atom_2 = int(bond_change[1])
            u0 = self.U_0
            Ys = self.data["Y_s"]
            d_eps = Ys[idx][-1] - Ys[idx-1][-1]
            Y_reshaped_1 = Ys[idx][:-1].reshape(self.natoms,3)
            Y_reshaped_2 = Ys[idx-1][:-1].reshape(self.natoms,3)

            x1_1 = u0[atom_1] + Y_reshaped_1[atom_1]
            x2_1 = u0[atom_2] + Y_reshaped_1[atom_2]
            x1_2 = u0[atom_1] + Y_reshaped_2[atom_1]
            x2_2 = u0[atom_2] + Y_reshaped_2[atom_2]
            vector_1 = correct_pbc_vector(self, x1_1, x2_1, np.zeros(self.natoms), np.zeros(self.natoms))
            vector_2 = correct_pbc_vector(self, x1_2, x2_2, np.zeros(self.natoms), np.zeros(self.natoms))
            diff_1 = np.linalg.norm(vector_1)
            diff_2 = np.linalg.norm(vector_2)
            d_diff = diff_1 - diff_2
            
            grad = d_diff / d_eps
            grads.append(grad)
            if self.rank == 0 and verbose:
                print("d_diff for atom pair", atom_1, atom_2, "is", d_diff)
                print("grad d_bond_length/deps for atom pair", atom_1, atom_2, "is", grad)
            if ((d_diff > 0.0) and (int(bond_change[3]) == 0)) and (grad < 0.0):
                bonds_on_saddle.append([atom_1, atom_2])
            elif ((d_diff < 0.0) and (int(bond_change[3]) == 1)) and (grad > 0.0):
                bonds_on_saddle.append([atom_1, atom_2])
        
        if self.rank == 0:
            if len(bonds_on_saddle) > 0:
                print(f"On saddle!")
                self.on_saddle=True
            else:
                print(f"Not on saddle!")
                self.on_saddle=False
        return grads
    
    def get_force_vector(self,Y):
        self.pass_ext_variable_info(Y)
        self.lmp.command('run 0')
        if self.size == 1:
            f_t = self.lmp.numpy.extract_compute('forces',LMP_STYLE_ATOM,LMP_TYPE_ARRAY)
            _IDS = self.lmp.numpy.extract_compute('ids',LMP_STYLE_ATOM,LMP_TYPE_VECTOR).astype('int32')
        else:
            f_t = extract_comp_parallel(self.comm, self.lmp, 'forces',LMP_STYLE_ATOM, LMP_TYPE_ARRAY, self.natoms)
            _IDS = extract_comp_parallel(self.comm, self.lmp, 'ids',LMP_STYLE_ATOM, LMP_TYPE_VECTOR, self.natoms, dtype='int32')

        f_t = f_t[np.argsort(_IDS)]
        G = f_t.flatten()
        return G
    
    def extended_system(self,Y,ds,Ydot):
        G = self.get_force_vector(Y)
        Ys = self.data["Y_s"]
        assert len(Ys) > 1
        Y0 = Ys[-1]
        YminusY0 = Y-Y0
        last_eqn = (YminusY0*Ydot).sum() - ds
        G = np.append(G,last_eqn)
        return G


    def continuation_step(self,ds,verbose = False,maxiter=6,fatol=1e-5):
        if verbose:
            if self.rank != 0:
                verbose = False
        Ys = self.data["Y_s"]
        Ydot = Ys[-1] - Ys[-2]
        Ydot = Ydot / np.linalg.norm(Ydot)
        Y_0 = Ys[-1] + ds*Ydot
        
        self.pass_ext_variable_info(Ys[-1])
        Y_1 = scipy.optimize.root(self.extended_system, Y_0,
                       args=(ds, Ydot),
                       method='krylov',
                       options={'disp': verbose,
                                'fatol': fatol,
                                'maxiter': maxiter,
                                'line_search': 'armijo' ,
                                'jac_options': {'inner_M': None, 
                                                'method': 'lgmres'}},
                       callback=None)

        return Y_1
            
        
    def continuation_run(self,n_iter,
                     ds_default = 1e-2,
                     ds_smallest = 1e-3,
                     ds_largest = 2e0,
                     verbose=True,
                     maxiter=6,
                     exit_on_turn=False,
                     checkpoint_freq=5,
                     checkpoint_path='./checkpoints',
                     min_steps=0,
                     fatol=1e-5,
                     cont_target=None,
                     target_tol=1e-5,
                     exit_callback=None
                    ):
        if self.rank == 0:
            print('''
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            A continuation run: solve extended system to find points on the solution path.
            ''')
        self.killed_by_callback = False
        self.converge_to_target = False
        self.cont_target = cont_target
        if cont_target is not None:
            # get cont param at start
            init_cont_param = self.data["Y_s"][-1][-1]
            # get difference between init and threshold
            diff_sign = np.sign(init_cont_param - cont_target)

        if checkpoint_freq > 0:
            self.clear_checkpoint(checkpoint_path)
        
        counter = 0
        accepted_steps = 0
        if self.overrule_ds is not None:
            if self.rank == 0:
                print('Overruling initial ds from checkpoint')
            ds = self.overrule_ds
        else:
            ds = ds_default
        for k in range(self.initial_step, n_iter):
            if cont_target is not None:
                if (np.sign(self.data["Y_s"][-1][-1] - cont_target) != diff_sign) or self.converge_to_target:
                    # passed target - check to see if within tolerance of target
                    if np.abs(self.data["Y_s"][-1][-1] - cont_target) < target_tol:
                        if self.rank == 0:
                            print(f"Reached target continuation parameter {cont_target} within tolerance {target_tol}. Stopping continuation.")
                        break
                    else:
                        # we have passed the target, turn converge_to_target on and flip diff_sign
                        if self.rank == 0 and not self.converge_to_target:
                            print(f"Passed target continuation parameter {cont_target}, now attempting to converge to it.")
                        self.converge_to_target = True
                        
            if checkpoint_freq > 0 and len(self.data['Y_s']) % checkpoint_freq == 0:
                self.write_checkpoint(checkpoint_path, checkpoint_freq)
            
            # if converge to target is on, then we need to adjust ds based on Ydot
            if self.converge_to_target:
                Ys = self.data["Y_s"]
                Ydot = Ys[-1] - Ys[-2]
                Ydot = Ydot / np.linalg.norm(Ydot)
                mu_diff = self.cont_target - Ys[-1][-1]
                ds_target = mu_diff/Ydot[-1]
                if np.abs(ds_target) < np.abs(ds):
                    ds = ds_target
                else:
                    if np.sign(ds_target) != np.sign(ds):
                        # we have passed the target, so we need to flip the sign of ds
                        ds = -ds
                if self.rank == 0:
                    print(f"Adjusting ds to {ds}.")

            Y_1 = self.continuation_step(ds,verbose=verbose,maxiter=maxiter,fatol=fatol)
            if Y_1.success == False:
                ds = ds/2
                if ds > ds_smallest:
                    if self.rank == 0:
                        print("ds halved, now equal to ", ds, ". We go 2 steps back.")
                else:
                    if self.rank == 0:
                        print("ds now below the threshold, equal to ", ds, ".  Aborting...")
                    break
                res_at = -2
                if len(self.data["Y_s"]) <= 3:
                    if self.rank == 0:
                        print("Not enough data to go back. Just reducing step size.")
                else:
                    self.data["Y_s"] = self.data["Y_s"][:res_at]
                    self.data["ds_s"] = self.data["ds_s"][:res_at]
                counter = 0
            else:
                self.data["Y_s"] += [Y_1.x]
                self.data["ds_s"] += [ds]
                self.get_gradient_wrt_cont_param(bond_changes=self.bond_changes)
                counter += 1
                if counter > 5 and ds < ds_largest:
                    ds = 2*ds
                    if self.rank == 0:
                        print("ds doubled, now equal to ", ds, ".")
                    counter = 0
                
                accepted_steps += 1

                Ys = self.data["Y_s"]
                if self.rank == 0:
                    print("Iteration step: ",k+1," ",", Solution step: ",len(Ys)," ",", Continuation parameter: ", Ys[-1][-1])
                if np.sign(Ys[-1][-1] - Ys[-2][-1])*np.sign(Ys[-2][-1] - Ys[-3][-1]) < 0.0:
                    if self.rank == 0:
                        print("turn, turn, turn")
                    if exit_on_turn and accepted_steps > min_steps:
                        break
                
                if exit_callback is not None and not self.converge_to_target:
                    kill = exit_callback(self)
                    if kill:
                        self.killed_by_callback = True
                        break
                    
                    
    def dump_data(self,path,file_name,replace=True,dump_stress=False):
        Ys = self.data["Y_s"]
        if self.rank == 0:
            if os.path.exists(path) == False:
                os.makedirs(path)
            elif os.path.exists(f'{path}/{file_name}.lammpstrj') and replace:
                print(f"Dump file {path}/{file_name}.lammpstrj exists, replacing...")
                os.remove(f'{path}/{file_name}.lammpstrj')
        stress_tensors = np.zeros((len(Ys),6))
        for i in range(len(Ys)):
            self.pass_ext_variable_info(Ys[i])
            self.lmp.command('run 0')
            self.lmp.command(f'write_dump all custom {path}/{file_name}.lammpstrj id type x y z ix iy iz fx fy fz modify append yes')
            if dump_stress:
                pxx = self.lmp.get_thermo("pxx")
                pyy = self.lmp.get_thermo("pyy")
                pzz = self.lmp.get_thermo("pzz")
                pxy = self.lmp.get_thermo("pxy")
                pxz = self.lmp.get_thermo("pxz")
                pyz = self.lmp.get_thermo("pyz")
                voigt_stress_tensor = -np.array([pxx,pyy,pzz,pyz,pxz,pxy])
                stress_tensors[i,:] = voigt_stress_tensor
        
        if dump_stress and self.rank == 0:
            if not replace:
                if os.path.exists(f"{path}/{file_name}_stresses.txt"):
                    prev_tensors = np.loadtxt(f"{path}/{file_name}_stresses.txt")
                    stress_tensors = np.vstack((prev_tensors,stress_tensors))

            np.savetxt(f"{path}/{file_name}_stresses.txt", stress_tensors, header="pxx,pyy,pzz,pyz,pxz,pxy")


    def compute_energies(self):
        Ys = self.data["Y_s"]
        E  = []
        for i in range(len(Ys)):
            self.pass_ext_variable_info(Ys[i])
            self.lmp.command('run 0')
            E += [self.lmp.get_thermo("pe")]
        self.data["energies"] = E

    def clear_checkpoint(self, path=None):
        if self.rank == 0:
            if os.path.exists(path):
                print(f"Checkpoint directory {path} exists, clearing checkpoints...")
                #delete all files starting with checkpoint
                for f in os.listdir(path):
                    if f.startswith('checkpoint'):
                        os.remove(os.path.join(path, f))

    def write_checkpoint(self, path='./checkpoints', data_points=None):
        if self.rank == 0:
            if not os.path.exists(path):
                os.makedirs(path)
            #save self.U0 and self.μ0
            np.savetxt(f'{path}/checkpoint_U0.txt',self.U_0)
            np.savetxt(f'{path}/checkpoint_μ0.txt',np.array([self.μ_0]))
            #save all the Ys
            Ys = np.array(self.data["Y_s"])
            if data_points is None:
                data_points = len(self.data["Y_s"])

            with open(f'{path}/checkpoint_Ys.txt', 'ab') as f:
                np.savetxt(f, Ys[-data_points:], fmt='%f', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
            #accumulate checkpoint metadata
            simulation_step = len(Ys)
            if len(self.data["ds_s"])>0:
                np.savetxt(f'{path}/checkpoint_metadata.txt',np.array([simulation_step, self.data['ds_s'][-1]]), header='simulation_step, ds')
            else:
                np.savetxt(f'{path}/checkpoint_metadata.txt',np.array([simulation_step, 0.0]), header='simulation_step, ds')


    def read_checkpoint(self, path):
        #read self.U0 and self.μ0
        U_0 = np.loadtxt(f'{path}/checkpoint_U0.txt')
        μ_0 = np.loadtxt(f'{path}/checkpoint_μ0.txt')
        self.set_u0_and_μ0(U_0, μ_0)
        #read all the Ys
        Ys = np.loadtxt(f'{path}/checkpoint_Ys.txt')
        Ys = list(Ys)
        self.data["Y_s"] = Ys
        #read checkpoint metadata
        metadata = np.loadtxt(f'{path}/checkpoint_metadata.txt')
        self.simulation_step = metadata[0]
        if metadata[1] != 0.0:
            self.overrule_ds = metadata[1]
            self.data["ds_s"].append(metadata[1])


class atom_cont_system_legacy:
    """
    A previous version of the code, kept for possible debugging.
    Notable changes compared to the main version:
    - Configuration based: variables are configurations and not deviations from some reference state
    - it can only handle simulations cells without tilts
    - handling periodic boundary conditions is ad-hoc (but seems to work well)
    """
    def __init__(self,lmp,update_command,comm=None):
        self.lmp = lmp
        self.org_box = lmp.extract_box()
        self.natoms = lmp.extract_global("natoms")
        self.ref_X = np.reshape(
            np.array(lmp.gather_atoms("x", 1, 3)),
            (self.natoms, 3),
        ).copy()
        self.data = {
            "Y_s": [],
            "ds_s": []
        }

        self.change_cont_param = lambda x : update_command(x,self.org_box)

        if parallel and comm is not None:
            rank = comm.Get_rank()
            size = comm.Get_size()
            self.rank = rank
            self.size = size
            self.comm = comm
        else:
            self.rank = 0
            self.size = 1
            self.comm = None
        
        
    def quasi_static_run(self,μ_start,increment,n_iter,verbose=False):
        if self.rank == 0:
            print('''
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            A quasi-static run: adjust μ and minimise in LAMMPS
            ''')
        if len(self.data["Y_s"]) > 0:
            if self.rank == 0:
                print("Warning: System contains some data already!")
        for k in range(n_iter):
            μ = μ_start + k*increment
            # if verbose:
            #     print(f'Iteration no: {k+1}')
            #     print(f'The continuation parameter is {μ}')
            self.lmp.commands_string(self.change_cont_param(μ))
            if k < 2:
                ref_X_flat = self.ref_X.flatten()
                ref_X_flat = ((3*self.natoms)*c_double)(*ref_X_flat)
                self.lmp.scatter_atoms("x", 1, 3, ref_X_flat)
            self.lmp.command('run 0')
            self.lmp.command('min_style cg')
            self.lmp.command('minimize 0 1e-5 3000 3000')
            _X = np.reshape(
                     np.array(self.lmp.gather_atoms("x", 1, 3)),
                     (self.natoms, 3),
                 ).copy()
            self.lmp.command('reset_timestep 0')
            _Y = np.append(_X.flatten(),μ)
            self.data["Y_s"] += [_Y]
            if verbose:
                Ys = self.data["Y_s"]
                if self.rank == 0:
                    print("Iteration step: ",k+1," ",", Solution step: ",len(Ys)," ",", Continuation parameter: ", Ys[-1][-1])
                
    
    def pass_ext_variable_info(self,Y):
        self.lmp.commands_string(self.change_cont_param(Y[-1]))
        box_size = self.lmp.extract_box()[:2]
        Y_x = Y[0:-1]
        fix_periodicity_flat(Y_x,box_size) 
        Y_x = ((len(Y_x))*c_double)(*Y_x)
        # synchronize the coordinates
        if parallel:
            MPI.COMM_WORLD.Barrier()
        self.lmp.scatter_atoms("x", 1, 3, Y_x)
    
    
    def extended_system(self,Y,ds):
        self.pass_ext_variable_info(Y)
        Ys = self.data["Y_s"]
        assert len(Ys) > 1
        Y0 = Ys[-1]
        Ydot = Ys[-1] - Ys[-2]
        box_size = self.lmp.extract_box()[:2]
        fix_periodicity_relative_flat(Ydot[:-1],box_size)
        Ydot = Ydot / np.linalg.norm(Ydot)

        self.lmp.command('run 0')
        if self.size == 1:
            f_t = self.lmp.numpy.extract_compute('forces',LMP_STYLE_ATOM,LMP_TYPE_ARRAY)
            _IDS = self.lmp.numpy.extract_compute('ids',LMP_STYLE_ATOM,LMP_TYPE_VECTOR).astype('int32')
        else:
            f_t = extract_comp_parallel(self.comm, self.lmp, 'forces',LMP_STYLE_ATOM, LMP_TYPE_ARRAY, self.natoms)
            _IDS = extract_comp_parallel(self.comm, self.lmp, 'ids',LMP_STYLE_ATOM, LMP_TYPE_VECTOR, self.natoms, dtype='int32')

        #print(f_t)
        #print(_IDS)
        f_t = f_t[np.argsort(_IDS)]
        G = f_t.flatten()
        YminusY0 = Y-Y0
        fix_periodicity_relative_flat(YminusY0[:-1],box_size)
        last_eqn = (YminusY0*Ydot).sum() - ds
        G = np.append(G,last_eqn)
        return G
    
    def continuation_step(self,ds,verbose = False,maxiter=6):
        if verbose:
            if self.rank != 0:
                verbose = False
        Ys = self.data["Y_s"]
        assert len(Ys) > 1
        self.pass_ext_variable_info(Ys[-1])
        Ydot = Ys[-1] - Ys[-2]
        box_size = self.lmp.extract_box()[:2]
        fix_periodicity_relative_flat(Ydot[:-1],box_size)
        Ydot = Ydot / np.linalg.norm(Ydot)
        Y_0 = Ys[-1] + ds*Ydot
        Y_1 = scipy.optimize.root(self.extended_system, Y_0,
                       args=(ds),
                       method='krylov',
                       options={'disp': verbose,
                                'fatol': 1e-4,
                                'maxiter': maxiter,
                                'line_search': 'armijo' ,
                                'jac_options': {'inner_M': None, 
                                                'method': 'lgmres'}},
                       callback=None)
        return Y_1
        #self.data["Y_s"] += [Y_1.x]
        #self.data["ds_s"] += [ds]
        
        
    def continuation_run(self,n_iter,
                     ds_default = 1e-2,
                     ds_smallest = 1e-3,
                     ds_largest = 2e0,
                     verbose=True,
                     maxiter=6
                    ):
        if self.rank == 0:
            print('''
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            A continuation run: solve extended system to find points on the solution path.
            ''')
        counter = 0
        ds = ds_default
        for k in range(n_iter):
            Y_1 = self.continuation_step(ds,verbose=verbose,maxiter=maxiter)
            if Y_1.success == False:
                ds = ds/2
                if ds > ds_smallest:
                    if self.rank == 0:
                        print("ds halved, now equal to ", ds, ". We go 2 steps back.")
                else:
                    if self.rank == 0:
                        print("ds now below the threshold, equal to ", ds, ".  Aborting...")
                    break
                res_at = -2
                self.data["Y_s"] = self.data["Y_s"][:res_at]
                counter = 0
            else:
                self.data["Y_s"] += [Y_1.x]
                self.data["ds_s"] += [ds]
                counter += 1
                if counter > 5 and ds < ds_largest:
                    ds = 2*ds
                    if self.rank == 0:
                        print("ds doubled, now equal to ", ds, ".")
                    counter = 0

                Ys = self.data["Y_s"]
                if self.rank == 0:
                    print("Iteration step: ",k+1," ",", Solution step: ",len(Ys)," ",", Continuation parameter: ", Ys[-1][-1])
                if np.sign(Ys[-1][-1] - Ys[-2][-1])*np.sign(Ys[-2][-1] - Ys[-3][-1]) < 0.0:
                    if self.rank == 0:
                        print("turn, turn, turn")
                    
                    
    def dump_data(self,path,file_name):
        Ys = self.data["Y_s"]
        for i in range(len(Ys)):
            self.pass_ext_variable_info(Ys[i])
            s1 = 'write_dump all custom '
            s2 = f'_{i} id type x y z xu yu zu'
            self.lmp.command(s1+path+file_name+s2)
            
            
    def compute_energies(self):
        Ys = self.data["Y_s"]
        E  = []
        for i in range(len(Ys)):
            self.pass_ext_variable_info(Ys[i])
            self.lmp.command('run 0')
            E += [self.lmp.get_thermo("pe")]
        self.data["energies"] = E

def correct_pbc_vector(system, x1, x2, image_arr1, image_arr2):
    """Get the vector between x1 and x2 corrected for PBC"""
    x1, x2 = np.array(x1), np.array(x2)
    cell = system.lmp.extract_box()
    boxlo = cell[0]
    boxhi = cell[1]
    xy = cell[2]
    yz = cell[3]
    xz = cell[4]
    a_vec = np.array([boxhi[0] - boxlo[0], 0, 0])
    b_vec = np.array([xy, boxhi[1] - boxlo[1], 0])
    c_vec = np.array([xz, yz, boxhi[2] - boxlo[2]])

    def unwrap(x, image):
        return x - image[0]*a_vec - image[1]*b_vec - image[2]*c_vec

    x1_unwrapped = unwrap(x1, image_arr1)
    x2_unwrapped = unwrap(x2, image_arr2)
    vector = x2_unwrapped - x1_unwrapped

    if np.linalg.norm(vector) > 15:
        images = [i*a_vec + j*b_vec + k*c_vec
                for i in [-1, 0, 1]
                for j in [-1, 0, 1]
                for k in [-1, 0, 1]]
        vectors = [x2_unwrapped + img - x1_unwrapped for img in images]
        norms = [np.linalg.norm(v) for v in vectors]

        min_idx = int(np.argmin(norms))
        if norms[min_idx] < 15:
            vector = vectors[min_idx]
        else:
            if system.rank == 0:
                warnings.warn("Bond atoms very far apart (> 15 Å), PBC adjustment ambiguous")
                print(f"Original vector was {vector}")
                print(f"Min vector is {vectors[min_idx]}, norm = {norms[min_idx]}")
                print("Assuming correct adjusted vector is min vector.")
            vector = vectors[min_idx]

    return vector