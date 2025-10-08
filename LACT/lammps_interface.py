import numpy as np

from lammps import lammps, LMP_STYLE_ATOM, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY
from ctypes import c_double, c_int
# stupid change
import scipy
from scipy import optimize

from .utils import *

class atom_cont_system:
    def __init__(self,lmp,update_command):
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
        
        
    def quasi_static_run(self,μ_start,increment,n_iter,verbose=False):
        print('''
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        A quasi-static run: adjust μ and minimise in LAMMPS
        ''')
        if len(self.data["Y_s"]) > 0:
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
                print("Iteration step: ",k+1," ",", Solution step: ",len(Ys)," ",", Continuation parameter: ", Ys[-1][-1])
                
    
    def pass_ext_variable_info(self,Y):
        self.lmp.commands_string(self.change_cont_param(Y[-1]))
        box_size = self.lmp.extract_box()[:2]
        Y_x = Y[0:-1]
        fix_periodicity_flat(Y_x,box_size)
        Y_x = ((len(Y_x))*c_double)(*Y_x)
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
        f_t = self.lmp.numpy.extract_compute('forces',LMP_STYLE_ATOM,LMP_TYPE_ARRAY)
        _IDS = self.lmp.numpy.extract_compute('ids',LMP_STYLE_ATOM,LMP_TYPE_VECTOR).astype('int32')
        f_t = f_t[np.argsort(_IDS)]
        G = f_t.flatten()
        YminusY0 = Y-Y0
        fix_periodicity_relative_flat(YminusY0[:-1],box_size)
        last_eqn = (YminusY0*Ydot).sum() - ds
        G = np.append(G,last_eqn)
        return G
    
    def continuation_step(self,ds,verbose = False):
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
                                'maxiter': 6,
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
                    ):
        print('''
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        A continuation run: solve extended system to find points on the solution path.
        ''')
        counter = 0
        ds = ds_default
        for k in range(n_iter):
            Y_1 = self.continuation_step(ds,verbose=verbose)
            if Y_1.success == False:
                ds = ds/2
                if ds > ds_smallest:
                    print("ds halved, now equal to ", ds, ". We go 2 steps back.")
                else:
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
                    print("ds doubled, now equal to ", ds, ".")
                    counter = 0

                Ys = self.data["Y_s"]
                print("Iteration step: ",k+1," ",", Solution step: ",len(Ys)," ",", Continuation parameter: ", Ys[-1][-1])
                if np.sign(Ys[-1][-1] - Ys[-2][-1])*np.sign(Ys[-2][-1] - Ys[-3][-1]) < 0.0:
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

            


        


