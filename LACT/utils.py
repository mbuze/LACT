import numpy as np

#### helper functions

### periodicity preservation
def fix_periodicity(X,box_size,show=False):
    kk = 0
    for ii in range(len(X)):
        for j in range(3):
            if X[ii][j] < box_size[0][j]:
                X[ii][j] += box_size[1][j] - box_size[0][j]
                kk+=1
            elif X[ii][j] > box_size[1][j]:
                X[ii][0] -= box_size[1][j] - box_size[0][j]
                kk+=1
    if show:
        print("Number of changed coordinates:",kk)
    
def fix_periodicity_flat(X,box_size,show=False):
    kk = 0
    for ii in range(len(X)):
        j = ii % 3
        if X[ii] < box_size[0][j]:
            X[ii] += box_size[1][j] - box_size[0][j]
            kk+=1
        elif X[ii] > box_size[1][j]:
            X[ii] -= box_size[1][j] - box_size[0][j]
            kk+=1
    if show:
        print("Number of changed coordinates:",kk)
        
def fix_periodicity_relative(X,box_size,show=False):
    kk = 0
    allowed_directions = (np.array(box_size[1]) - np.array(box_size[0]))/2
    for ii in range(len(X)):
        for j in range(3):
            if X[ii][j] < -allowed_directions[j]:
                X[ii][j] += 2*allowed_directions[j]
                kk+=1
            elif X[ii][j] > allowed_directions[j]:
                X[ii][0] -= 2*allowed_directions[j]
                kk+=1
    if show:
        print("Number of changed coordinates:",kk)
        
def fix_periodicity_relative_flat(X,box_size,show=False):
    kk = 0
    allowed_directions = (np.array(box_size[1]) - np.array(box_size[0]))/2
    for ii in range(len(X)):
        j = ii % 3
        if X[ii] < -allowed_directions[j]:
            X[ii] += 2*allowed_directions[j]
            kk+=1
        elif X[ii] > allowed_directions[j]:
            X[ii] -= 2*allowed_directions[j]
            kk+=1
        
    if show:
        print("Number of changed coordinates:",kk)
        
### LAMMPS interface:

# passing the flat dN+1 variable Y to LAMMPS
def pass_X(Y):
    lammps_commands = f"""
     change_box all x final {original_box_size[0][0]+Y[-1]} {original_box_size[1][0]-Y[-1]} units box
    #fix        relax all box/relax iso 0.0    
    """
    lmp.commands_string(lammps_commands)
    box_size = lmp.extract_box()[:2]
    Y_x = Y[0:-1]
    fix_periodicity_flat(Y_x,box_size)
    Y_x = ((len(Y_x))*c_double)(*Y_x)
    lmp.scatter_atoms("x", 1, 3, Y_x)

# extended system
def G(Y,Y0,Ydot,ds,verbose = False):
    pass_X(Y)
    lmp.command('run 0')
    f_t = lmp.numpy.extract_compute('forces',LMP_STYLE_ATOM,LMP_TYPE_ARRAY)
    _IDS = lmp.numpy.extract_compute('ids',LMP_STYLE_ATOM,LMP_TYPE_VECTOR).astype('int32')
    f_t = f_t[np.argsort(_IDS)]
    G0 = f_t.flatten()
    box_size = lmp.extract_box()[:2]
    YminusY0 = Y-Y0
    fix_periodicity_relative_flat(YminusY0[:-1],box_size)
    last_eqn = (YminusY0*Ydot).sum() - ds
    G0 = np.append(G0,last_eqn)
    if verbose:
        print(Y[-1])
        print(last_eqn)
        print(np.linalg.norm(G0,np.Inf))
        #lmp.command('write_dump all custom run_dump2 id type x y z xu yu zu')
    return G0
