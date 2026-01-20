import numpy as np
try:
    import mpi4py.MPI as MPI
    parallel = True
except ImportError:
    print("mpi4py not found, serial runs only!")
    MPI = None
    parallel = False
    pass

def fix_periodicity(X,box_size,show=False):
    """ Transform an atomistic configuration X 
        to ensure each atom lies inside the supercell. 
        If any atom lies otuside the box, its position is shifted 
        by the box size. Here X is an n dimensional array 
        of d dimensional vectors.
    """
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
    """ Transform an atomistic configuration X 
        to ensure each atom lies inside the supercell. 
        If any atom lies otuside the box, its position is 
        shifted by the box size. Here X is a flat dN vector. 
    """
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
    """ Transform an atomistic configuration *difference* X to ensure 
        each difference spans at most half of the simulation box in each dimension 
        Here X is an n dimensional array of d dimensional vectors.
    """
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
    """ Transform an atomistic configuration *difference* X
        to ensure each difference spans at most half 
        of the simulation box in each dimension Here X is a flat dN vector.
    """
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


def extract_comp_parallel(comm, lmp, name, lmp_style, lmp_type, natoms, dtype='float64'):
    """ Extract compute data in parallel"""
    dtypes = {'float64': (MPI.DOUBLE, np.float64), 'int32': (MPI.INT32_T, np.int32)}

    local_v = lmp.numpy.extract_compute(name, lmp_style, lmp_type).astype(dtypes[dtype][1])
    local_v_flat = local_v.ravel()
    local_count = np.array([local_v_flat.size], dtype=np.int32)

    # Gather sizes from all ranks
    counts = np.empty(comm.size, dtype=np.int32)
    comm.Allgather(local_count, counts)

    displacements = np.insert(np.cumsum(counts[:-1]), 0, 0)
    global_flat = np.empty(counts.sum(), dtype=dtypes[dtype][1])

    # Gather all data into the global array
    comm.Allgatherv(local_v_flat, [global_flat, counts, displacements, dtypes[dtype][0]])

    # Restore original shape
    ndim = 1 if local_v.ndim == 1 else local_v.shape[1]
    if ndim > 1:
        global_v = global_flat.reshape(natoms, ndim)
    else:
        global_v = global_flat

    return global_v
    