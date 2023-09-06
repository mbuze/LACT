import numpy as np

def fix_periodicity(X,box_size,show=False):
    """ Transform an atomistic configuration X to ensure each atom lies inside the supercell. If any atom lies otuside the box, its position is shifted by the box size. Here X is an n dimensional array of d dimensional vectors.
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
    """ Transform an atomistic configuration X to ensure each atom lies inside the supercell. If any atom lies otuside the box, its position is shifted by the box size. Here X is a flat dN vector. 
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
    """ Transform an atomistic configuration *difference* X to ensure each difference spans at most half of the simulation box in each dimension Here X is an n dimensional array of d dimensional vectors.
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
    """ Transform an atomistic configuration *difference* X to ensure each difference spans at most half of the simulation box in each dimension Here X is a flat dN vector.
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