import igl
from scipy.sparse import csc_matrix

def massmatrix_tets(V,T):
    # Libigl python binding of massmatrix does not work for tet meshes, so
    # this is a quick wrapper doing a lumped tet mass matrix
    vol = (igl.volume(V, T)/4.0).repeat(4)
    i = T.flatten()
    j = i

    M = csc_matrix((vol, (i, j)), (V.shape[0], V.shape[0]))
    return M