from scipy.sparse import csc_matrix
from sksparse.cholmod import cholesky

def sparse_sqrt(A):
    # Given positive semi definite square A, find a square R
    # such that R.T @ R = A
    decomp = cholesky(csc_matrix(A), beta=1e-12,
        ordering_method='natural')
    L,D = decomp.L_D()
    D.data[D.data<0.] = 0.
    return D.sqrt() @ L.T