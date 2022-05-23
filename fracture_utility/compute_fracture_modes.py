# Include existing libraries
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, kron, eye
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
# Libigl
import igl


# Local includes
from . explode_mesh import explode_mesh
from . conic_solve import conic_solve
from . massmatrix_tets import massmatrix_tets
from . sparse_sqrt import sparse_sqrt
from . tictoc import tic, toc


# @profile
def compute_fracture_modes(vertices,elements,parameters):
    # Takes as input an (unexploded) tetrahedral mesh and a number of modes, returns a matrix UU dim x #T by #parameters.num_modes with computed fracture modes.
    if parameters.verbose:
        print("Starting fracture mode computation")
        print("We will find ",parameters.num_modes," unique fracture modes")
        print("Our input (unexploded) mesh has ",vertices.shape[0]," vertices and ",elements.shape[0]," tetrahedra.")
        tic()

    # Step 1: Compute traditional Laplacian eigenmodes.
    blockdiag_kron = eye(parameters.d)
    laplacian_unexploded = igl.cotmatrix(vertices,elements)
    massmatrix_unexploded = kron(blockdiag_kron,massmatrix_tets(vertices,elements),format='csc')
    Q_unexploded = kron(blockdiag_kron,laplacian_unexploded,format='csc')
    # print(-Q_unexploded)
    # print(massmatrix_unexploded)
    # unexploded_Q_eigenvalues, unexploded_Q_eigenmodes = eigsh(-Q_unexploded,parameters.num_modes,massmatrix_unexploded,which='SM') # <- our bottleneck outside of conic solves
    unexploded_Q_eigenvalues, unexploded_Q_eigenmodes = eigsh(-Q_unexploded,parameters.num_modes,massmatrix_unexploded,which='LM', sigma=0)
    unexploded_Q_eigenmodes = np.real(unexploded_Q_eigenmodes)

    # Step 2: Explode mesh, get unexploded-to-exploded matrix, get discontinuity and exploded Laplacian matrices
    exploded_vertices, exploded_elements, discontinuity_matrix, unexploded_to_exploded_matrix, tet_to_vertex_matrix, tet_neighbors = explode_mesh(vertices,elements,num_quad=1)
    discontinuity_matrix_full = kron(parameters.omega*blockdiag_kron,discontinuity_matrix,format='coo')
    unexploded_to_exploded_matrix_full = kron(blockdiag_kron,unexploded_to_exploded_matrix,format='csc')
    tet_to_vertex_matrix_full = kron(blockdiag_kron,tet_to_vertex_matrix,format='csc')
    laplacian_exploded = igl.cotmatrix(exploded_vertices,exploded_elements)
    massmatrix_exploded = massmatrix_tets(exploded_vertices,exploded_elements)
    Q = kron(blockdiag_kron,laplacian_exploded,format='csc')
    R = coo_matrix(sparse_sqrt(-Q))
    M = kron(blockdiag_kron,massmatrix_exploded,format='csc')

   
    
    # Step 3: Solve iteratively to find all modes
    
    # Initialization
    UU = unexploded_to_exploded_matrix_full @ unexploded_Q_eigenmodes


    # We convert everything into per-tet quantities
    UU = tet_to_vertex_matrix_full.T @ UU
    Q = coo_matrix(tet_to_vertex_matrix_full.T @ Q @ tet_to_vertex_matrix_full)
    R = coo_matrix(sparse_sqrt(-Q))
    M = coo_matrix(tet_to_vertex_matrix_full.T @ M @ tet_to_vertex_matrix_full)
    discontinuity_matrix_full = coo_matrix(discontinuity_matrix_full @ tet_to_vertex_matrix_full)



    if parameters.verbose:
        t_before_modes = toc(silence=True)
        print("Building matrices before starting mode computation: ", round(t_before_modes,2)," seconds.")



    # "Outer" loop to find all modes
    Us = []
    ts = []
    labels_full = np.zeros((elements.shape[0],parameters.num_modes))
    for k in range(parameters.num_modes):
        if parameters.verbose:
            tic()
        iter_num = 0
        diff = 1.0
        c = UU[:,k] # initialize to exploded laplacian mode
        # "Inner" loop to find each mode
        while diff>parameters.tol and iter_num<parameters.max_iter:
            cprev = c
            # Solve conic problem
            Ui = conic_solve(discontinuity_matrix_full, M, Us, c, parameters.d)
            c = Ui / np.sqrt(np.dot(Ui, M @ Ui))
            diff = np.max(np.abs(c-cprev))
            iter_num = iter_num + 1
        # Now, identify pieces:
        tet_tet_distances = np.linalg.norm(np.reshape(c,(-1,parameters.d),order='F')[tet_neighbors[:,0],:] - np.reshape(c,(-1,parameters.d),order='F')[tet_neighbors[:,1],:],axis=1)
        actual_neighbors = tet_neighbors[(tet_tet_distances<0.1),:]

        tet_adjacency_matrix = csr_matrix((np.ones(actual_neighbors.shape[0]),(actual_neighbors[:,0],actual_neighbors[:,1])),shape=(exploded_elements.shape[0],exploded_elements.shape[0]),dtype=int)
        n_components,labels = connected_components(tet_adjacency_matrix)
        labels_full[:,k] = labels
        Us.append(c)
        UU[:,k] = c
        if parameters.verbose:
            t_mode = toc(silence=True)
            ts.append(t_mode)
            print("Computed unique mode number", k+1, "using", iter_num, "iterations and", round(t_mode, 3), "seconds. This mode breaks the shape into",n_components,"pieces.")
    if parameters.verbose:
        print("Average time per mode: ", round(sum(ts)/len(ts),3)," seconds")

    modes = UU

    # Placeholder return
    return exploded_vertices,exploded_elements,modes,labels_full,tet_to_vertex_matrix,tet_neighbors,M,unexploded_to_exploded_matrix