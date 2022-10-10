# Include existing libraries
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, kron, save_npz
from scipy.sparse.linalg import lsqr, spsolve
from scipy.sparse.csgraph import connected_components
# Libigl
import igl
from scipy.stats import multivariate_normal
import polyscope as ps
from .fracture_modes_parameters import fracture_modes_parameters
from .massmatrix_tets import massmatrix_tets
from .compute_fracture_modes import compute_fracture_modes
from .tictoc import tic,toc

import sys
import os
from gpytoolbox.copyleft import mesh_boolean

# TODO: CHECK I DIDN'T BREAK 3D MODES
# TODO: Write unit tests for all dimensions and boolean options

class fracture_modes:
    impact_precomputed = False
    impact_projected = False
    def __init__(self,vertices,elements):
        # Initialize this class with an n by 3 matrix of vertices and an n by 4 integer matrix of tet indeces
        self.vertices = vertices
        self.elements = elements

    def compute_modes(self,parameters = fracture_modes_parameters()):
        # This is just a call to compute_fracture_modes, saving all the information we will need for impact projection
        self.exploded_vertices,self.exploded_elements,self.modes, self.labels, self.tet_to_vertex_matrix,self.tet_neighbors,self.massmatrix,self.unexploded_to_exploded_matrix = compute_fracture_modes(self.vertices,self.elements,parameters)
        self.verbose = parameters.verbose

    def transfer_modes_to_3d(self):
        # Computing modes in 3D can be slow. One trick we can do for efficiency is compute the modes in 1D and then transfer them to 3D by taking every possible combination of every 1D mode in the x, y and z directions
        modes_3d = []
        labels_3d = []
        for k in range(self.modes.shape[1]):
            # We do the k=0 case differently, because we know these will be the x, y, z displacements.
            if k==0:
                label = self.labels[:,0] # should be all zeros
                for j in range(3):
                    mode_3d = np.zeros((3*self.elements.shape[0],1))
                    indeces = j*self.elements.shape[0] + np.linspace(0,self.elements.shape[0]-1,self.elements.shape[0],dtype=int)
                    mode_3d[indeces] = np.mean(self.modes[:,0])
                    modes_3d.append(mode_3d)
                    labels_3d.append(np.reshape(label,(-1,1)))
            else:
                # In this case, we can remove one degree of freedom because we know displacements will be in the span of the modes
                labels = self.labels[:,k]
                n_components = np.max(labels.astype(int))+1
                # Compute per-piece displacements and tet-to-piece indeces for each mode
                displacements = np.zeros(n_components)
                indeces = []
                for j in range(n_components):
                    displacements[j] = np.mean(self.modes[labels==j,k])
                    indeces.append(np.nonzero(labels==j))
                # This is how many 3D modes we'll get... (a lot!)
                multiplicity = 3**(n_components-1) 
                for i in range(multiplicity):
                    # We will do this in a cute way, by taking the ternary expansion of the number i and use each ternary digit to choose whether we displace the piece in the x, y or z directions.
                    mode_3d = np.zeros((3*self.elements.shape[0],1))
                    ter = ternary(i,n_components-1)
                    for j in range(n_components-1):
                        remainder = int(ter[j])
                        mode_3d[remainder*self.elements.shape[0] + indeces[j][0]] = displacements[j]
                    modes_3d.append(mode_3d)
                    labels_3d.append(np.reshape(labels,(-1,1)))
        # Stack everything into 3D modes
        modes_3d = np.hstack(modes_3d)
        labels_3d = np.hstack(labels_3d)
        # Overwrite our previous 1D modes with the current 3D ones:
        self.modes = modes_3d
        self.labels = labels_3d
        # We also need to repeat the mass matrix that we use later to make it 3D
        self.massmatrix = kron(eye(3),self.massmatrix)
        # Ta-dah! We have 3D modes :)
        # Please we have no proof that these are exactly the same modes as if you had computed the 3D modes directly. I *think* they are, but maybe they're not! 

    def impact_precomputation(self,v_fine = None, f_fine = None, wave_h = 1/30, upper_envelope = False, smoothing_lambda = 100):
        # This is not strictly part of the mode computation but it can be
        # precomputed to make the impact projection as fast as possible:
        tic()
        dim = self.modes.shape[0]//self.elements.shape[0] # mode dimension
        # Do the kronecker product by these matrices to replicate the "tile" behaviour in matlab and the "blockdiag" behaviour
        blockdiag_mat = eye(dim)
        repmat_mat = np.ones((dim,1))
        def ind2dim(I): # This will take anything indexing elements and make it index dim x elements
            J = [] 
            for d in range(dim):
                J.append(I + d*self.elements.shape[0])        
            return np.concatenate(J)
        
        

        
        # Tet-tet adjacency matrix
        tet_tet_adjacency_matrix = csr_matrix((np.ones(self.tet_neighbors.shape[0]),(self.tet_neighbors[:,0],self.tet_neighbors[:,1])),shape=(self.exploded_elements.shape[0],self.exploded_elements.shape[0]),dtype=int)
        # For efficienty, we will later store and do math on *per-piece* impacts, instead of per-tet. For this to work, we need to identify all the possible pieces that break off and mappings between tets and pieces.
        
        tet_tet_distances_rep = np.abs(self.modes[ind2dim(self.tet_neighbors[:,0]),:] - self.modes[ind2dim(self.tet_neighbors[:,1]),:]) # This is a dim x num_neighbor_pairs by num_modes matrix
        
        # Need to turn this into L2 distances per tet
        tet_tet_distances = np.zeros((self.tet_neighbors.shape[0],self.modes.shape[1]))
        for d in range(dim):
            indeces = d*self.tet_neighbors.shape[0] + np.linspace(0,self.tet_neighbors.shape[0]-1,self.tet_neighbors.shape[0],dtype=int)
            tet_tet_distances = tet_tet_distances + (tet_tet_distances_rep[indeces,:]**2.0)
        tet_tet_distances = np.sqrt(tet_tet_distances)

        # These are the tets that are together in every mode, which means that no impact projected onto our modes can separate them
        always_neighbors = self.tet_neighbors[np.all(tet_tet_distances<0.1,axis=1),:]
        # In this matrix, two tets are connected if they are always neighbors
        always_adjacency_matrix = csr_matrix((np.ones(always_neighbors.shape[0]),(always_neighbors[:,0],always_neighbors[:,1])),shape=(self.exploded_elements.shape[0],self.exploded_elements.shape[0]),dtype=int)
        # Taking connected components lets us know all the pieces that can break off, and tet-to-piece labeling
        
        n_total,self.all_modes_labels = connected_components(always_adjacency_matrix,directed=False)
        self.precomputed_num_pieces = n_total
        # ^ This lets us now build a piece_to_tet matrix mapping values in one to the other.
        I = np.linspace(0,self.elements.shape[0]-1,self.elements.shape[0])
        J = self.all_modes_labels
        self.piece_to_tet_matrix = csr_matrix((np.ones(I.shape[0]),(I,J)),shape=(self.elements.shape[0],self.precomputed_num_pieces),dtype=int)
        # Then, a piece adjacency graph
        piece_piece_adjacency_matrix = coo_matrix(((self.piece_to_tet_matrix.T @ tet_tet_adjacency_matrix @ self.piece_to_tet_matrix)>0).astype(int))
        self.piece_neighbors = np.vstack((np.array(piece_piece_adjacency_matrix.row),np.array(piece_piece_adjacency_matrix.col))).T

        # Also need the modes and labels defined at pieces
        self.piece_modes = np.zeros((dim*self.precomputed_num_pieces,self.modes.shape[1]))
        self.piece_labels = np.zeros((self.precomputed_num_pieces,self.modes.shape[1]))
        for k in range(self.modes.shape[1]):
            self.piece_modes[:,k] = lsqr(kron(blockdiag_mat,self.piece_to_tet_matrix),self.modes[:,k])[0]
            self.piece_labels[:,k] = np.rint(lsqr(self.piece_to_tet_matrix,self.labels[:,k])[0]).astype(int)
            # print(lsqr(self.piece_to_tet_matrix,self.labels[:,k])[0])
            # print(lsqr(self.piece_to_tet_matrix,self.labels[:,k])[0].astype(int))
            # print(np.rint(lsqr(self.piece_to_tet_matrix,self.labels[:,k])[0]).astype(int))
            
        self.piece_massmatrix = kron(blockdiag_mat,self.piece_to_tet_matrix.T) @ self.massmatrix @ kron(blockdiag_mat,self.piece_to_tet_matrix)




        #  This precomputation will allow us to approximate the propagation of any impact with the wave equation without a linear solve at runtime.
        # At runtime, we will project an impact u into the best-fit (LS) per-piece impact. So, we will do
        # piece_impact = (piece_to_tet' M piece_to_tet)^{-1} piece_to_tet' u
        # So let's define ^-------------  tet_to_piece  ----------------^
        self.tet_to_piece_matrix = spsolve((kron(blockdiag_mat,self.piece_to_tet_matrix.T) @ self.massmatrix @ kron(blockdiag_mat,self.piece_to_tet_matrix)),kron(blockdiag_mat,self.piece_to_tet_matrix.T))
        # Now, say we have a contact point t[i] at runtime and d is the vector with all zeros except on the i-th position (called "onehot" later). Then, what we'd want to make the impact vector is
        # u = C (M - hL)^{-1} M d
        #       ^--A--^
        self.A = massmatrix_tets(self.vertices,self.elements) - wave_h*igl.cotmatrix(self.vertices,self.elements)
        self.M = massmatrix_tets(self.vertices,self.elements)
        # (C blurs per-unexploded-vertex values into tets)
        self.C = 0.25*(self.tet_to_vertex_matrix.T @ self.unexploded_to_exploded_matrix)

        # But then the full runtime computation will be
        # piece_impact = tet_to_piece * C * A^{-1} * M * d
        # So we might as well call 
        # wave_piece_lsqr' = tet_to_piece * C  * A^{-1} * M
        self.wave_piece_lsqr = spsolve(kron(blockdiag_mat,self.A.T), kron(blockdiag_mat,self.C.T) @ self.massmatrix.T @ self.tet_to_piece_matrix.T)
        # and then we no longer have to do a solve at runtime
        # we only need to do
        # piece_impact = wave_piece_lsqr' M d


        # We also may want to use a Gaussian, instead of a wave equation, to blur our impact from the contact point to the rest of the shape. In case we want to do this, we pre-build a normal distribution (not sure if this is really necessary)
        self.rv = multivariate_normal([0.0,0.0,0.0], [[0.01, 0.0, 0.0], [0.0,0.01, 0.0],[0.0,0.0,0.01]])

        # So far, we have precomputed everything we need to answer the question "which pieces will our input mesh break into given an impact". But, often, our input mesh is not the mesh we want to break; rather, it is a cage of a finer mesh, and we want a broken version of the latter to be the output. In that case, what we'll need to precompute are the possible fracture pieces *of the fine mesh* as well as a piece-to-fine-mesh-vertex mapping

        # We will be appending to these to stack later
        running_n = 0 # for combining meshes
        fine_piece_vertices = []
        fine_piece_triangles = []
        Js = []

        if(v_fine is not None):
            # If we want to alleviate the effect of mesh dependency, we can use a post-facto smoothing combined with upper envelope extraction
            # This is unsupported now because we still need to port the upper envelope code to gpytoolbox.
            # if upper_envelope:
            # # We convert our per-tet labels into "material densities"
            #     LT_elements = np.zeros((self.elements.shape[0],self.precomputed_num_pieces))
            #     for i in range(self.precomputed_num_pieces):
            #         LT_elements[self.all_modes_labels==i,i] = 1.0
            #     # Convert per-tet material densities into per-vertex material densities
            #     LT = blur_onto_vertices(self.elements,LT_elements)
            #     # Smooth the densities     
            #     LT = spsolve(eye(self.vertices.shape[0]) - smoothing_lambda*igl.cotmatrix(self.vertices,self.elements),LT)
            #     # Extract upper envelopes
            #     u, g, l = gpytoolbox.upper_envelope(self.vertices,self.elements,LT)


            # All this loop is doing is convert each coarse mesh piece into a triangle mesh, intersect it by the fine mesh, save that as a fine mesh piece, and keep track of indeces to get an index-to-fine mapping
            for i in range(self.precomputed_num_pieces):  
                if upper_envelope:
                    if np.any(l[:,i]): # Sometimes upper envelope entirely removes a material
                        vi, ti = igl.remove_unreferenced(u,g[l[:,i],:])[:2]
                        fi = boundary_faces_fixed(ti)
                        fi = fi[:,[1,0,2]] #libigl uses different ordering!??
                    else:
                        vi = np.zeros((0,3))
                        fi = np.zeros((0,3),dtype=int)
                else:    
                    vi, ti = igl.remove_unreferenced(self.vertices,self.elements[self.all_modes_labels==i,:])[:2]
                    fi = boundary_faces_fixed(ti)
                    fi = fi[:,[1,0,2]] #libigl uses different ordering!??
                # This should be replaced by a call to igl.mesh_booleans once the official binding is published
                vi_fine, fi_fine = mesh_boolean(v_fine,f_fine.astype(np.int32),vi,fi.astype(np.int32),boolean_type='intersection')
                fine_piece_vertices.append(vi_fine.copy())
                fine_piece_triangles.append(fi_fine + running_n)
                running_n = running_n + vi_fine.shape[0]
                Js.append(i*np.ones(vi_fine.shape[0],dtype=int))
            self.fine_vertices = np.vstack(fine_piece_vertices)
            self.fine_triangles = np.vstack(fine_piece_triangles)
            J = np.concatenate(Js)
            I = np.linspace(0,self.fine_vertices.shape[0]-1,self.fine_vertices.shape[0],dtype=int)
            # These correspondences work just like the tet ones from before
            self.piece_to_fine_vertices_matrix = csr_matrix((np.ones(I.shape[0]),(I,J)),shape=(self.fine_vertices.shape[0],self.precomputed_num_pieces),dtype=int)
            self.fine_labels = np.zeros((self.fine_vertices.shape[0],self.modes.shape[1]))
            for k in range(self.modes.shape[1]):
                self.fine_labels[:,k] = self.piece_to_fine_vertices_matrix @ lsqr(self.piece_to_tet_matrix,self.labels[:,k])[0] # We don't really need this lsqr (self.labels is constant per piece), but this is not a bottleneck.
        else:
            self.fine_vertices = None
            self.fine_triangles = None

        # Store and print timing details
        self.t_impact_pre = round(toc(silence=True),5)
        if self.verbose:
            print("Impact precomputation: ", self.t_impact_pre," seconds. Will produce a maximum of",self.precomputed_num_pieces,"pieces.")
        # This is a boolean that we'll check before projecting an impact
        self.impact_precomputed = True


    def impact_projection(self,contact_point=None,threshold=0.02,wave=True,direction=np.array([1]),impact=None,project_on_modes=False,num_modes_used = None):
        if (num_modes_used is None):
            num_modes_used = self.modes.shape[1]
        # This is the code we will run on runtime, when an impact is detected. Anything that can be precomputed has been precomputed, we should only do what strictly needs impact details here for efficiency

        # Make sure we've populated all the precomputation stuff, otherwise populate it
        if not self.impact_precomputed:
            self.impact_precomputation()

        
        tic() # Start counting time!
        # We will *assume* that the dimension of the modes you computed and the impact you're giving this function matches. If you want a 3D impact, compute 3D modes, and same for 1D.
        dim = self.modes.shape[0]//self.elements.shape[0] # mode dimension
        # There are two ways in which you can provide an impact: with an impact vector or with a contact point and direction.
        if (impact is None):
            # If you gave us a contact point and direction, then
            assert(direction.shape[0]==dim)
            # We will build an impact vector that is the size of the input vertices, since that's what we assumed for the least squares precomputation stuff
            if wave:
                onehot = np.zeros(self.vertices.shape[0])
                onehot[np.linalg.norm(self.vertices - np.tile(contact_point,(self.vertices.shape[0],1)),axis=1)<0.05] = 1.0
                impact_1d = onehot
            else:
                # Propagate with a gaussian directly
                impact_1d = 1.0*self.rv.pdf(self.vertices[self.elements[:,0],:] - np.tile(np.reshape(contact_point,(1,3)),(self.elements.shape[0],1)))
            # Both these impact versions are effectively repeated accross dimensions, but weighed by the direction of the impact. Let's do this to get a num_verts x dim impact
            impacts_dim = []
            for d in range(dim):
                impacts_dim.append(direction[d]*impact_1d)
            self.impact = np.concatenate(impacts_dim)
        else:
            # If we are here, you gave us an impact vector directly
            # Let's check that its dimension matches
            assert(impact.shape[0]==dim*self.vertices.shape[0])
            # We obviously won't use wave propagation if you already gave us an impact
            wave = False

        # This is just to mimic MATLAB's blockdiag function later
        blockdiag_kron = eye(dim)

        # We use the information from our precomputation step to project the impact onto the best (LS) constant-per-piece impact
        if wave:
            #self.piece_impact = self.wave_piece_lsqr.T @ kron(blockdiag_kron,self.M) @ self.impact
            self.piece_impact = self.wave_piece_lsqr.T @ self.impact
        else:
            self.piece_impact = kron(blockdiag_kron,self.tet_to_piece_matrix @ self.massmatrix) @ self.impact

        
        # However, not all constant-per-piece impacts are actually spanned by our modes (pieces can be linked and only break if others do, etc.), so if we want to project directly onto our modes, we need to do this extra step (note everything is happening per-piece, so the complexity of this loop is O(num_pieces*num_modes), irrespective of mesh size)
        if project_on_modes:
            self.projected_impact = np.zeros((self.piece_impact.shape[0]))
            for k in range(num_modes_used):
                self.projected_impact = self.projected_impact + (self.piece_impact.T @ self.piece_massmatrix @ self.piece_modes[:,k])*self.piece_modes[:,k]
        else:
            # If we're happy with our least squares projection, that's also fine:
            self.projected_impact = self.piece_impact

        # Calculate the difference in displacements between neighboring pieces (we use the piece adjancency we precomputed)
        piece_distances = np.linalg.norm(np.reshape(self.projected_impact,(-1,dim),order='F')[self.piece_neighbors[:,0],:] - np.reshape(self.projected_impact,(-1,dim),order='F')[self.piece_neighbors[:,1],:],axis=1)
        # Use these distances and our threshold parameter to decide which pieces break off from which pieces
        piece_neighbors_after_impact = self.piece_neighbors[piece_distances<threshold,:]
        # Build an impact-dependent piece adjancency graph
        piece_piece_adjacency_after_impact = csr_matrix((np.ones(piece_neighbors_after_impact.shape[0]),(piece_neighbors_after_impact[:,0],piece_neighbors_after_impact[:,1])),shape=(self.precomputed_num_pieces,self.precomputed_num_pieces),dtype=int)
        debug_distances = csr_matrix((piece_distances,(self.piece_neighbors[:,0],self.piece_neighbors[:,1])),shape=(self.precomputed_num_pieces,self.precomputed_num_pieces))
        # Get connected components of adjacency graph to know per-piece labels
        self.n_pieces_after_impact,self.piece_labels_after_impact = connected_components(piece_piece_adjacency_after_impact,directed=False)
        

        # Strictly speaking, this finishes our impact computation: for each piece, we've decided whether it breaks off or not.
        self.impact_projected = True
        self.t_impact = round(toc(silence=True),5) # Save and print runtime
        if self.verbose:
            print("Impact projection: ", self.t_impact," seconds. Produced",self.n_pieces_after_impact, "pieces.")
        
        # Still there's a little more information we may want to gather outside of the strict impact projection to display our fracture.

        # Now that we know per-piece labels, we can transfer this labels to tets
        self.tet_labels_after_impact = self.piece_to_tet_matrix @ self.piece_labels_after_impact # O(tets)

        # We can also compute labels in the fine mesh, if we're using a cage
        if(self.fine_vertices is not None):
            self.fine_vertex_labels_after_impact = self.piece_to_fine_vertices_matrix @ self.piece_labels_after_impact

        # We may also want to save the impact so we can visualize it. Of course, if you gave us an impact vector, we already have that. If we used a Gaussian to blur the impact from the contact point, then we had to compute this impact for the projection step.        
        if wave:
            # But if we used the wave equation, we have never actually computed our wave-equation-blurred impact A^{-1} M onehot, since that would involve a linear solve (see precomputation). So, if we want to visualize the wave impact, we need to actually do that linear solve:
            #self.impact_vis = spsolve(kron(blockdiag_kron,self.A),kron(blockdiag_kron,self.M) @ self.impact)
            # u = C (M - hL)^{-1} M d
            self.impact_vis = spsolve(kron(blockdiag_kron,self.A),kron(blockdiag_kron,self.M) @ self.impact)
        else:
            self.impact_vis = self.impact.copy()

        # Make it n by dim so that it can easily be added to vertex positions
        self.impact_vis = np.reshape(self.impact_vis,(-1,dim),order='F')
    
    def write_generic_data_compressed(self,filename):
        write_file_name = os.path.join(filename,"compressed_mesh.obj")
        write_data_name = os.path.join(filename,"compressed_data.npz")
        igl.write_obj(write_file_name,self.fine_vertices,self.fine_triangles)
        save_npz(write_data_name, self.piece_to_fine_vertices_matrix)

    def write_segmented_output_compressed(self,filename = None):
        write_fracture_name = os.path.join(filename,"compressed_fracture.npy")
        np.save(write_fracture_name,self.piece_labels_after_impact)

    def write_segmented_modes_compressed(self,filename = None):
        for j in range(self.modes.shape[1]):
            new_dir = os.path.join(filename,"mode_"+str(j))
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            write_fracture_name = os.path.join(new_dir,"compressed_fracture.npy")
            mode_labels = self.piece_labels[:,j]
            np.save(write_fracture_name,mode_labels)

    def write_segmented_output(self,filename = None,pieces=False):
        # All this routine is doing is write the fractured output, as a triangle mesh with num_broken_pieces connected components, so you can load it into an animation in another software. If you gave our algorithm a fine mesh, it will write the fractured fine mesh directly.
        # What variables do I need for this:
        # General, per-object data:
        # self.fine_vertices, self.fine_triangles
        # self.piece_to_fine_vertices_matrix
        # Per-impact data:
        # self.piece_labels_after_impact

        assert(self.impact_projected)
        self.fine_vertex_labels_after_impact = self.piece_to_fine_vertices_matrix @ self.piece_labels_after_impact
        Vs = []
        Fs = []
        running_n = 0 # for combining meshes
        for i in range(self.n_pieces_after_impact):
                if (self.fine_vertices is not None):
                    tri_labels = self.fine_vertex_labels_after_impact[self.fine_triangles[:,0]]
                    if np.any(tri_labels==i):
                        vi, fi = igl.remove_unreferenced(self.fine_vertices,self.fine_triangles[tri_labels==i,:])[:2]
                    else:
                        continue
                else:
                    vi, ti = igl.remove_unreferenced(self.vertices,self.elements[self.tet_labels_after_impact==i,:])[:2]
                    fi = boundary_faces_fixed(ti)
                ui, I, J, _ = igl.remove_duplicate_vertices(vi,fi,1e-10)
                gi = J[fi]
                
                if pieces:
                    write_file_name = os.path.join(filename,"piece_" + str(i) + ".obj")
                    igl.write_obj(write_file_name,ui,gi)
                Vs.append(ui)
                Fs.append(gi + running_n)
                running_n = running_n + ui.shape[0]
        self.mesh_to_write_vertices = np.vstack(Vs)
        self.mesh_to_write_triangles = np.vstack(Fs)
        if (filename is not None):
            if (not pieces):
                igl.write_obj(filename,self.mesh_to_write_vertices,self.mesh_to_write_triangles)

    def write_segmented_modes(self,filename = None,pieces=False):

        for j in range(self.modes.shape[1]):
            Vs = []
            Fs = []
            self.fine_labels = self.piece_to_fine_vertices_matrix @ self.piece_labels
            if pieces:
                new_dir = os.path.join(filename,"mode_"+str(j))
                if not os.path.exists(new_dir):
                    os.mkdir(new_dir)
            running_n = 0 # for combining meshes
            self.fine_labels = self.fine_labels.astype(int)
            for i in range(np.max(self.fine_labels[:,j])+1): # 
                
                # Double check this loop limit
                if (self.fine_vertices is not None):
                    tri_labels = self.fine_labels[self.fine_triangles[:,0],j]
                    if np.any(tri_labels==i):
                        vi, fi = igl.remove_unreferenced(self.fine_vertices,self.fine_triangles[tri_labels==i,:])[:2]
                        
                    else:
                        continue
                ui, I, J, _ = igl.remove_duplicate_vertices(vi,fi,1e-10)
                gi = J[fi]
                if pieces:
                    write_file_name = os.path.join(new_dir,"piece_" + str(i) + ".obj")
                    igl.write_obj(write_file_name,ui,gi)
                Vs.append(ui)
                Fs.append(gi + running_n)
                running_n = running_n + ui.shape[0]
            self.mesh_to_write_vertices = np.vstack(Vs)
            self.mesh_to_write_triangles = np.vstack(Fs)
            if (filename is not None):
                if (not pieces):
                    igl.write_obj(filename + "_mode_" + str(j) + ".obj",self.mesh_to_write_vertices,self.mesh_to_write_triangles)

        
def boundary_faces_fixed(ti):
    ti = np.reshape(ti,(-1,4))
    return igl.boundary_facets(ti)

def blur_onto_vertices(F,f_vals):
    v_vals = np.zeros((np.max(F.astype(int)) + 1,f_vals.shape[1]))
    valences = np.zeros((np.max(F.astype(int)) + 1,1))
    vec_kron = np.ones((F.shape[1],1))
    for i in range(F.shape[0]):
        valences[F[i,:]] = valences[F[i,:]] + 1
        v_vals[F[i,:],:] = v_vals[F[i,:],:] + f_vals[i,:]
    return v_vals/np.tile(valences,(1,f_vals.shape[1]))

def ternary(n,m):
    # if n == 0:
    #     return '0'
    nums = []
    for it in range(m):
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))